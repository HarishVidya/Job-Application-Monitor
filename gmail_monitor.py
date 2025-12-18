import os
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pickle
import base64
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

# Google API Scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/spreadsheets'
]

class JobApplicationEmail(BaseModel):
    """Structure for job application email data"""
    is_job_application: bool = Field(description="Whether this email is related to a job application")
    company: Optional[str] = Field(description="Company name from the email")
    role: Optional[str] = Field(description="Job role/position mentioned")
    date_received: str = Field(description="Date the email was received")
    status: Optional[str] = Field(description="Status like 'received', 'interview', 'rejected', etc.")

class GmailJobMonitor:
    def __init__(self, gemini_api_key: str, spreadsheet_id: str, discord_webhook_url: str = None):
        """
        Initialize the Gmail Job Monitor
        
        Args:
            gemini_api_key: Your Google Gemini API key
            spreadsheet_id: ID of the Google Sheet to log data
        """
        self.gemini_api_key = gemini_api_key
        self.spreadsheet_id = spreadsheet_id
        self.discord_webhook_url = discord_webhook_url
        self.gmail_service = None
        self.sheets_service = None
        self.llm = None
        self.setup_services()
        
    def setup_services(self):
        """Setup Gmail, Sheets, and Gemini services"""
        creds = self.authenticate_google()
        self.gmail_service = build('gmail', 'v1', credentials=creds)
        self.sheets_service = build('sheets', 'v4', credentials=creds)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0
        )
        
    def authenticate_google(self):
        """Authenticate with Google APIs"""
        creds = None
        
        # Token file stores user's access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
                
        # If no valid credentials, let user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                
            # Save credentials for next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
                
        return creds
    
    def get_recent_emails(self, max_results=10):
        """Fetch recent emails from Gmail"""
        try:
            results = self.gmail_service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q='in:inbox -category:promotions -category:social newer_than:1d'  # Emails from last 7 days
            ).execute()
            
            messages = results.get('messages', [])
            return messages
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []
    
    def get_email_content(self, msg_id):
        """Get full email content"""
        try:
            message = self.gmail_service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            # Extract body
            body = self.extract_body(message['payload'])
            
            return {
                'id': msg_id,
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body
            }
        except Exception as e:
            print(f"Error getting email content: {e}")
            return None
    
    def extract_body(self, payload):
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    if data:
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
                        break
        elif 'body' in payload:
            data = payload['body'].get('data', '')
            if data:
                body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body[:2000]  # Limit to first 2000 chars
    
    def analyze_email_with_gemini(self, email_content):
        """Use Gemini to analyze if email is job-related and extract info"""
        parser = PydanticOutputParser(pydantic_object=JobApplicationEmail)
        
        prompt = PromptTemplate(
            template="""You are a helpfuly assistant who is going to extract emails pertaining to replies about submitted job applications. Analyze the following email and determine if it's related to a submitted job application to a role in software development. 
                        If it is, extract the company name, job role, and application status.

Email Subject: {subject}
Email From: {sender}
Email Date: {date}
Email Body: {body}

{format_instructions}

Provide your analysis:""",
            input_variables=["subject", "sender", "date", "body"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "subject": email_content['subject'],
                "sender": email_content['sender'],
                "date": email_content['date'],
                "body": email_content['body']
            })
            return result
        except Exception as e:
            print(f"Error analyzing email: {e}")
            return None
    
    def log_to_sheets(self, job_data: JobApplicationEmail):
        """Log job application data to Google Sheets"""
        try:
            # Prepare row data
            values = [[
                job_data.company or "Unknown",
                job_data.role or "Unknown",
                job_data.date_received,
                job_data.status or "Received",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]]
            
            body = {
                'values': values
            }
            
            # Append to sheet
            self.sheets_service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Sheet1!A:E',
                valueInputOption='RAW',
                body=body
            ).execute()
            
            print(f"✓ Logged: {job_data.company} - {job_data.role}")
            
        except Exception as e:
            print(f"Error logging to sheets: {e}")
    
    def initialize_sheet(self):
        """Initialize sheet with headers if needed"""
        try:
            # Add headers
            values = [['Company', 'Role', 'Date Received', 'Status', 'Logged At']]
            body = {'values': values}
            
            self.sheets_service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range='Sheet1!A1:E1',
                valueInputOption='RAW',
                body=body
            ).execute()
            
            print("Sheet initialized with headers")
        except Exception as e:
            print(f"Note: {e}")
            
    def send_discord_notification(self, job_applications: list[JobApplicationEmail]):
        """Send Discord notification with job application findings"""
        if not self.discord_webhook_url or not job_applications:
            return
        
        try:
            # Create embed for Discord
            if len(job_applications) == 1:
                title = "🎯 Found 1 New Job Application Email!"
                color = 3447003  # Blue
            else:
                title = f"🎯 Found {len(job_applications)} New Job Application Emails!"
                color = 3066993  # Green
            
            # Build description with job details
            description = ""
            for i, job in enumerate(job_applications, 1):
                description += f"\n**{i}. {job.company or 'Unknown Company'}**\n"
                if job.role:
                    description += f"📋 Role: {job.role}\n"
                if job.status:
                    description += f"📊 Status: {job.status}\n"
                description += f"📅 Received: {job.date_received}\n"
            
            # Create Discord embed
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {
                    "text": "Job Application Tracker"
                }
            }
            
            # Send to Discord
            payload = {
                "embeds": [embed]
            }
            
            response = requests.post(self.discord_webhook_url, json=payload)
            
            if response.status_code == 204:
                print("✅ Discord notification sent successfully!")
            else:
                print(f"⚠️ Discord notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"Error sending Discord notification: {e}")
    
    def send_discord_summary(self, total_checked: int, jobs_found: int):
        """Send summary notification to Discord"""
        if not self.discord_webhook_url:
            return
        
        try:
            if jobs_found == 0:
                title = "✅ Monitoring Complete"
                description = f"Checked {total_checked} emails - No new job applications found."
                color = 10070709  # Gray
            else:
                title = "✅ Monitoring Complete"
                description = f"Checked {total_checked} emails and logged {jobs_found} job application(s) to Google Sheets!"
                color = 5763719  # Green
            
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {
                    "text": "Job Application Tracker"
                }
            }
            
            payload = {"embeds": [embed]}
            requests.post(self.discord_webhook_url, json=payload)
            
        except Exception as e:
            print(f"Error sending Discord summary: {e}")
        """Initialize sheet with headers if needed"""
        try:
            # Add headers
            values = [['Company', 'Role', 'Date Received', 'Status', 'Logged At']]
            body = {'values': values}
            
            self.sheets_service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range='Sheet1!A1:E1',
                valueInputOption='RAW',
                body=body
            ).execute()
            
            print("Sheet initialized with headers")
        except Exception as e:
            print(f"Note: {e}")
    
    
    def run_monitor(self, max_emails=10):
        """Main monitoring function"""
        print("🔍 Starting Gmail Job Application Monitor...")
        print(f"Checking last {max_emails} emails...\n")
        
        # Get recent emails
        emails = self.get_recent_emails(max_emails)
        
        if not emails:
            print("No emails found.")
            return
        
        job_count = 0
        
        for email in emails:
            email_content = self.get_email_content(email['id'])
            
            if email_content:
                print(f"Analyzing: {email_content['subject'][:50]}...")
                
                # Analyze with Gemini
                analysis = self.analyze_email_with_gemini(email_content)
                
                if analysis and analysis.is_job_application:
                    job_count += 1
                    self.log_to_sheets(analysis)
                    
        self.send_discord_summary(max_emails, job_count)
        print(f"\n✅ Monitoring complete! Found {job_count} job-related emails.")


def main():
    """Main execution function"""
    # Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your API key
    SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")  # Replace with your Sheet ID
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    if not SPREADSHEET_ID:
        raise ValueError("SPREADSHEET_ID environment variable not set")
    if not DISCORD_WEBHOOK_URL:
        raise ValueError("Discord environment variable not set")
    
    # Create monitor instance
    monitor = GmailJobMonitor(
        gemini_api_key=GEMINI_API_KEY,
        spreadsheet_id=SPREADSHEET_ID,
        discord_webhook_url=DISCORD_WEBHOOK_URL
    )
    
    # Initialize sheet with headers (run once)
    monitor.initialize_sheet()
    
    # Run the monitor
    monitor.run_monitor(max_emails=10)


if __name__ == "__main__":
    main()