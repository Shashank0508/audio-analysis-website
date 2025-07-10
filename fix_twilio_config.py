#!/usr/bin/env python3
"""
Script to update Twilio phone number configuration with correct webhook URLs
"""

import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def update_twilio_phone_config():
    """Update Twilio phone number configuration"""
    try:
        # Get credentials
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        webhook_base_url = os.getenv('WEBHOOK_BASE_URL')
        
        if not all([account_sid, auth_token, phone_number, webhook_base_url]):
            print("❌ Missing required environment variables")
            return False
            
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Get phone number SID
        phone_numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        
        if not phone_numbers:
            print(f"❌ Phone number {phone_number} not found")
            return False
            
        phone_number_sid = phone_numbers[0].sid
        
        # Update phone number configuration
        voice_url = f"{webhook_base_url}/webhook/call/{{CallSid}}"
        status_callback_url = f"{webhook_base_url}/webhook/status/{{CallSid}}"
        
        print(f"📞 Updating phone number: {phone_number}")
        print(f"🔗 Voice URL: {voice_url}")
        print(f"📊 Status URL: {status_callback_url}")
        
        # Update the phone number
        phone_number_resource = client.incoming_phone_numbers(phone_number_sid).update(
            voice_url=voice_url,
            voice_method='POST',
            status_callback=status_callback_url,
            status_callback_method='POST'
        )
        
        print("✅ Phone number configuration updated successfully!")
        print(f"   - Voice URL: {phone_number_resource.voice_url}")
        print(f"   - Status Callback: {phone_number_resource.status_callback}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating phone number configuration: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing Twilio Phone Number Configuration...")
    print("=" * 50)
    
    success = update_twilio_phone_config()
    
    if success:
        print("\n🎉 Configuration updated! Your calls should now work properly.")
        print("📞 Try making a test call now.")
    else:
        print("\n❌ Configuration update failed. Please check your credentials.") 