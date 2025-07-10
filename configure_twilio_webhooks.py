#!/usr/bin/env python3
"""
Configure Twilio Webhooks Script
This script automatically configures Twilio phone number webhooks with ngrok URLs.
"""

import os
import sys
import requests
from urllib.parse import urljoin

def load_env_file():
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print("❌ .env file not found. Please create one with your Twilio credentials.")
        return None
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
    
    return env_vars

def get_ngrok_url():
    """Get the current ngrok tunnel URL."""
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        response.raise_for_status()
        
        tunnels_data = response.json()
        
        for tunnel in tunnels_data.get('tunnels', []):
            public_url = tunnel.get('public_url', '')
            if public_url.startswith('https://'):
                return public_url
        
        return None
    except Exception as e:
        print(f"❌ Error getting ngrok URL: {e}")
        print("Make sure ngrok is running on port 4040")
        return None

def configure_twilio_webhooks(account_sid, auth_token, phone_number, base_url):
    """Configure Twilio webhooks with ngrok URLs."""
    try:
        # Import Twilio client
        try:
            from twilio.rest import Client
        except ImportError:
            print("❌ Twilio library not installed. Installing...")
            os.system("pip install twilio")
            from twilio.rest import Client
        
        client = Client(account_sid, auth_token)
        
        # Configure webhook URLs
        webhook_urls = {
            'voice_url': urljoin(base_url, '/api/call/webhook'),
            'voice_method': 'POST',
            'status_callback': urljoin(base_url, '/api/call/status'),
            'status_callback_method': 'POST',
            'sms_url': urljoin(base_url, '/api/sms/webhook'),
            'sms_method': 'POST'
        }
        
        print(f"🔧 Configuring webhooks for phone number: {phone_number}")
        print(f"📡 Base URL: {base_url}")
        
        # Find and update phone number
        phone_numbers = client.incoming_phone_numbers.list()
        
        target_number = None
        for number in phone_numbers:
            if number.phone_number == phone_number:
                target_number = number
                break
        
        if not target_number:
            print(f"❌ Phone number {phone_number} not found in your Twilio account")
            print("Available phone numbers:")
            for number in phone_numbers:
                print(f"  - {number.phone_number}")
            return False
        
        # Update webhook configuration
        target_number.update(**webhook_urls)
        
        print("✅ Twilio webhooks configured successfully!")
        print("\n🔗 Configured webhook URLs:")
        for key, url in webhook_urls.items():
            if key.endswith('_url'):
                print(f"  - {key}: {url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to configure Twilio webhooks: {e}")
        return False

def verify_webhooks(account_sid, auth_token, phone_number):
    """Verify current webhook configuration."""
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        
        phone_numbers = client.incoming_phone_numbers.list()
        
        for number in phone_numbers:
            if number.phone_number == phone_number:
                print(f"\n📋 Current webhook configuration for {phone_number}:")
                print(f"  - Voice URL: {number.voice_url}")
                print(f"  - Voice Method: {number.voice_method}")
                print(f"  - Status Callback: {number.status_callback}")
                print(f"  - Status Callback Method: {number.status_callback_method}")
                print(f"  - SMS URL: {number.sms_url}")
                print(f"  - SMS Method: {number.sms_method}")
                return True
        
        print(f"❌ Phone number {phone_number} not found")
        return False
        
    except Exception as e:
        print(f"❌ Failed to verify webhooks: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Configuring Twilio Webhooks with ngrok URLs...")
    print("=" * 50)
    
    # Load environment variables
    env_vars = load_env_file()
    if not env_vars:
        return
    
    # Get required credentials
    account_sid = env_vars.get('TWILIO_ACCOUNT_SID')
    auth_token = env_vars.get('TWILIO_AUTH_TOKEN')
    phone_number = env_vars.get('TWILIO_PHONE_NUMBER')
    
    if not all([account_sid, auth_token, phone_number]):
        print("❌ Missing Twilio credentials in .env file. Required:")
        print("  - TWILIO_ACCOUNT_SID")
        print("  - TWILIO_AUTH_TOKEN")
        print("  - TWILIO_PHONE_NUMBER")
        return
    
    # Get ngrok URL
    ngrok_url = get_ngrok_url()
    if not ngrok_url:
        return
    
    print(f"📡 Using ngrok URL: {ngrok_url}")
    
    # Configure webhooks
    success = configure_twilio_webhooks(account_sid, auth_token, phone_number, ngrok_url)
    
    if success:
        print("\n🔍 Verifying configuration...")
        verify_webhooks(account_sid, auth_token, phone_number)
        
        print("\n✅ Webhook configuration completed!")
        print("\n💡 Next steps:")
        print("1. Start your Flask application: python app.py")
        print("2. Test by calling your Twilio number")
        print("3. Monitor requests at: http://localhost:4040")
        print("4. Check Flask logs for webhook processing")
    else:
        print("\n❌ Webhook configuration failed!")

if __name__ == "__main__":
    main() 