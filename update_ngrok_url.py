#!/usr/bin/env python3
"""
Update ngrok URL in .env file and Twilio configuration
"""

import os
import requests
from twilio.rest import Client
from dotenv import load_dotenv

def get_ngrok_url():
    """Get current ngrok URL from local API"""
    try:
        response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
        if response.status_code == 200:
            tunnels = response.json()
            for tunnel in tunnels.get('tunnels', []):
                if tunnel.get('proto') == 'https':
                    return tunnel.get('public_url')
        return None
    except:
        return None

def update_env_file(new_url):
    """Update .env file with new ngrok URL"""
    try:
        env_file = '.env'
        
        # Read current .env file
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update the WEBHOOK_BASE_URL line
        updated_lines = []
        for line in lines:
            if line.startswith('WEBHOOK_BASE_URL='):
                updated_lines.append(f'WEBHOOK_BASE_URL={new_url}\n')
            elif line.startswith('NGROK_URL='):
                updated_lines.append(f'NGROK_URL={new_url}\n')
            else:
                updated_lines.append(line)
        
        # Write back to .env file
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        
        print(f"✅ Updated .env file with new URL: {new_url}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating .env file: {str(e)}")
        return False

def update_twilio_config(new_url):
    """Update Twilio phone number configuration"""
    try:
        # Reload environment variables
        load_dotenv(override=True)
        
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        if not all([account_sid, auth_token, phone_number]):
            print("❌ Missing Twilio credentials")
            return False
        
        client = Client(account_sid, auth_token)
        
        # Get phone number SID
        phone_numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not phone_numbers:
            print(f"❌ Phone number {phone_number} not found")
            return False
        
        phone_number_sid = phone_numbers[0].sid
        
        # Update URLs
        voice_url = f"{new_url}/webhook/call/{{CallSid}}"
        status_callback_url = f"{new_url}/webhook/status/{{CallSid}}"
        
        # Update phone number
        client.incoming_phone_numbers(phone_number_sid).update(
            voice_url=voice_url,
            voice_method='POST',
            status_callback=status_callback_url,
            status_callback_method='POST'
        )
        
        print(f"✅ Updated Twilio configuration:")
        print(f"   - Voice URL: {voice_url}")
        print(f"   - Status URL: {status_callback_url}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating Twilio config: {str(e)}")
        return False

def main():
    print("🔧 Updating ngrok URL Configuration...")
    print("=" * 50)
    
    # Get current ngrok URL
    current_url = get_ngrok_url()
    
    if not current_url:
        print("❌ No active ngrok tunnel found!")
        print("Please start ngrok with: ngrok http 5000")
        return
    
    print(f"🔗 Found ngrok URL: {current_url}")
    
    # Update .env file
    env_updated = update_env_file(current_url)
    
    # Update Twilio configuration
    twilio_updated = update_twilio_config(current_url)
    
    print("\n" + "=" * 50)
    if env_updated and twilio_updated:
        print("🎉 All configurations updated successfully!")
        print("📞 Your calls should now work properly.")
        print("🔄 Please restart your Flask app to load the new configuration.")
    else:
        print("❌ Some configurations failed to update.")

if __name__ == "__main__":
    main() 