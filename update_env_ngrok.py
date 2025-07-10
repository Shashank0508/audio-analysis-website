#!/usr/bin/env python3
"""
Simple script to update .env file with ngrok URL
"""

import os
import requests

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
        print(f"Error getting ngrok URL: {e}")
        return None

def update_env_file(ngrok_url):
    """Update .env file with ngrok URL."""
    env_file = '.env'
    env_vars = {}
    
    # Read existing .env file
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    
    # Update with ngrok URL
    env_vars['NGROK_URL'] = ngrok_url
    env_vars['WEBHOOK_BASE_URL'] = ngrok_url
    
    # Write updated .env file
    with open(env_file, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ Updated {env_file} with ngrok URL: {ngrok_url}")

def main():
    """Main function."""
    print("🔧 Updating environment variables with ngrok URL...")
    
    ngrok_url = get_ngrok_url()
    if not ngrok_url:
        print("❌ Could not get ngrok URL. Make sure ngrok is running.")
        return
    
    print(f"📡 Found ngrok URL: {ngrok_url}")
    
    update_env_file(ngrok_url)
    
    print("\n🔗 Webhook endpoints:")
    print(f"  - Call webhook: {ngrok_url}/api/call/webhook")
    print(f"  - Status webhook: {ngrok_url}/api/call/status")
    print(f"  - SMS webhook: {ngrok_url}/api/sms/webhook")
    print(f"  - AWS Transcribe check: {ngrok_url}/api/aws/transcribe/check")
    
    print("\n💡 Next steps:")
    print("1. Copy the webhook URLs above")
    print("2. Go to Twilio Console → Phone Numbers → Manage → Active Numbers")
    print("3. Update your phone number's webhook configuration")
    print("4. Test the webhooks")

if __name__ == "__main__":
    main() 