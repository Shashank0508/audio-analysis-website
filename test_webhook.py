#!/usr/bin/env python3
"""
Test webhook endpoints to identify the issue
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_webhook():
    """Test webhook endpoint"""
    webhook_base_url = os.getenv('WEBHOOK_BASE_URL', 'https://5e90b17ee2f4.ngrok-free.app')
    
    print(f"🔍 Testing webhook URL: {webhook_base_url}")
    
    # Test the main webhook endpoint
    test_url = f"{webhook_base_url}/webhook/call/TEST123"
    
    try:
        print(f"📞 Testing: {test_url}")
        
        # Add headers that Twilio would send
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'TwilioProxy/1.1',
            'X-Twilio-Signature': 'test-signature'
        }
        
        # Add form data that Twilio would send
        data = {
            'CallSid': 'TEST123',
            'From': '+919392723953',
            'To': '+18382594031',
            'CallStatus': 'in-progress'
        }
        
        response = requests.post(test_url, headers=headers, data=data, timeout=10)
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        print(f"📝 Response Body: {response.text[:500]}")
        
        if response.status_code == 200:
            print("🎉 Webhook is working!")
            return True
        else:
            print(f"❌ Webhook failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def test_ngrok_status():
    """Test if ngrok is running"""
    try:
        # Test ngrok status endpoint
        response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
        if response.status_code == 200:
            tunnels = response.json()
            print("🔗 Active ngrok tunnels:")
            for tunnel in tunnels.get('tunnels', []):
                print(f"   - {tunnel.get('public_url')} -> {tunnel.get('config', {}).get('addr')}")
            return True
        else:
            print("❌ ngrok API not accessible")
            return False
    except:
        print("❌ ngrok not running or not accessible")
        return False

if __name__ == "__main__":
    print("🔧 Testing Webhook Configuration...")
    print("=" * 50)
    
    # Test ngrok status
    ngrok_ok = test_ngrok_status()
    print()
    
    # Test webhook
    webhook_ok = test_webhook()
    
    print("\n" + "=" * 50)
    if ngrok_ok and webhook_ok:
        print("🎉 Everything looks good!")
    else:
        print("❌ Issues found:")
        if not ngrok_ok:
            print("   - ngrok is not running properly")
        if not webhook_ok:
            print("   - Webhook is not responding correctly") 