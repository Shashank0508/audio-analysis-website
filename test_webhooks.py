#!/usr/bin/env python3
"""
Test Webhook Delivery and Processing Script
This script tests webhook endpoints and validates their functionality.
"""

import os
import sys
import json
import time
import requests
from urllib.parse import urljoin

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
        return None

def test_webhook_endpoint(url, method='POST', data=None, headers=None):
    """Test a webhook endpoint."""
    try:
        if method.upper() == 'POST':
            response = requests.post(url, data=data, headers=headers, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        return {
            'status_code': response.status_code,
            'response_text': response.text[:200] if response.text else '',
            'headers': dict(response.headers),
            'success': response.status_code < 400
        }
    except Exception as e:
        return {
            'status_code': None,
            'response_text': str(e),
            'headers': {},
            'success': False
        }

def test_call_webhook(base_url):
    """Test call webhook with Twilio-like data."""
    url = urljoin(base_url, '/api/call/webhook')
    
    # Simulate Twilio call webhook data
    test_data = {
        'CallSid': 'CAtest123456789',
        'From': '+1234567890',
        'To': '+18382594031',
        'CallStatus': 'ringing',
        'Direction': 'inbound',
        'CallerName': 'Test Caller',
        'CallerCity': 'Test City',
        'CallerState': 'Test State',
        'CallerCountry': 'US',
        'CallerZip': '12345'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'TwilioProxy/1.1'
    }
    
    print(f"🧪 Testing call webhook: {url}")
    result = test_webhook_endpoint(url, 'POST', test_data, headers)
    
    print(f"   Status: {result['status_code']}")
    print(f"   Response: {result['response_text']}")
    print(f"   Success: {'✅' if result['success'] else '❌'}")
    
    return result

def test_status_webhook(base_url):
    """Test status webhook with Twilio-like data."""
    url = urljoin(base_url, '/api/call/status')
    
    # Simulate Twilio status webhook data
    test_data = {
        'CallSid': 'CAtest123456789',
        'CallStatus': 'completed',
        'CallDuration': '120',
        'RecordingUrl': 'https://api.twilio.com/test-recording.mp3',
        'RecordingSid': 'REtest123456789'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'TwilioProxy/1.1'
    }
    
    print(f"🧪 Testing status webhook: {url}")
    result = test_webhook_endpoint(url, 'POST', test_data, headers)
    
    print(f"   Status: {result['status_code']}")
    print(f"   Response: {result['response_text']}")
    print(f"   Success: {'✅' if result['success'] else '❌'}")
    
    return result

def test_sms_webhook(base_url):
    """Test SMS webhook with Twilio-like data."""
    url = urljoin(base_url, '/api/sms/webhook')
    
    # Simulate Twilio SMS webhook data
    test_data = {
        'MessageSid': 'SMtest123456789',
        'From': '+1234567890',
        'To': '+18382594031',
        'Body': 'Test SMS message',
        'NumMedia': '0'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'TwilioProxy/1.1'
    }
    
    print(f"🧪 Testing SMS webhook: {url}")
    result = test_webhook_endpoint(url, 'POST', test_data, headers)
    
    print(f"   Status: {result['status_code']}")
    print(f"   Response: {result['response_text']}")
    print(f"   Success: {'✅' if result['success'] else '❌'}")
    
    return result

def test_aws_transcribe_endpoint(base_url):
    """Test AWS Transcribe check endpoint."""
    url = urljoin(base_url, '/api/aws/transcribe/check')
    
    print(f"🧪 Testing AWS Transcribe endpoint: {url}")
    result = test_webhook_endpoint(url, 'GET')
    
    print(f"   Status: {result['status_code']}")
    print(f"   Response: {result['response_text']}")
    print(f"   Success: {'✅' if result['success'] else '❌'}")
    
    return result

def monitor_ngrok_requests():
    """Monitor ngrok requests for webhook testing."""
    try:
        response = requests.get("http://localhost:4040/api/requests/http")
        response.raise_for_status()
        
        requests_data = response.json()
        
        print(f"\n📊 Recent ngrok requests:")
        for req in requests_data.get('requests', [])[-5:]:  # Last 5 requests
            uri = req.get('uri', '')
            method = req.get('method', '')
            status = req.get('response', {}).get('status_code', 'N/A')
            timestamp = req.get('start_time', '')
            
            print(f"   {method} {uri} - {status} ({timestamp})")
        
        return True
    except Exception as e:
        print(f"❌ Error monitoring ngrok requests: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Testing Webhook Delivery and Processing...")
    print("=" * 60)
    
    # Get ngrok URL
    ngrok_url = get_ngrok_url()
    if not ngrok_url:
        print("❌ Could not get ngrok URL. Make sure ngrok is running.")
        return
    
    print(f"📡 Using ngrok URL: {ngrok_url}")
    print()
    
    # Test all webhook endpoints
    results = []
    
    # Test call webhook
    results.append(test_call_webhook(ngrok_url))
    print()
    
    # Test status webhook
    results.append(test_status_webhook(ngrok_url))
    print()
    
    # Test SMS webhook
    results.append(test_sms_webhook(ngrok_url))
    print()
    
    # Test AWS Transcribe endpoint
    results.append(test_aws_transcribe_endpoint(ngrok_url))
    print()
    
    # Monitor ngrok requests
    monitor_ngrok_requests()
    
    # Summary
    successful_tests = sum(1 for result in results if result['success'])
    total_tests = len(results)
    
    print(f"\n📋 Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Failed: {total_tests - successful_tests}")
    
    if successful_tests == total_tests:
        print("✅ All webhook tests passed!")
    else:
        print("❌ Some webhook tests failed. Check Flask app logs.")
    
    print(f"\n🔍 Monitor live requests at: http://localhost:4040")

if __name__ == "__main__":
    main() 