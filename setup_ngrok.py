#!/usr/bin/env python3
"""
Ngrok Setup and Webhook Configuration Script
This script helps automate the setup of ngrok tunnels and Twilio webhook configuration.
"""

import os
import sys
import json
import time
import requests
import subprocess
from typing import Dict, Optional
from urllib.parse import urljoin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NgrokSetup:
    def __init__(self):
        self.ngrok_api_url = "http://localhost:4040/api"
        self.tunnels = {}
        self.config_file = "ngrok.yml"
        
    def check_ngrok_installed(self) -> bool:
        """Check if ngrok is installed and accessible."""
        try:
            result = subprocess.run(['ngrok', 'version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"ngrok version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ngrok is not installed or not in PATH")
            return False
    
    def check_ngrok_auth(self) -> bool:
        """Check if ngrok is authenticated."""
        try:
            result = subprocess.run(['ngrok', 'config', 'check'], 
                                  capture_output=True, text=True, check=True)
            logger.info("ngrok authentication verified")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"ngrok authentication failed: {e.stderr}")
            return False
    
    def start_ngrok_tunnels(self) -> bool:
        """Start ngrok tunnels using the configuration file."""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_file):
                logger.error(f"Configuration file {self.config_file} not found")
                return False
            
            # Start tunnels
            cmd = ['ngrok', 'start', '--all', '--config', self.config_file]
            logger.info("Starting ngrok tunnels...")
            
            # Start ngrok in background
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for ngrok to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info("ngrok tunnels started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Failed to start ngrok: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting ngrok: {e}")
            return False
    
    def get_tunnel_urls(self) -> Dict[str, str]:
        """Get the current tunnel URLs from ngrok API."""
        try:
            response = requests.get(f"{self.ngrok_api_url}/tunnels")
            response.raise_for_status()
            
            tunnels_data = response.json()
            tunnel_urls = {}
            
            for tunnel in tunnels_data.get('tunnels', []):
                name = tunnel.get('name', 'unknown')
                public_url = tunnel.get('public_url', '')
                
                if public_url.startswith('https://'):
                    tunnel_urls[name] = public_url
                    logger.info(f"Tunnel '{name}': {public_url}")
            
            self.tunnels = tunnel_urls
            return tunnel_urls
            
        except requests.RequestException as e:
            logger.error(f"Failed to get tunnel URLs: {e}")
            return {}
    
    def update_env_file(self, tunnel_urls: Dict[str, str]) -> bool:
        """Update .env file with ngrok URLs."""
        try:
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
            
            # Update with ngrok URLs
            if 'flask-app' in tunnel_urls:
                env_vars['NGROK_URL'] = tunnel_urls['flask-app']
                env_vars['WEBHOOK_BASE_URL'] = tunnel_urls['flask-app']
            
            if 'aws-transcribe' in tunnel_urls:
                ws_url = tunnel_urls['aws-transcribe'].replace('https://', 'wss://')
                env_vars['WEBSOCKET_URL'] = ws_url
            
            # Write updated .env file
            with open(env_file, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            logger.info(f"Updated {env_file} with ngrok URLs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
            return False
    
    def configure_twilio_webhooks(self, base_url: str) -> bool:
        """Configure Twilio webhooks with ngrok URLs."""
        try:
            # Load Twilio credentials from environment
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            phone_number = os.getenv('TWILIO_PHONE_NUMBER')
            
            if not all([account_sid, auth_token, phone_number]):
                logger.error("Missing Twilio credentials in environment variables")
                return False
            
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
            
            # Update phone number configuration
            phone_numbers = client.incoming_phone_numbers.list()
            
            for number in phone_numbers:
                if number.phone_number == phone_number:
                    number.update(**webhook_urls)
                    logger.info(f"Updated webhooks for {phone_number}")
                    return True
            
            logger.error(f"Phone number {phone_number} not found in Twilio account")
            return False
            
        except Exception as e:
            logger.error(f"Failed to configure Twilio webhooks: {e}")
            return False
    
    def test_webhooks(self, base_url: str) -> bool:
        """Test webhook endpoints."""
        try:
            test_endpoints = [
                '/api/call/webhook',
                '/api/call/status',
                '/api/sms/webhook',
                '/api/aws/transcribe/check'
            ]
            
            for endpoint in test_endpoints:
                url = urljoin(base_url, endpoint)
                try:
                    response = requests.get(url, timeout=5)
                    logger.info(f"Endpoint {endpoint}: {response.status_code}")
                except requests.RequestException as e:
                    logger.warning(f"Endpoint {endpoint} not accessible: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to test webhooks: {e}")
            return False
    
    def display_setup_info(self, tunnel_urls: Dict[str, str]):
        """Display setup information and next steps."""
        print("\n" + "="*60)
        print("NGROK SETUP COMPLETE")
        print("="*60)
        
        print("\n📡 TUNNEL URLS:")
        for name, url in tunnel_urls.items():
            print(f"  {name}: {url}")
        
        print("\n🔗 WEBHOOK ENDPOINTS:")
        if 'flask-app' in tunnel_urls:
            base_url = tunnel_urls['flask-app']
            endpoints = [
                '/api/call/webhook',
                '/api/call/status',
                '/api/sms/webhook',
                '/api/aws/transcribe/check'
            ]
            for endpoint in endpoints:
                print(f"  {urljoin(base_url, endpoint)}")
        
        print("\n🌐 MONITORING:")
        print(f"  ngrok Web Interface: http://localhost:4040")
        print(f"  Twilio Console: https://console.twilio.com")
        
        print("\n⚡ NEXT STEPS:")
        print("  1. Verify webhook configuration in Twilio Console")
        print("  2. Test call functionality from your application")
        print("  3. Monitor requests in ngrok web interface")
        print("  4. Check Flask application logs for webhook processing")
        
        print("\n" + "="*60)

def main():
    """Main setup function."""
    setup = NgrokSetup()
    
    print("🚀 Starting ngrok setup for Call Insights application...")
    
    # Check prerequisites
    if not setup.check_ngrok_installed():
        print("❌ Please install ngrok first. See ngrok_setup.md for instructions.")
        sys.exit(1)
    
    if not setup.check_ngrok_auth():
        print("❌ Please authenticate ngrok with your auth token.")
        print("   Run: ngrok config add-authtoken YOUR_AUTH_TOKEN")
        sys.exit(1)
    
    # Start tunnels
    if not setup.start_ngrok_tunnels():
        print("❌ Failed to start ngrok tunnels. Check configuration.")
        sys.exit(1)
    
    # Wait for tunnels to be ready
    print("⏳ Waiting for tunnels to be ready...")
    time.sleep(5)
    
    # Get tunnel URLs
    tunnel_urls = setup.get_tunnel_urls()
    if not tunnel_urls:
        print("❌ No tunnel URLs found. Check ngrok status.")
        sys.exit(1)
    
    # Update environment file
    setup.update_env_file(tunnel_urls)
    
    # Configure Twilio webhooks (optional)
    if 'flask-app' in tunnel_urls:
        print("🔧 Configuring Twilio webhooks...")
        setup.configure_twilio_webhooks(tunnel_urls['flask-app'])
    
    # Test webhooks
    if 'flask-app' in tunnel_urls:
        print("🧪 Testing webhook endpoints...")
        setup.test_webhooks(tunnel_urls['flask-app'])
    
    # Display setup information
    setup.display_setup_info(tunnel_urls)
    
    print("\n✅ ngrok setup completed successfully!")
    print("💡 Keep this terminal open to maintain the tunnels.")

if __name__ == "__main__":
    main() 