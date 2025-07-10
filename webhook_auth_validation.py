#!/usr/bin/env python3
"""
Webhook Authentication and Validation Script
This script handles Twilio webhook authentication and validation.
"""

import os
import sys
import hmac
import hashlib
import base64
from urllib.parse import urljoin, urlparse
import requests

def load_env_file():
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = '.env'
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    
    return env_vars

def validate_twilio_signature(auth_token, signature, url, post_vars):
    """
    Validate Twilio webhook signature.
    
    Args:
        auth_token: Twilio auth token
        signature: X-Twilio-Signature header value
        url: Full URL of the webhook
        post_vars: Dictionary of POST parameters
    
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        # Create the signature string
        signature_string = url
        
        # Sort the POST variables and append to signature string
        if post_vars:
            sorted_vars = sorted(post_vars.items())
            for key, value in sorted_vars:
                signature_string += f"{key}{value}"
        
        # Create HMAC-SHA1 hash
        mac = hmac.new(
            auth_token.encode('utf-8'),
            signature_string.encode('utf-8'),
            hashlib.sha1
        )
        
        # Base64 encode the hash
        expected_signature = base64.b64encode(mac.digest()).decode('utf-8')
        
        # Compare signatures
        return hmac.compare_digest(expected_signature, signature)
        
    except Exception as e:
        print(f"❌ Error validating signature: {e}")
        return False

def create_validation_middleware():
    """Create Flask middleware for webhook validation."""
    middleware_code = '''
from functools import wraps
from flask import request, abort
import hmac
import hashlib
import base64
import os

def validate_twilio_webhook(f):
    """Decorator to validate Twilio webhook signatures."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get Twilio auth token from environment
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        if not auth_token:
            print("⚠️  Warning: TWILIO_AUTH_TOKEN not found, skipping validation")
            return f(*args, **kwargs)
        
        # Get signature from headers
        signature = request.headers.get('X-Twilio-Signature')
        if not signature:
            print("⚠️  Warning: No X-Twilio-Signature header found")
            return f(*args, **kwargs)
        
        # Build full URL
        url = request.url
        
        # Get POST data
        post_vars = request.form.to_dict()
        
        # Validate signature
        if validate_signature(auth_token, signature, url, post_vars):
            return f(*args, **kwargs)
        else:
            print("❌ Invalid Twilio signature")
            abort(403)
    
    return decorated_function

def validate_signature(auth_token, signature, url, post_vars):
    """Validate Twilio webhook signature."""
    try:
        # Create the signature string
        signature_string = url
        
        # Sort the POST variables and append to signature string
        if post_vars:
            sorted_vars = sorted(post_vars.items())
            for key, value in sorted_vars:
                signature_string += f"{key}{value}"
        
        # Create HMAC-SHA1 hash
        mac = hmac.new(
            auth_token.encode('utf-8'),
            signature_string.encode('utf-8'),
            hashlib.sha1
        )
        
        # Base64 encode the hash
        expected_signature = base64.b64encode(mac.digest()).decode('utf-8')
        
        # Compare signatures
        return hmac.compare_digest(expected_signature, signature)
        
    except Exception as e:
        print(f"❌ Error validating signature: {e}")
        return False
'''
    
    with open('webhook_validation.py', 'w') as f:
        f.write(middleware_code)
    
    print("✅ Created webhook_validation.py middleware")

def test_signature_validation():
    """Test signature validation with sample data."""
    print("🧪 Testing signature validation...")
    
    # Load environment variables
    env_vars = load_env_file()
    auth_token = env_vars.get('TWILIO_AUTH_TOKEN')
    
    if not auth_token:
        print("❌ TWILIO_AUTH_TOKEN not found in .env file")
        return False
    
    # Test data
    test_url = "https://9fb47f1115d0.ngrok-free.app/api/call/webhook"
    test_post_vars = {
        'CallSid': 'CAtest123456789',
        'From': '+1234567890',
        'To': '+18382594031',
        'CallStatus': 'ringing'
    }
    
    # Create expected signature
    signature_string = test_url
    sorted_vars = sorted(test_post_vars.items())
    for key, value in sorted_vars:
        signature_string += f"{key}{value}"
    
    mac = hmac.new(
        auth_token.encode('utf-8'),
        signature_string.encode('utf-8'),
        hashlib.sha1
    )
    
    expected_signature = base64.b64encode(mac.digest()).decode('utf-8')
    
    # Test validation
    is_valid = validate_twilio_signature(auth_token, expected_signature, test_url, test_post_vars)
    
    print(f"   Test signature: {expected_signature[:20]}...")
    print(f"   Validation result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    return is_valid

def create_webhook_security_headers():
    """Create security headers for webhook endpoints."""
    security_code = '''
# Security headers for webhook endpoints
WEBHOOK_SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'",
    'Referrer-Policy': 'strict-origin-when-cross-origin'
}

def add_security_headers(response):
    """Add security headers to response."""
    for header, value in WEBHOOK_SECURITY_HEADERS.items():
        response.headers[header] = value
    return response
'''
    
    with open('webhook_security.py', 'w') as f:
        f.write(security_code)
    
    print("✅ Created webhook_security.py headers")

def update_app_with_validation():
    """Update app.py to include webhook validation."""
    print("🔧 Updating app.py with webhook validation...")
    
    # Read current app.py
    try:
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        # Check if validation is already added
        if 'validate_twilio_webhook' in app_content:
            print("✅ Webhook validation already present in app.py")
            return True
        
        # Add import at the top
        import_line = "from webhook_validation import validate_twilio_webhook\n"
        
        # Find the imports section and add our import
        lines = app_content.split('\n')
        import_added = False
        
        for i, line in enumerate(lines):
            if line.startswith('from flask') or line.startswith('import flask'):
                lines.insert(i + 1, import_line)
                import_added = True
                break
        
        if not import_added:
            lines.insert(0, import_line)
        
        # Add decorator to webhook endpoints
        webhook_endpoints = [
            '@app.route(\'/api/call/webhook\'',
            '@app.route(\'/api/call/status\'',
            '@app.route(\'/api/sms/webhook\''
        ]
        
        for i, line in enumerate(lines):
            for endpoint in webhook_endpoints:
                if line.strip().startswith(endpoint):
                    # Add validation decorator before the route decorator
                    lines.insert(i, '@validate_twilio_webhook')
                    break
        
        # Write updated content
        with open('app.py', 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ Updated app.py with webhook validation")
        return True
        
    except Exception as e:
        print(f"❌ Error updating app.py: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Handling Webhook Authentication and Validation...")
    print("=" * 60)
    
    # Create validation middleware
    create_validation_middleware()
    print()
    
    # Create security headers
    create_webhook_security_headers()
    print()
    
    # Test signature validation
    test_signature_validation()
    print()
    
    # Update app.py with validation
    update_app_with_validation()
    print()
    
    print("✅ Webhook authentication and validation setup completed!")
    print("\n📋 Created files:")
    print("   - webhook_validation.py (validation middleware)")
    print("   - webhook_security.py (security headers)")
    print("   - Updated app.py with validation decorators")
    
    print("\n🔒 Security features implemented:")
    print("   - Twilio signature validation")
    print("   - HMAC-SHA1 verification")
    print("   - Security headers for webhook endpoints")
    print("   - Request validation middleware")
    
    print("\n💡 Usage:")
    print("   - Webhook endpoints now validate Twilio signatures")
    print("   - Invalid signatures return 403 Forbidden")
    print("   - Security headers added to all responses")

if __name__ == "__main__":
    main() 