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
            print("Warning: TWILIO_AUTH_TOKEN not found, skipping validation")
            return f(*args, **kwargs)
        
        # Get signature from headers
        signature = request.headers.get('X-Twilio-Signature')
        if not signature:
            print("Warning: No X-Twilio-Signature header found")
            return f(*args, **kwargs)
        
        # Build full URL
        url = request.url
        
        # Get POST data
        post_vars = request.form.to_dict()
        
        # Validate signature
        if validate_signature(auth_token, signature, url, post_vars):
            return f(*args, **kwargs)
        else:
            print("Invalid Twilio signature")
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
        print(f"Error validating signature: {e}")
        return False

