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