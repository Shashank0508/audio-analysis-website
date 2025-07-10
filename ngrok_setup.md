# Ngrok Setup Guide for Call Insights Webhook Testing

## Overview
This guide helps you set up ngrok to expose your local Flask application to the internet, enabling Twilio webhooks to reach your development server.

## Prerequisites
- Flask application running on localhost
- Twilio account with phone numbers configured
- Internet connection for ngrok tunnel

## Step 1: Install ngrok (if not already installed)

### Option A: Download from ngrok.com
1. Go to https://ngrok.com/download
2. Download the appropriate version for Windows
3. Extract the executable to a folder in your PATH

### Option B: Install via Chocolatey (Windows)
```bash
choco install ngrok
```

### Option C: Install via Scoop (Windows)
```bash
scoop install ngrok
```

## Step 2: Create ngrok Account and Get Auth Token

1. Sign up at https://ngrok.com/signup
2. Go to https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your authentication token
4. Run the following command to configure ngrok:

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
```

## Step 3: Configure ngrok for Your Flask App

### Basic Configuration
Create a `ngrok.yml` configuration file in your project directory:

```yaml
version: "2"
authtoken: YOUR_AUTH_TOKEN_HERE
tunnels:
  flask-app:
    proto: http
    addr: 5000
    subdomain: call-insights-dev  # Optional: custom subdomain (requires paid plan)
    bind_tls: true
    inspect: true
  
  websocket:
    proto: http
    addr: 8766
    subdomain: call-insights-ws   # Optional: for WebSocket server
    bind_tls: true
    inspect: true
```

### Advanced Configuration with Multiple Services
```yaml
version: "2"
authtoken: YOUR_AUTH_TOKEN_HERE
tunnels:
  main-app:
    proto: http
    addr: 5000
    bind_tls: true
    inspect: true
    headers:
      Host: localhost:5000
  
  aws-transcribe:
    proto: http
    addr: 8766
    bind_tls: true
    inspect: true
    
  phone-service:
    proto: http
    addr: 5001
    bind_tls: true
    inspect: true
```

## Step 4: Start ngrok Tunnel

### For Single Service (Flask App)
```bash
ngrok http 5000
```

### For Multiple Services (using config file)
```bash
ngrok start --all --config ngrok.yml
```

### For Specific Tunnel
```bash
ngrok start flask-app --config ngrok.yml
```

## Step 5: Configure Twilio Webhooks

Once ngrok is running, you'll get URLs like:
- `https://abc123.ngrok.io` (for your Flask app)
- `https://def456.ngrok.io` (for WebSocket service)

### Update Twilio Phone Number Webhooks

1. Go to Twilio Console → Phone Numbers → Manage → Active Numbers
2. Click on your phone number
3. Update webhook URLs:

**Voice Configuration:**
- Webhook: `https://YOUR_NGROK_URL.ngrok.io/api/call/webhook`
- HTTP Method: POST

**Messaging Configuration:**
- Webhook: `https://YOUR_NGROK_URL.ngrok.io/api/sms/webhook`
- HTTP Method: POST

**Status Callback:**
- Webhook: `https://YOUR_NGROK_URL.ngrok.io/api/call/status`
- HTTP Method: POST

## Step 6: Environment Variables

Update your `.env` file with ngrok URLs:

```env
# Existing variables
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number

# ngrok URLs
NGROK_URL=https://abc123.ngrok.io
WEBHOOK_BASE_URL=https://abc123.ngrok.io
WEBSOCKET_URL=wss://def456.ngrok.io

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
```

## Step 7: Testing Commands

### Test Webhook Connectivity
```bash
# Test call webhook
curl -X POST https://YOUR_NGROK_URL.ngrok.io/api/call/webhook \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "CallSid=test123&From=+1234567890&To=+0987654321&CallStatus=ringing"

# Test status webhook
curl -X POST https://YOUR_NGROK_URL.ngrok.io/api/call/status \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "CallSid=test123&CallStatus=completed&CallDuration=120"
```

### Test Flask App Endpoints
```bash
# Test main app
curl https://YOUR_NGROK_URL.ngrok.io/

# Test call API
curl https://YOUR_NGROK_URL.ngrok.io/api/call/test

# Test AWS Transcribe
curl https://YOUR_NGROK_URL.ngrok.io/api/aws/transcribe/check
```

## Step 8: Monitoring and Debugging

### ngrok Web Interface
- Access: http://localhost:4040
- View all requests and responses
- Inspect webhook payloads
- Debug connection issues

### Useful ngrok Commands
```bash
# View tunnel status
ngrok status

# View configuration
ngrok config check

# View help
ngrok help

# Kill all tunnels
ngrok kill
```

## Step 9: Development Workflow

### Starting Development Environment
```bash
# Terminal 1: Start Flask app
python app.py

# Terminal 2: Start AWS Transcribe WebSocket
python aws_transcribe_streaming.py

# Terminal 3: Start ngrok tunnels
ngrok start --all --config ngrok.yml
```

### Updating Webhooks After Restart
Every time you restart ngrok, the URLs change (unless using paid custom domains).
You'll need to:

1. Copy new ngrok URLs
2. Update Twilio webhook configurations
3. Update environment variables
4. Restart your Flask application

## Step 10: Production Considerations

### Free vs Paid Plans
- **Free**: Random URLs, 8-hour sessions, basic features
- **Paid**: Custom domains, longer sessions, more tunnels

### Security Best Practices
- Use HTTPS only (bind_tls: true)
- Validate webhook signatures
- Implement rate limiting
- Use basic authentication if needed

### Performance Optimization
```yaml
# In ngrok.yml
tunnels:
  flask-app:
    proto: http
    addr: 5000
    bind_tls: true
    inspect: false  # Disable for better performance
    compression: true
    request_header:
      add:
        - "X-Forwarded-Proto: https"
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 5000
   netstat -ano | findstr :5000
   ```

2. **Webhook Not Receiving Data**
   - Check ngrok web interface (http://localhost:4040)
   - Verify Twilio webhook configuration
   - Check Flask app logs

3. **SSL/TLS Issues**
   - Ensure `bind_tls: true` in configuration
   - Use HTTPS URLs in Twilio webhooks

4. **Authentication Errors**
   - Verify auth token: `ngrok config check`
   - Re-add auth token if needed

### Debug Commands
```bash
# Check ngrok configuration
ngrok config check

# View detailed logs
ngrok http 5000 --log=stdout --log-level=debug

# Test connectivity
curl -I https://YOUR_NGROK_URL.ngrok.io
```

## Next Steps

After ngrok is configured:
1. Test call functionality end-to-end
2. Verify real-time transcription works
3. Test webhook delivery and processing
4. Monitor performance and logs
5. Set up error handling and alerts

## Quick Reference

### Essential Commands
```bash
# Start basic tunnel
ngrok http 5000

# Start with config file
ngrok start --all --config ngrok.yml

# View web interface
open http://localhost:4040

# Kill tunnels
ngrok kill
```

### Important URLs
- ngrok Dashboard: https://dashboard.ngrok.com
- Local Web Interface: http://localhost:4040
- Twilio Console: https://console.twilio.com
- AWS Console: https://console.aws.amazon.com 