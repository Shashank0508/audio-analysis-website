@echo off
echo =====================================================
echo  Call Insights Development Environment Startup
echo =====================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if ngrok is available
ngrok version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ngrok is not installed or not in PATH
    echo Please install ngrok and try again
    echo See ngrok_setup.md for instructions
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📚 Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found
    echo Please create .env file with your credentials
    echo See .env.example for reference
    pause
    exit /b 1
)

REM Start services in separate windows
echo 🚀 Starting development services...

REM Start Flask app
echo 🌐 Starting Flask application...
start "Flask App" cmd /k "venv\Scripts\activate.bat && python app.py"

REM Wait a moment for Flask to start
timeout /t 3 /nobreak >nul

REM Start AWS Transcribe WebSocket service
echo 🎤 Starting AWS Transcribe WebSocket service...
start "AWS Transcribe" cmd /k "venv\Scripts\activate.bat && python aws_transcribe_streaming.py"

REM Wait a moment for WebSocket service to start
timeout /t 3 /nobreak >nul

REM Start ngrok tunnels
echo 🌍 Starting ngrok tunnels...
start "ngrok Tunnels" cmd /k "ngrok start --all --config ngrok.yml"

REM Wait for ngrok to start
timeout /t 5 /nobreak >nul

REM Run setup script
echo 🔧 Configuring webhooks...
python setup_ngrok.py

echo.
echo ✅ Development environment started successfully!
echo.
echo 📊 Access Points:
echo   - Flask App: http://localhost:5000
echo   - ngrok Web Interface: http://localhost:4040
echo   - AWS Transcribe WebSocket: ws://localhost:8766
echo.
echo 💡 Next Steps:
echo   1. Check ngrok web interface for tunnel URLs
echo   2. Update Twilio webhook configuration if needed
echo   3. Test call functionality from the web interface
echo.
echo Press any key to exit...
pause >nul 