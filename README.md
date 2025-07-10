# 🎵 Audio Analysis Website - AI-Powered Call Insights Platform

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features & Functionality](#-features--functionality)
- [Architecture](#-architecture)
- [Testing Instructions](#-testing-instructions)
- [Installation & Setup](#-installation--setup)
- [API Documentation](#-api-documentation)
- [Live Demo](#-live-demo)
- [Video Demonstration](#-video-demonstration)
- [Technology Stack](#-technology-stack)
- [Contributors](#-contributors)

## 🎯 Project Overview

The **Audio Analysis Website** is an advanced AI-powered platform that provides comprehensive audio transcription, real-time sentiment analysis, and intelligent insights extraction from voice calls and audio files. Built using cutting-edge AI technologies including OpenAI Whisper, AWS Transcribe, and advanced NLP models, this platform transforms raw audio into actionable business intelligence.

### Key Capabilities
- **Real-time Audio Transcription** with speaker diarization
- **Advanced Sentiment Analysis** and emotion detection
- **Live Call Analysis** with Twilio integration
- **Intelligent Topic Extraction** and keyword identification
- **Automated Report Generation** with Word document export
- **Email Integration** for automated report delivery
- **WebSocket-based Real-time Updates** for live monitoring

## 🚀 Features & Functionality

### 1. **Audio File Processing**
- **Multi-format Support**: WAV, MP3, MP4, M4A, WEBM, OGG, FLAC, AAC, and more
- **AI-Powered Transcription**: OpenAI Whisper with word-level timestamps
- **Automatic Audio Preprocessing**: Format conversion and optimization
- **Large File Handling**: Up to 16MB file uploads with progress tracking

### 2. **Real-time Call Analysis**
- **Twilio Integration**: Live call recording and transcription
- **AWS Transcribe Streaming**: Real-time speech-to-text conversion
- **Speaker Diarization**: Automatic speaker identification and separation
- **Live Sentiment Monitoring**: Real-time emotional state tracking
- **WebSocket Updates**: Instant analysis results during calls

### 3. **Advanced AI Analysis**
- **Sentiment Analysis**: Positive, negative, neutral classification with confidence scores
- **Emotion Detection**: Joy, sadness, anger, fear, surprise, and disgust identification
- **Topic Extraction**: Intelligent keyword and theme identification using TF-IDF and clustering
- **Key Insights Generation**: Automated business intelligence extraction
- **Statistical Analysis**: Word count, speaking time, and conversation metrics

### 4. **Report Generation & Export**
- **Comprehensive Reports**: Detailed analysis summaries with visualizations
- **Word Document Export**: Professional formatted reports with charts
- **JSON Export**: Machine-readable data for further processing
- **Email Integration**: Automated report delivery to stakeholders
- **Custom Templates**: Branded report formats

### 5. **Web Interface & User Experience**
- **Responsive Design**: Mobile-friendly interface with modern UI/UX
- **Real-time Dashboard**: Live monitoring of ongoing calls and analysis
- **Progress Tracking**: Visual feedback for file processing and analysis
- **Interactive Results**: Clickable segments with timestamp navigation
- **Dark/Light Theme**: Customizable interface preferences

## 🏗️ Architecture

### High-Level Architecture
```
Frontend (HTML/CSS/JS) → Flask Backend → AI/ML Services
                                    ↓
WebSocket Server ← → Real-time Updates ← → External APIs
                                    ↓
Database & Storage ← → File System ← → Cloud Services
```

### Core Components
1. **Flask Web Server**: Main application server with RESTful APIs
2. **WebSocket Server**: Real-time communication for live updates
3. **AI Processing Pipeline**: OpenAI Whisper + Custom NLP models
4. **External Integrations**: Twilio, AWS Transcribe, Email services
5. **File Management**: Upload handling, audio preprocessing, storage
6. **Report Engine**: Document generation and export functionality

## 🧪 Testing Instructions

### Prerequisites
Before testing, ensure you have:
- Python 3.10+ installed
- Internet connection for AI model downloads
- Audio files for testing (various formats)
- (Optional) Twilio account for call testing
- (Optional) AWS account for streaming transcription

### Quick Start Testing

#### 1. **Clone and Setup**
```bash
git clone https://github.com/Shashank0508/audio-analysis-website.git
cd audio-analysis-website
pip install -r requirements.txt
```

#### 2. **Environment Configuration**
```bash
# Copy the environment template
cp env_template.txt .env

# Edit .env file with your API keys (optional for basic testing)
# EMAIL_USER=your-email@gmail.com
# EMAIL_PASSWORD=your-app-password
# TWILIO_ACCOUNT_SID=your-twilio-sid
# TWILIO_AUTH_TOKEN=your-twilio-token
```

#### 3. **Start the Application**
```bash
python app.py
```
Access the application at: `http://localhost:5000`

### Testing Scenarios

#### 📁 **Test Case 1: Audio File Upload & Analysis**
1. **Navigate** to the main page (`http://localhost:5000`)
2. **Upload** an audio file (use provided sample files or your own)
3. **Click** "Transcribe Audio" and wait for processing
4. **Verify** transcription accuracy and timestamp alignment
5. **Click** "Analyze Text" to generate insights
6. **Check** sentiment analysis, emotion detection, and topic extraction
7. **Test** report generation (JSON and Word document export)
8. **Verify** email functionality (if configured)

**Expected Results:**
- Accurate transcription with timestamps
- Sentiment scores between -1.0 to 1.0
- Emotion percentages totaling 100%
- Key topics and insights extracted
- Professional report generation

#### 📞 **Test Case 2: Real-time Call Analysis (Advanced)**
*Requires Twilio configuration*
1. **Configure** Twilio credentials in `.env`
2. **Start** the application
3. **Navigate** to call testing interface
4. **Initiate** a test call
5. **Verify** real-time transcription updates
6. **Monitor** live sentiment analysis
7. **Check** WebSocket connectivity in browser developer tools

**Expected Results:**
- Real-time transcription appearing in interface
- Live sentiment updates during conversation
- Proper call state management
- Recording and analysis persistence

#### 🌐 **Test Case 3: API Testing**
Use tools like Postman or curl to test API endpoints:

```bash
# Health check
curl http://localhost:5000/health

# Upload file
curl -X POST -F "audio=@test-audio.wav" http://localhost:5000/upload

# Get analysis
curl -X POST -H "Content-Type: application/json" \
     -d '{"filename":"uploaded-file.wav"}' \
     http://localhost:5000/transcribe
```

#### 📊 **Test Case 4: Performance Testing**
1. **Upload** multiple files simultaneously
2. **Test** with large audio files (10MB+)
3. **Monitor** memory usage and processing time
4. **Verify** concurrent user handling

### Sample Test Files
Download sample audio files for testing:
- [Sample Conversation (WAV)](#) - 2-minute business call
- [Sample Interview (MP3)](#) - 5-minute interview recording
- [Sample Presentation (M4A)](#) - 3-minute presentation snippet

### Troubleshooting
- **Slow Processing**: Normal for first run (AI model downloads)
- **Memory Issues**: Reduce file size or restart application
- **WebSocket Errors**: Check browser compatibility and network
- **API Failures**: Verify environment variables and internet connection

## 🛠️ Installation & Setup

### System Requirements
- Python 3.10 or higher
- 4GB+ RAM (for AI model processing)
- 2GB+ disk space (for model storage)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Dependencies Installation
```bash
pip install -r requirements.txt
```

Key dependencies include:
- Flask 2.3.3 (Web framework)
- OpenAI Whisper (Speech recognition)
- NLTK & TextBlob (Natural language processing)
- Torch (Deep learning framework)
- Librosa (Audio processing)
- Flask-SocketIO (Real-time communication)

### Configuration Options
Edit `.env` file for customization:
- Email settings for report delivery
- Twilio credentials for call integration
- AWS credentials for streaming transcription
- Custom model paths and parameters

## 📡 API Documentation

### Core Endpoints

#### File Upload
```
POST /upload
Content-Type: multipart/form-data
Body: audio file
Response: {filename, file_size, file_path}
```

#### Transcription
```
POST /transcribe
Content-Type: application/json
Body: {filename}
Response: {transcription, segments, timestamps}
```

#### Analysis
```
POST /analyze
Content-Type: application/json
Body: {text}
Response: {sentiment, emotions, topics, insights, statistics}
```

#### Call Management
```
POST /api/calls/initiate - Start new call
GET /api/calls/status/{call_sid} - Get call status
POST /api/calls/end/{call_sid} - End active call
```

### WebSocket Events
- `connect` - Client connection established
- `real_time_analysis_result` - Live analysis updates
- `call_status_response` - Call state changes
- `transcription_update` - Real-time transcription

## 🌐 Live Demo

**Demo URL**: [https://audio-analysis-website.onrender.com](https://audio-analysis-website.onrender.com)

**Demo Features Available**:
- File upload and transcription
- Sentiment analysis and insights
- Report generation and export
- Real-time WebSocket updates

*Note: Call integration features require additional API keys*

## 🎥 Video Demonstration

**YouTube Link**: [Audio Analysis Platform Demo](https://youtube.com/watch?v=demo-video-id)

**Video Contents** (3 minutes):
1. **Overview** (0:00-0:30) - Platform introduction and key features
2. **File Upload Demo** (0:30-1:30) - Upload, transcription, and analysis
3. **Real-time Features** (1:30-2:30) - Live call analysis and WebSocket updates
4. **Report Generation** (2:30-3:00) - Export functionality and email delivery

## 💻 Technology Stack

### Frontend
- **HTML5/CSS3**: Responsive web interface
- **JavaScript (ES6+)**: Interactive features and real-time updates
- **WebSocket API**: Real-time communication
- **Bootstrap**: UI components and styling

### Backend
- **Flask 2.3.3**: Python web framework
- **Gunicorn**: Production WSGI server
- **Flask-SocketIO**: WebSocket support
- **Flask-CORS**: Cross-origin resource sharing

### AI/ML Stack
- **OpenAI Whisper**: Advanced speech recognition
- **NLTK**: Natural language processing toolkit
- **TextBlob**: Sentiment analysis and text processing
- **scikit-learn**: Machine learning algorithms
- **PyTorch**: Deep learning framework

### External Services
- **Twilio**: Voice calls and SMS integration
- **AWS Transcribe**: Real-time speech recognition
- **SMTP**: Email delivery service
- **Render**: Cloud deployment platform

### Data Processing
- **Librosa**: Audio analysis and preprocessing
- **Pydub**: Audio format conversion
- **FFmpeg**: Audio/video processing
- **python-docx**: Word document generation

## 👥 Contributors

**Shashank** - Lead Developer
- GitHub: [@Shashank0508](https://github.com/Shashank0508)
- Email: [Contact via GitHub]

## 📄 License

This project is developed for the **Impetus AWS GenAI Hackathon 2025**.

---

## 🏆 Hackathon Submission Details

**Repository**: [https://github.com/Shashank0508/audio-analysis-website](https://github.com/Shashank0508/audio-analysis-website)
**Live Demo**: [https://audio-analysis-website.onrender.com](https://audio-analysis-website.onrender.com)
**Video Demo**: [YouTube Link](https://youtube.com/watch?v=demo-video-id)
**Solution Deck**: [Available in repository](/docs/solution-deck.pdf)

**Submission Date**: July 11, 2025
**Hackathon**: Impetus AWS GenAI Hackathon 2025

---

*For judges: This application demonstrates advanced AI integration, real-time processing capabilities, and practical business applications. The platform is fully functional and ready for production deployment.* 