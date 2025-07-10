import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Optional, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from amazon_transcribe.client import TranscribeStreamingClient  # ✅ Added correct import
import websockets
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AWSTranscribeStreamingService:
    """
    AWS Transcribe Streaming service for real-time speech-to-text
    with speaker diarization and live analysis integration
    """
    
    def __init__(self, socketio=None):
        """Initialize AWS Transcribe Streaming service"""
        self.socketio = socketio
        self.logger = logging.getLogger(__name__)
        
        # AWS Configuration
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Transcribe client
        self.transcribe_client = None
        self.streaming_client = None
        
        # Active sessions
        self.active_sessions: Dict[str, Dict] = {}
        
        # WebSocket server
        self.websocket_server = None
        self.server_port = 8766
        
        # Thread pool for handling multiple sessions
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize AWS clients
        self._initialize_aws_clients()
        
        # WebSocket server will be started manually after Flask app initialization
        self.websocket_server_started = False
    
    def _initialize_aws_clients(self):
        """Initialize AWS Transcribe clients"""
        try:
            if not all([self.aws_access_key, self.aws_secret_key]):
                self.logger.warning("AWS credentials not found. Streaming will use simulation mode.")
                return
            
            # Create boto3 session for regular transcribe client
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            
            # Initialize clients
            self.transcribe_client = session.client('transcribe')
            self.streaming_client = TranscribeStreamingClient(region=self.aws_region)  # ✅ Using correct client
            
            self.logger.info(f"✅ AWS Transcribe Streaming initialized (Region: {self.aws_region})")
            
        except (NoCredentialsError, ClientError) as e:
            self.logger.error(f"AWS initialization failed: {str(e)}")
            self.transcribe_client = None
            self.streaming_client = None
    
    def _start_websocket_server(self):
        """Start WebSocket server for real-time transcription"""
        # Prevent multiple server instances
        if hasattr(self, '_server_running') and self._server_running:
            self.logger.info("WebSocket server already running, skipping startup")
            return
            
        def run_server():
            try:
                self._server_running = True
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def handler(websocket, path=None):
                    await self._handle_websocket_connection(websocket, path)
                
                async def start_server():
                    try:
                        server = await websockets.serve(
                            handler, 
                            "localhost", 
                            self.server_port,
                            ping_interval=20,
                            ping_timeout=10
                        )
                        self.logger.info(f"🔗 WebSocket server started on ws://localhost:{self.server_port}")
                        await server.wait_closed()
                    except OSError as e:
                        if "address already in use" in str(e).lower() or "10048" in str(e):
                            self.logger.warning(f"Port {self.server_port} already in use, WebSocket server not started")
                        else:
                            raise
                
                loop.run_until_complete(start_server())
                
            except Exception as e:
                self.logger.error(f"WebSocket server error: {str(e)}")
                # Don't restart automatically to prevent spam
                self._server_running = False
        
        # Start server in separate thread only if not already running
        if not hasattr(self, '_server_running') or not self._server_running:
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connection for streaming transcription"""
        client_id = str(uuid.uuid4())
        session_id = None
        
        try:
            self.logger.info(f"📱 New WebSocket connection: {client_id}")
            
            async for message in websocket:
                try:
                    if isinstance(message, str):
                        # JSON control message
                        data = json.loads(message)
                        result = await self._handle_control_message(
                            websocket, data, client_id
                        )
                        
                        if result and 'session_id' in result:
                            session_id = result['session_id']
                    
                    elif isinstance(message, bytes):
                        # Binary audio data
                        if session_id and session_id in self.active_sessions:
                            await self._handle_audio_data(
                                websocket, message, session_id
                            )
                
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON message'
                    }))
                except Exception as e:
                    self.logger.error(f"Message handling error: {str(e)}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"🔌 WebSocket connection closed: {client_id}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
        finally:
            # Cleanup session
            if session_id and session_id in self.active_sessions:
                await self._cleanup_session(session_id)
    
    async def _handle_control_message(self, websocket, data, client_id):
        """Handle control messages from client"""
        action = data.get('action')
        
        if action == 'start':
            return await self._start_transcription_session(websocket, data, client_id)
        elif action == 'stop':
            session_id = data.get('session_id')
            if session_id:
                await self._stop_transcription_session(session_id)
        elif action == 'configure':
            session_id = data.get('session_id')
            if session_id and session_id in self.active_sessions:
                await self._configure_session(session_id, data)
        
        return None
    
    async def _start_transcription_session(self, websocket, config, client_id):
        """Start new transcription session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Session configuration
            session_config = {
                'session_id': session_id,
                'client_id': client_id,
                'websocket': websocket,
                'language': config.get('language', 'en-US'),
                'sample_rate': config.get('sampleRate', 16000),
                'enable_diarization': config.get('enableSpeakerDiarization', True),
                'max_speakers': config.get('maxSpeakers', 4),
                'call_uuid': config.get('call_uuid'),
                'start_time': datetime.now(),
                'status': 'active',
                'transcription_buffer': [],
                'speaker_map': {},
                'partial_results': config.get('enablePartialResults', True)
            }
            
            # Store session
            self.active_sessions[session_id] = session_config
            
            # Start AWS Transcribe streaming if available
            if self.streaming_client:
                await self._start_aws_transcribe_stream(session_id)
            else:
                # Use simulation mode
                await self._start_simulation_mode(session_id)
            
            # Send confirmation
            await websocket.send(json.dumps({
                'type': 'session_started',
                'session_id': session_id,
                'config': {
                    'language': session_config['language'],
                    'sample_rate': session_config['sample_rate'],
                    'diarization_enabled': session_config['enable_diarization'],
                    'aws_enabled': self.streaming_client is not None
                }
            }))
            
            # Emit to main app via SocketIO
            if self.socketio and session_config['call_uuid']:
                self.socketio.emit('transcription_session_started', {
                    'call_uuid': session_config['call_uuid'],
                    'session_id': session_id,
                    'config': session_config
                })
            
            self.logger.info(f"🎙️ Started transcription session: {session_id}")
            
            return {'session_id': session_id}
            
        except Exception as e:
            self.logger.error(f"Failed to start transcription session: {str(e)}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Failed to start session: {str(e)}'
            }))
            return None
    
    async def _start_aws_transcribe_stream(self, session_id):
        """Start AWS Transcribe streaming for session"""
        try:
            session = self.active_sessions[session_id]
            
            # Create transcribe streaming request
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._create_transcribe_stream,
                session
            )
            
            # Store stream in session
            session['aws_stream'] = response
            session['aws_mode'] = True
            
            # Start processing stream responses
            asyncio.create_task(self._process_aws_stream_responses(session_id))
            
        except Exception as e:
            self.logger.error(f"AWS Transcribe stream failed: {str(e)}")
            # Fallback to simulation
            await self._start_simulation_mode(session_id)
    
    def _create_transcribe_stream(self, session):
        """Create AWS Transcribe streaming request (runs in executor)"""
        try:
            # Configure stream parameters
            stream_params = {
                'LanguageCode': session['language'],
                'MediaSampleRateHertz': session['sample_rate'],
                'MediaEncoding': 'pcm',
                'AudioStream': self._audio_stream_generator(session['session_id'])
            }
            
            # Add speaker diarization if enabled
            if session['enable_diarization']:
                stream_params['Settings'] = {
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': session['max_speakers']
                }
            
            # Start streaming
            response = self.streaming_client.start_stream_transcription(**stream_params)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to create transcribe stream: {str(e)}")
            raise
    
    async def _audio_stream_generator(self, session_id):
        """Generator for audio stream data"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        while session.get('status') == 'active':
            # Wait for audio data
            if 'audio_queue' in session and not session['audio_queue'].empty():
                audio_data = await session['audio_queue'].get()
                yield {'AudioEvent': {'AudioChunk': audio_data}}
            else:
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
    
    async def _process_aws_stream_responses(self, session_id):
        """Process responses from AWS Transcribe stream"""
        try:
            session = self.active_sessions.get(session_id)
            if not session or 'aws_stream' not in session:
                return
            
            async for event in session['aws_stream']['TranscriptResultStream']:
                if 'TranscriptEvent' in event:
                    transcript_event = event['TranscriptEvent']
                    
                    # Process transcript results
                    for result in transcript_event.get('Transcript', {}).get('Results', []):
                        await self._process_transcription_result(session_id, result)
                
                elif 'BadRequestException' in event:
                    error = event['BadRequestException']
                    self.logger.error(f"AWS Transcribe error: {error}")
                    await self._handle_transcription_error(session_id, error)
        
        except Exception as e:
            self.logger.error(f"AWS stream processing error: {str(e)}")
            await self._handle_transcription_error(session_id, str(e))
    
    async def _process_transcription_result(self, session_id, result):
        """Process individual transcription result"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            # Extract transcript data
            is_partial = not result.get('IsPartial', True)
            alternatives = result.get('Alternatives', [])
            
            if not alternatives:
                return
            
            alternative = alternatives[0]
            transcript = alternative.get('Transcript', '')
            confidence = alternative.get('Confidence', 0.0)
            
            # Extract speaker information
            speaker_label = None
            if 'Items' in alternative:
                for item in alternative['Items']:
                    if 'SpeakerLabel' in item:
                        speaker_label = item['SpeakerLabel']
                        break
            
            # Process speaker diarization
            speaker_info = self._process_speaker_diarization(
                session_id, speaker_label, transcript
            )
            
            # Create result object
            transcription_result = {
                'type': 'transcript',
                'session_id': session_id,
                'text': transcript,
                'confidence': confidence,
                'is_partial': is_partial,
                'speaker': speaker_info,
                'timestamp': datetime.now().isoformat(),
                'source': 'aws_transcribe'
            }
            
            # Send to client
            websocket = session.get('websocket')
            if websocket:
                await websocket.send(json.dumps(transcription_result))
            
            # Store in buffer
            if not is_partial:
                session['transcription_buffer'].append(transcription_result)
            
            # Emit to main app
            if self.socketio and session.get('call_uuid'):
                self.socketio.emit('live_transcription', {
                    'call_uuid': session['call_uuid'],
                    'transcription': transcript,
                    'confidence': confidence,
                    'speaker': speaker_info,
                    'is_partial': is_partial,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Trigger real-time analysis for complete sentences
            if not is_partial and len(transcript.strip()) > 10:
                await self._trigger_real_time_analysis(session_id, transcription_result)
            
        except Exception as e:
            self.logger.error(f"Error processing transcription result: {str(e)}")
    
    def _process_speaker_diarization(self, session_id, speaker_label, transcript):
        """Process speaker diarization information"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        if not speaker_label:
            return {'label': 'Unknown', 'name': 'Unknown Speaker'}
        
        # Map speaker labels to friendly names
        speaker_map = session.get('speaker_map', {})
        
        if speaker_label not in speaker_map:
            # Assign friendly name
            speaker_count = len(speaker_map)
            if speaker_count == 0:
                friendly_name = 'Customer'
            elif speaker_count == 1:
                friendly_name = 'Agent'
            else:
                friendly_name = f'Speaker {speaker_count + 1}'
            
            speaker_map[speaker_label] = {
                'name': friendly_name,
                'first_heard': datetime.now().isoformat(),
                'utterance_count': 0,
                'total_words': 0
            }
            
            session['speaker_map'] = speaker_map
        
        # Update speaker stats
        speaker_info = speaker_map[speaker_label]
        speaker_info['utterance_count'] += 1
        speaker_info['total_words'] += len(transcript.split())
        speaker_info['last_heard'] = datetime.now().isoformat()
        
        return {
            'label': speaker_label,
            'name': speaker_info['name'],
            'utterance_count': speaker_info['utterance_count'],
            'total_words': speaker_info['total_words']
        }
    
    async def _trigger_real_time_analysis(self, session_id, transcription_result):
        """Trigger real-time analysis for transcription"""
        try:
            session = self.active_sessions.get(session_id)
            if not session or not self.socketio:
                return
            
            # Get recent context (last 5 utterances)
            buffer = session.get('transcription_buffer', [])
            recent_context = buffer[-5:] if len(buffer) >= 5 else buffer
            
            # Combine recent text for analysis
            context_text = ' '.join([item['text'] for item in recent_context])
            
            # Emit analysis request
            self.socketio.emit('real_time_analysis_request', {
                'call_uuid': session.get('call_uuid'),
                'session_id': session_id,
                'text': context_text,
                'current_utterance': transcription_result,
                'speaker_info': transcription_result.get('speaker'),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Real-time analysis trigger error: {str(e)}")
    
    async def _start_simulation_mode(self, session_id):
        """Start simulation mode for testing without AWS"""
        try:
            session = self.active_sessions[session_id]
            session['aws_mode'] = False
            session['simulation_mode'] = True
            
            # Start simulation task
            asyncio.create_task(self._simulation_transcription(session_id))
            
            self.logger.info(f"🎭 Started simulation mode for session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Simulation mode error: {str(e)}")
    
    async def _simulation_transcription(self, session_id):
        """Simulate transcription for testing"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            # Simulate realistic conversation
            simulation_phrases = [
                ("Customer", "Hello, I'm calling about my recent order"),
                ("Agent", "Hi! I'd be happy to help you with your order. Can you provide your order number?"),
                ("Customer", "Yes, it's order number 12345. I haven't received it yet"),
                ("Agent", "Let me check on that for you. I see your order was shipped yesterday"),
                ("Customer", "Oh great! Do you have a tracking number?"),
                ("Agent", "Yes, your tracking number is 1Z999AA1234567890"),
                ("Customer", "Perfect, thank you so much for your help!"),
                ("Agent", "You're welcome! Is there anything else I can help you with today?"),
                ("Customer", "No, that's all. Have a great day!"),
                ("Agent", "You too! Thanks for calling.")
            ]
            
            for i, (speaker, text) in enumerate(simulation_phrases):
                if session.get('status') != 'active':
                    break
                
                # Simulate typing delay
                await asyncio.sleep(3 + (len(text) * 0.05))
                
                # Create transcription result
                result = {
                    'type': 'transcript',
                    'session_id': session_id,
                    'text': text,
                    'confidence': 0.95,
                    'is_partial': False,
                    'speaker': {'label': f'spk_{i % 2}', 'name': speaker},
                    'timestamp': datetime.now().isoformat(),
                    'source': 'simulation'
                }
                
                # Send to client
                websocket = session.get('websocket')
                if websocket:
                    await websocket.send(json.dumps(result))
                
                # Store in buffer
                session['transcription_buffer'].append(result)
                
                # Emit to main app
                if self.socketio and session.get('call_uuid'):
                    self.socketio.emit('live_transcription', {
                        'call_uuid': session['call_uuid'],
                        'transcription': text,
                        'confidence': 0.95,
                        'speaker': {'name': speaker},
                        'is_partial': False,
                        'timestamp': datetime.now().isoformat()
                    })
        
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
    
    async def _handle_audio_data(self, websocket, audio_data, session_id):
        """Handle incoming audio data"""
        try:
            session = self.active_sessions.get(session_id)
            if not session or session.get('status') != 'active':
                return
            
            # Initialize audio queue if needed
            if 'audio_queue' not in session:
                session['audio_queue'] = asyncio.Queue()
            
            # Add audio data to queue
            await session['audio_queue'].put(audio_data)
            
        except Exception as e:
            self.logger.error(f"Audio data handling error: {str(e)}")
    
    async def _stop_transcription_session(self, session_id):
        """Stop transcription session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            # Update status
            session['status'] = 'stopped'
            session['end_time'] = datetime.now()
            
            # Stop AWS stream if active
            if session.get('aws_stream'):
                # Close AWS stream
                session['aws_stream'].close()
            
            # Send final results
            websocket = session.get('websocket')
            if websocket:
                await websocket.send(json.dumps({
                    'type': 'session_ended',
                    'session_id': session_id,
                    'final_transcript': self._get_final_transcript(session_id),
                    'session_stats': self._get_session_stats(session_id)
                }))
            
            # Emit to main app
            if self.socketio and session.get('call_uuid'):
                self.socketio.emit('transcription_session_ended', {
                    'call_uuid': session['call_uuid'],
                    'session_id': session_id,
                    'final_transcript': self._get_final_transcript(session_id),
                    'duration': (session['end_time'] - session['start_time']).total_seconds()
                })
            
            self.logger.info(f"🛑 Stopped transcription session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error stopping session: {str(e)}")
    
    def _get_final_transcript(self, session_id):
        """Get final transcript for session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return ""
        
        buffer = session.get('transcription_buffer', [])
        return ' '.join([item['text'] for item in buffer if not item.get('is_partial')])
    
    def _get_session_stats(self, session_id):
        """Get session statistics"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {}
        
        buffer = session.get('transcription_buffer', [])
        speaker_map = session.get('speaker_map', {})
        
        return {
            'total_utterances': len(buffer),
            'total_words': sum(len(item['text'].split()) for item in buffer),
            'speakers': len(speaker_map),
            'speaker_breakdown': speaker_map,
            'duration_seconds': (
                session.get('end_time', datetime.now()) - session['start_time']
            ).total_seconds()
        }
    
    async def _cleanup_session(self, session_id):
        """Clean up session resources"""
        try:
            if session_id in self.active_sessions:
                await self._stop_transcription_session(session_id)
                del self.active_sessions[session_id]
                self.logger.info(f"🧹 Cleaned up session: {session_id}")
        except Exception as e:
            self.logger.error(f"Session cleanup error: {str(e)}")
    
    async def _handle_transcription_error(self, session_id, error):
        """Handle transcription errors"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            # Send error to client
            websocket = session.get('websocket')
            if websocket:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'session_id': session_id,
                    'message': str(error),
                    'timestamp': datetime.now().isoformat()
                }))
            
            # Emit to main app
            if self.socketio and session.get('call_uuid'):
                self.socketio.emit('transcription_error', {
                    'call_uuid': session['call_uuid'],
                    'session_id': session_id,
                    'error': str(error),
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            self.logger.error(f"Error handling transcription error: {str(e)}")
    
    def is_aws_available(self):
        """Check if AWS Transcribe is available"""
        return self.streaming_client is not None
    
    def get_active_sessions(self):
        """Get list of active sessions"""
        return {
            session_id: {
                'call_uuid': session.get('call_uuid'),
                'language': session.get('language'),
                'start_time': session.get('start_time').isoformat() if session.get('start_time') else None,
                'status': session.get('status'),
                'aws_mode': session.get('aws_mode', False),
                'speaker_count': len(session.get('speaker_map', {}))
            }
            for session_id, session in self.active_sessions.items()
        }
    
    def get_session_transcript(self, session_id):
        """Get transcript for specific session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return {
            'session_id': session_id,
            'transcript': self._get_final_transcript(session_id),
            'buffer': session.get('transcription_buffer', []),
            'stats': self._get_session_stats(session_id)
        }
    
    def start_websocket_server(self):
        """Manually start WebSocket server (call after Flask app is ready)"""
        if not self.websocket_server_started:
            self._start_websocket_server()
            self.websocket_server_started = True
            self.logger.info("🚀 WebSocket server startup initiated")