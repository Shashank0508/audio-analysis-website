#!/usr/bin/env python3
"""
Streaming Transcription Pipeline
Real-time audio transcription processing with AWS Transcribe integration.
"""

import asyncio
import websockets
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
import queue
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamingTranscriptionPipeline:
    """Manages real-time audio transcription pipeline."""
    
    def __init__(self):
        self.active_streams = {}
        self.transcription_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.callbacks = {
            'on_transcript': [],
            'on_speaker_change': [],
            'on_analysis_complete': []
        }
        
    def start_pipeline(self):
        """Start the transcription pipeline."""
        if self.running:
            logger.warning("Pipeline already running")
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_transcription_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("✅ Streaming transcription pipeline started")
    
    def stop_pipeline(self):
        """Stop the transcription pipeline."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("⏹️ Streaming transcription pipeline stopped")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for transcription events."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"✅ Registered callback for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def start_stream(self, session_id: str, stream_config: Dict) -> Dict:
        """Start a new transcription stream."""
        try:
            stream_data = {
                'session_id': session_id,
                'start_time': datetime.now().isoformat(),
                'config': stream_config,
                'status': 'active',
                'transcript_buffer': [],
                'current_speaker': None,
                'speaker_segments': [],
                'audio_buffer': [],
                'stats': {
                    'total_segments': 0,
                    'total_words': 0,
                    'avg_confidence': 0.0,
                    'speaker_changes': 0
                }
            }
            
            self.active_streams[session_id] = stream_data
            
            # Start AWS Transcribe stream if configured
            if stream_config.get('use_aws_transcribe', True):
                self._start_aws_transcribe_stream(session_id, stream_config)
            
            logger.info(f"✅ Started transcription stream: {session_id}")
            return {'success': True, 'session_id': session_id}
            
        except Exception as e:
            logger.error(f"❌ Error starting stream: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_audio_chunk(self, session_id: str, audio_data: bytes, metadata: Dict = None):
        """Process incoming audio chunk."""
        try:
            if session_id not in self.active_streams:
                logger.warning(f"Stream {session_id} not found")
                return
            
            stream = self.active_streams[session_id]
            
            # Add to audio buffer
            stream['audio_buffer'].append({
                'data': audio_data,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })
            
            # Process with AWS Transcribe (simulation)
            self._process_audio_with_transcribe(session_id, audio_data, metadata)
            
        except Exception as e:
            logger.error(f"❌ Error processing audio chunk: {e}")
    
    def _start_aws_transcribe_stream(self, session_id: str, config: Dict):
        """Start AWS Transcribe streaming session."""
        try:
            # This would integrate with actual AWS Transcribe streaming
            # For now, we'll simulate the connection
            
            transcribe_config = {
                'language_code': config.get('language', 'en-US'),
                'sample_rate': config.get('sample_rate', 16000),
                'enable_speaker_diarization': config.get('speaker_diarization', True),
                'max_speakers': config.get('max_speakers', 4),
                'enable_partial_results': True
            }
            
            logger.info(f"🎤 AWS Transcribe stream configured for {session_id}")
            logger.info(f"   Language: {transcribe_config['language_code']}")
            logger.info(f"   Sample Rate: {transcribe_config['sample_rate']}")
            logger.info(f"   Speaker Diarization: {transcribe_config['enable_speaker_diarization']}")
            
            # Store transcribe config in stream
            self.active_streams[session_id]['transcribe_config'] = transcribe_config
            
        except Exception as e:
            logger.error(f"❌ Error starting AWS Transcribe: {e}")
    
    def _process_audio_with_transcribe(self, session_id: str, audio_data: bytes, metadata: Dict):
        """Process audio data with AWS Transcribe (simulated)."""
        try:
            # Simulate transcription processing
            # In real implementation, this would send audio to AWS Transcribe
            
            # Create simulated transcript result
            simulated_transcript = self._simulate_transcription(audio_data, metadata)
            
            # Add to transcription queue for processing
            self.transcription_queue.put({
                'session_id': session_id,
                'transcript': simulated_transcript,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error in transcribe processing: {e}")
    
    def _simulate_transcription(self, audio_data: bytes, metadata: Dict) -> Dict:
        """Simulate AWS Transcribe response for testing."""
        # This simulates what AWS Transcribe would return
        sample_transcripts = [
            "Hello, how can I help you today?",
            "I'm having trouble with my account.",
            "Let me look into that for you.",
            "Thank you for your patience.",
            "Is there anything else I can help with?"
        ]
        
        import random
        
        transcript_text = random.choice(sample_transcripts)
        confidence = random.uniform(0.85, 0.98)
        speaker_label = random.choice(['Customer', 'Agent'])
        
        return {
            'text': transcript_text,
            'confidence': confidence,
            'speaker_label': speaker_label,
            'start_time': time.time(),
            'end_time': time.time() + 2.0,
            'is_final': random.choice([True, False]),
            'word_details': [
                {
                    'word': word,
                    'confidence': random.uniform(0.8, 0.95),
                    'start_time': time.time() + i * 0.3,
                    'end_time': time.time() + (i + 1) * 0.3
                }
                for i, word in enumerate(transcript_text.split())
            ]
        }
    
    def _process_transcription_queue(self):
        """Process transcription results from queue."""
        while self.running:
            try:
                # Get transcript from queue (with timeout)
                transcript_data = self.transcription_queue.get(timeout=1)
                
                session_id = transcript_data['session_id']
                transcript = transcript_data['transcript']
                
                # Process transcript
                self._handle_transcript_result(session_id, transcript)
                
                # Mark task as done
                self.transcription_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error processing transcription queue: {e}")
    
    def _handle_transcript_result(self, session_id: str, transcript: Dict):
        """Handle transcription result."""
        try:
            if session_id not in self.active_streams:
                return
            
            stream = self.active_streams[session_id]
            
            # Update stream stats
            stream['stats']['total_segments'] += 1
            stream['stats']['total_words'] += len(transcript['text'].split())
            
            # Update average confidence
            current_avg = stream['stats']['avg_confidence']
            total_segments = stream['stats']['total_segments']
            new_confidence = transcript['confidence']
            
            stream['stats']['avg_confidence'] = (
                (current_avg * (total_segments - 1) + new_confidence) / total_segments
            )
            
            # Handle speaker changes
            current_speaker = stream['current_speaker']
            new_speaker = transcript['speaker_label']
            
            if current_speaker != new_speaker:
                stream['current_speaker'] = new_speaker
                stream['stats']['speaker_changes'] += 1
                
                # Trigger speaker change callbacks
                for callback in self.callbacks['on_speaker_change']:
                    try:
                        callback(session_id, current_speaker, new_speaker)
                    except Exception as e:
                        logger.error(f"Error in speaker change callback: {e}")
            
            # Add to transcript buffer
            transcript_entry = {
                'text': transcript['text'],
                'speaker': new_speaker,
                'confidence': transcript['confidence'],
                'timestamp': datetime.now().isoformat(),
                'is_final': transcript['is_final'],
                'start_time': transcript['start_time'],
                'end_time': transcript['end_time'],
                'word_details': transcript.get('word_details', [])
            }
            
            stream['transcript_buffer'].append(transcript_entry)
            
            # Keep only last 100 entries in buffer
            if len(stream['transcript_buffer']) > 100:
                stream['transcript_buffer'] = stream['transcript_buffer'][-100:]
            
            # Add to speaker segments
            if transcript['is_final']:
                stream['speaker_segments'].append({
                    'speaker': new_speaker,
                    'text': transcript['text'],
                    'start_time': transcript['start_time'],
                    'end_time': transcript['end_time'],
                    'confidence': transcript['confidence']
                })
            
            # Trigger transcript callbacks
            for callback in self.callbacks['on_transcript']:
                try:
                    callback(session_id, transcript_entry)
                except Exception as e:
                    logger.error(f"Error in transcript callback: {e}")
            
            # Send to real-time analysis if final transcript
            if transcript['is_final']:
                self._send_to_analysis(session_id, transcript_entry)
            
            logger.info(f"📝 Processed transcript for {session_id}: {transcript['text'][:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error handling transcript: {e}")
    
    def _send_to_analysis(self, session_id: str, transcript_entry: Dict):
        """Send transcript to real-time analysis."""
        try:
            # Prepare transcript data for analysis
            analysis_data = {
                'text': transcript_entry['text'],
                'speaker': transcript_entry['speaker'],
                'timestamp': transcript_entry['timestamp'],
                'confidence': transcript_entry['confidence'],
                'is_final': transcript_entry['is_final']
            }
            
            # Send to real-time analysis (import here to avoid circular import)
            try:
                from flask_realtime_integration import process_transcript_for_analysis
                process_transcript_for_analysis(session_id, analysis_data)
            except ImportError:
                logger.warning("Flask integration not available")
            
            # Trigger analysis callbacks
            for callback in self.callbacks['on_analysis_complete']:
                try:
                    callback(session_id, analysis_data)
                except Exception as e:
                    logger.error(f"Error in analysis callback: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error sending to analysis: {e}")
    
    def get_stream_status(self, session_id: str) -> Dict:
        """Get current stream status."""
        if session_id not in self.active_streams:
            return {'error': 'Stream not found'}
        
        stream = self.active_streams[session_id]
        
        return {
            'session_id': session_id,
            'status': stream['status'],
            'start_time': stream['start_time'],
            'current_speaker': stream['current_speaker'],
            'stats': stream['stats'],
            'recent_transcripts': stream['transcript_buffer'][-10:],  # Last 10
            'speaker_segments_count': len(stream['speaker_segments']),
            'buffer_size': len(stream['transcript_buffer'])
        }
    
    def get_full_transcript(self, session_id: str) -> Dict:
        """Get full transcript for session."""
        if session_id not in self.active_streams:
            return {'error': 'Stream not found'}
        
        stream = self.active_streams[session_id]
        
        return {
            'session_id': session_id,
            'full_transcript': stream['transcript_buffer'],
            'speaker_segments': stream['speaker_segments'],
            'stats': stream['stats']
        }
    
    def end_stream(self, session_id: str) -> Dict:
        """End transcription stream."""
        try:
            if session_id not in self.active_streams:
                return {'error': 'Stream not found'}
            
            stream = self.active_streams[session_id]
            stream['status'] = 'completed'
            stream['end_time'] = datetime.now().isoformat()
            
            # Generate final transcript
            final_transcript = self._generate_final_transcript(stream)
            
            # Remove from active streams
            completed_stream = self.active_streams.pop(session_id)
            
            logger.info(f"✅ Ended transcription stream: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'final_transcript': final_transcript,
                'stats': completed_stream['stats']
            }
            
        except Exception as e:
            logger.error(f"❌ Error ending stream: {e}")
            return {'error': str(e)}
    
    def _generate_final_transcript(self, stream: Dict) -> Dict:
        """Generate final transcript summary."""
        try:
            transcript_buffer = stream['transcript_buffer']
            speaker_segments = stream['speaker_segments']
            
            # Create chronological transcript
            chronological_transcript = []
            for entry in transcript_buffer:
                if entry['is_final']:
                    chronological_transcript.append({
                        'speaker': entry['speaker'],
                        'text': entry['text'],
                        'timestamp': entry['timestamp'],
                        'confidence': entry['confidence']
                    })
            
            # Create speaker-based summary
            speaker_summary = {}
            for segment in speaker_segments:
                speaker = segment['speaker']
                if speaker not in speaker_summary:
                    speaker_summary[speaker] = {
                        'total_words': 0,
                        'total_segments': 0,
                        'avg_confidence': 0.0,
                        'text_segments': []
                    }
                
                speaker_summary[speaker]['total_words'] += len(segment['text'].split())
                speaker_summary[speaker]['total_segments'] += 1
                speaker_summary[speaker]['text_segments'].append(segment['text'])
                
                # Update average confidence
                current_avg = speaker_summary[speaker]['avg_confidence']
                total_segments = speaker_summary[speaker]['total_segments']
                new_confidence = segment['confidence']
                
                speaker_summary[speaker]['avg_confidence'] = (
                    (current_avg * (total_segments - 1) + new_confidence) / total_segments
                )
            
            return {
                'chronological_transcript': chronological_transcript,
                'speaker_summary': speaker_summary,
                'total_duration': self._calculate_duration(stream),
                'total_words': stream['stats']['total_words'],
                'total_segments': stream['stats']['total_segments'],
                'avg_confidence': stream['stats']['avg_confidence'],
                'speaker_changes': stream['stats']['speaker_changes']
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating final transcript: {e}")
            return {}
    
    def _calculate_duration(self, stream: Dict) -> str:
        """Calculate stream duration."""
        try:
            start_time = datetime.fromisoformat(stream['start_time'])
            end_time = datetime.fromisoformat(stream.get('end_time', datetime.now().isoformat()))
            
            duration = end_time - start_time
            total_seconds = int(duration.total_seconds())
            
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
                
        except Exception as e:
            logger.error(f"❌ Error calculating duration: {e}")
            return "Unknown"
    
    def get_all_streams(self) -> List[Dict]:
        """Get all active streams."""
        return [
            {
                'session_id': session_id,
                'status': stream['status'],
                'start_time': stream['start_time'],
                'current_speaker': stream['current_speaker'],
                'stats': stream['stats']
            }
            for session_id, stream in self.active_streams.items()
        ]

# Global pipeline instance
streaming_pipeline = StreamingTranscriptionPipeline()

# Flask integration functions
def start_transcription_stream(session_id: str, config: Dict) -> Dict:
    """Start transcription stream."""
    return streaming_pipeline.start_stream(session_id, config)

def process_audio_data(session_id: str, audio_data: bytes, metadata: Dict = None):
    """Process audio data."""
    streaming_pipeline.process_audio_chunk(session_id, audio_data, metadata)

def end_transcription_stream(session_id: str) -> Dict:
    """End transcription stream."""
    return streaming_pipeline.end_stream(session_id)

def get_transcription_status(session_id: str) -> Dict:
    """Get transcription status."""
    return streaming_pipeline.get_stream_status(session_id)

def get_active_transcription_streams() -> List[Dict]:
    """Get all active transcription streams."""
    return streaming_pipeline.get_all_streams()

# WebSocket integration for real-time updates
async def handle_audio_websocket(websocket, path):
    """Handle WebSocket connections for audio streaming."""
    session_id = None
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'start_stream':
                session_id = data['session_id']
                config = data.get('config', {})
                result = start_transcription_stream(session_id, config)
                await websocket.send(json.dumps(result))
                
            elif data['type'] == 'audio_data':
                if session_id:
                    # Decode base64 audio data
                    audio_data = base64.b64decode(data['audio'])
                    metadata = data.get('metadata', {})
                    process_audio_data(session_id, audio_data, metadata)
                    
            elif data['type'] == 'end_stream':
                if session_id:
                    result = end_transcription_stream(session_id)
                    await websocket.send(json.dumps(result))
                    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        if session_id:
            try:
                end_transcription_stream(session_id)
            except:
                pass

if __name__ == "__main__":
    # Test the streaming pipeline
    print("🚀 Testing Streaming Transcription Pipeline...")
    
    # Start pipeline
    streaming_pipeline.start_pipeline()
    
    # Test stream
    test_config = {
        'language': 'en-US',
        'sample_rate': 16000,
        'speaker_diarization': True,
        'max_speakers': 2
    }
    
    result = streaming_pipeline.start_stream("test_stream", test_config)
    print(f"✅ Test stream started: {result}")
    
    # Simulate audio processing
    for i in range(5):
        audio_data = b"fake_audio_data_" + str(i).encode()
        streaming_pipeline.process_audio_chunk("test_stream", audio_data)
        time.sleep(1)
    
    # Get status
    status = streaming_pipeline.get_stream_status("test_stream")
    print(f"📊 Stream status: {status['stats']}")
    
    # End stream
    final_result = streaming_pipeline.end_stream("test_stream")
    print(f"✅ Stream ended: {final_result['success']}")
    
    # Stop pipeline
    streaming_pipeline.stop_pipeline()
    
    print("🎉 Streaming Transcription Pipeline test completed!") 