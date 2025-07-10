import os
import json
import hmac
import hashlib
from datetime import datetime
import logging
from flask import request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WebhookService:
    def __init__(self, call_service, recording_service, socketio):
        """
        Initialize webhook service
        
        Args:
            call_service: Instance of CallService
            recording_service: Instance of RecordingService
            socketio: Flask-SocketIO instance
        """
        self.call_service = call_service
        self.recording_service = recording_service
        self.socketio = socketio
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Event handlers mapping
        self.call_status_handlers = {
            'queued': self._handle_call_queued,
            'initiated': self._handle_call_initiated,
            'ringing': self._handle_call_ringing,
            'in-progress': self._handle_call_in_progress,
            'completed': self._handle_call_completed,
            'busy': self._handle_call_busy,
            'failed': self._handle_call_failed,
            'no-answer': self._handle_call_no_answer,
            'canceled': self._handle_call_canceled
        }
        
        # Recording status handlers
        self.recording_status_handlers = {
            'in-progress': self._handle_recording_in_progress,
            'completed': self._handle_recording_completed,
            'failed': self._handle_recording_failed,
            'absent': self._handle_recording_absent
        }
    
    def validate_webhook_signature(self, request_url, post_data, signature):
        """
        Validate Twilio webhook signature for security
        
        Args:
            request_url (str): The full URL of the request
            post_data (dict): POST data from the webhook
            signature (str): X-Twilio-Signature header
            
        Returns:
            bool: True if signature is valid
        """
        try:
            # Create the signature
            data = request_url
            for key in sorted(post_data.keys()):
                data += key + post_data[key]
            
            # Generate expected signature
            expected_signature = hmac.new(
                self.auth_token.encode('utf-8'),
                data.encode('utf-8'),
                hashlib.sha1
            ).digest()
            
            # Convert to base64
            import base64
            expected_signature_b64 = base64.b64encode(expected_signature).decode()
            
            return signature == expected_signature_b64
            
        except Exception as e:
            self.logger.error(f"Webhook signature validation error: {str(e)}")
            return False
    
    def handle_call_status_webhook(self, call_uuid, request_data):
        """
        Handle call status webhook events
        
        Args:
            call_uuid (str): Unique call identifier
            request_data (dict): Webhook request data
            
        Returns:
            dict: Processing result
        """
        try:
            call_status = request_data.get('CallStatus')
            call_sid = request_data.get('CallSid')
            call_duration = request_data.get('CallDuration', '0')
            from_number = request_data.get('From')
            to_number = request_data.get('To')
            direction = request_data.get('Direction')
            answered_by = request_data.get('AnsweredBy')
            
            # Log the event
            self.logger.info(f"Call status webhook: {call_uuid} - {call_status}")
            
            # Update call information
            call_info = {
                'duration': call_duration,
                'from_number': from_number,
                'to_number': to_number,
                'direction': direction,
                'answered_by': answered_by,
                'last_updated': datetime.now()
            }
            
            # Update call service
            self.call_service.update_call_status(call_uuid, call_status, **call_info)
            
            # Handle specific status
            if call_status in self.call_status_handlers:
                result = self.call_status_handlers[call_status](call_uuid, request_data)
            else:
                result = self._handle_unknown_call_status(call_uuid, call_status, request_data)
            
            # Emit real-time update
            self.socketio.emit('call_status_update', {
                'call_uuid': call_uuid,
                'call_sid': call_sid,
                'status': call_status,
                'duration': call_duration,
                'answered_by': answered_by,
                'timestamp': datetime.now().isoformat()
            })
            
            return {'success': True, 'status': call_status, 'result': result}
            
        except Exception as e:
            self.logger.error(f"Call status webhook error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def handle_recording_status_webhook(self, call_uuid, request_data):
        """
        Handle recording status webhook events
        
        Args:
            call_uuid (str): Unique call identifier
            request_data (dict): Webhook request data
            
        Returns:
            dict: Processing result
        """
        try:
            recording_status = request_data.get('RecordingStatus')
            recording_sid = request_data.get('RecordingSid')
            recording_url = request_data.get('RecordingUrl')
            recording_duration = request_data.get('RecordingDuration', '0')
            call_sid = request_data.get('CallSid')
            
            self.logger.info(f"Recording status webhook: {call_uuid} - {recording_status}")
            
            # Handle specific recording status
            if recording_status in self.recording_status_handlers:
                result = self.recording_status_handlers[recording_status](
                    call_uuid, request_data
                )
            else:
                result = self._handle_unknown_recording_status(
                    call_uuid, recording_status, request_data
                )
            
            # Emit real-time update
            self.socketio.emit('recording_status_update', {
                'call_uuid': call_uuid,
                'call_sid': call_sid,
                'recording_sid': recording_sid,
                'status': recording_status,
                'duration': recording_duration,
                'url': recording_url,
                'timestamp': datetime.now().isoformat()
            })
            
            return {'success': True, 'status': recording_status, 'result': result}
            
        except Exception as e:
            self.logger.error(f"Recording status webhook error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def handle_transcription_webhook(self, call_uuid, request_data):
        """
        Handle transcription webhook events
        
        Args:
            call_uuid (str): Unique call identifier
            request_data (dict): Webhook request data
            
        Returns:
            dict: Processing result
        """
        try:
            transcription_text = request_data.get('TranscriptionText', '')
            transcription_status = request_data.get('TranscriptionStatus')
            transcription_sid = request_data.get('TranscriptionSid')
            recording_sid = request_data.get('RecordingSid')
            
            self.logger.info(f"Transcription webhook: {call_uuid} - {transcription_status}")
            
            # Store transcription data
            transcription_data = {
                'transcription_text': transcription_text,
                'transcription_status': transcription_status,
                'transcription_sid': transcription_sid,
                'recording_sid': recording_sid,
                'timestamp': datetime.now()
            }
            
            # Update call service with transcription
            self.call_service.update_call_status(call_uuid, 'transcribed', **transcription_data)
            
            # Emit real-time transcription
            self.socketio.emit('live_transcription', {
                'call_uuid': call_uuid,
                'transcription': transcription_text,
                'status': transcription_status,
                'recording_sid': recording_sid,
                'timestamp': datetime.now().isoformat()
            })
            
            # If transcription is completed, trigger analysis
            if transcription_status == 'completed' and transcription_text:
                self.socketio.emit('transcription_complete', {
                    'call_uuid': call_uuid,
                    'transcription': transcription_text,
                    'ready_for_analysis': True
                })
            
            return {'success': True, 'transcription_status': transcription_status}
            
        except Exception as e:
            self.logger.error(f"Transcription webhook error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # Call Status Handlers
    def _handle_call_queued(self, call_uuid, request_data):
        """Handle queued call status"""
        self.socketio.emit('call_queued', {
            'call_uuid': call_uuid,
            'message': 'Call has been queued for processing'
        })
        return {'status': 'queued', 'action': 'waiting_for_initiation'}
    
    def _handle_call_initiated(self, call_uuid, request_data):
        """Handle initiated call status"""
        self.socketio.emit('call_initiated', {
            'call_uuid': call_uuid,
            'message': 'Call has been initiated'
        })
        return {'status': 'initiated', 'action': 'connecting'}
    
    def _handle_call_ringing(self, call_uuid, request_data):
        """Handle ringing call status"""
        self.socketio.emit('call_ringing', {
            'call_uuid': call_uuid,
            'message': 'Phone is ringing'
        })
        return {'status': 'ringing', 'action': 'waiting_for_answer'}
    
    def _handle_call_in_progress(self, call_uuid, request_data):
        """Handle in-progress call status"""
        answered_by = request_data.get('AnsweredBy', 'human')
        
        self.socketio.emit('call_answered', {
            'call_uuid': call_uuid,
            'answered_by': answered_by,
            'message': f'Call answered by {answered_by}'
        })
        
        # Start recording indicator
        self.socketio.emit('recording_started', {
            'call_uuid': call_uuid,
            'message': 'Call recording has started'
        })
        
        return {'status': 'in_progress', 'action': 'recording', 'answered_by': answered_by}
    
    def _handle_call_completed(self, call_uuid, request_data):
        """Handle completed call status"""
        duration = request_data.get('CallDuration', '0')
        
        self.socketio.emit('call_completed', {
            'call_uuid': call_uuid,
            'duration': duration,
            'message': f'Call completed after {duration} seconds'
        })
        
        return {'status': 'completed', 'action': 'processing_recording', 'duration': duration}
    
    def _handle_call_busy(self, call_uuid, request_data):
        """Handle busy call status"""
        self.socketio.emit('call_busy', {
            'call_uuid': call_uuid,
            'message': 'Number is busy'
        })
        return {'status': 'busy', 'action': 'retry_later'}
    
    def _handle_call_failed(self, call_uuid, request_data):
        """Handle failed call status"""
        self.socketio.emit('call_failed', {
            'call_uuid': call_uuid,
            'message': 'Call failed to connect'
        })
        return {'status': 'failed', 'action': 'check_number'}
    
    def _handle_call_no_answer(self, call_uuid, request_data):
        """Handle no-answer call status"""
        self.socketio.emit('call_no_answer', {
            'call_uuid': call_uuid,
            'message': 'No answer received'
        })
        return {'status': 'no_answer', 'action': 'try_again'}
    
    def _handle_call_canceled(self, call_uuid, request_data):
        """Handle canceled call status"""
        self.socketio.emit('call_canceled', {
            'call_uuid': call_uuid,
            'message': 'Call was canceled'
        })
        return {'status': 'canceled', 'action': 'none'}
    
    def _handle_unknown_call_status(self, call_uuid, status, request_data):
        """Handle unknown call status"""
        self.logger.warning(f"Unknown call status: {status} for call {call_uuid}")
        self.socketio.emit('call_status_unknown', {
            'call_uuid': call_uuid,
            'status': status,
            'message': f'Unknown call status: {status}'
        })
        return {'status': status, 'action': 'monitor'}
    
    # Recording Status Handlers
    def _handle_recording_in_progress(self, call_uuid, request_data):
        """Handle recording in progress"""
        self.socketio.emit('recording_in_progress', {
            'call_uuid': call_uuid,
            'message': 'Recording is in progress'
        })
        return {'status': 'in_progress', 'action': 'monitoring'}
    
    def _handle_recording_completed(self, call_uuid, request_data):
        """Handle recording completed"""
        recording_url = request_data.get('RecordingUrl')
        recording_sid = request_data.get('RecordingSid')
        call_sid = request_data.get('CallSid')
        
        if recording_url:
            # Automatically download the recording
            audio_recording_url = recording_url + '.wav'
            download_result = self.recording_service.download_recording(
                audio_recording_url, call_uuid, call_sid
            )
            
            if download_result['success']:
                self.socketio.emit('recording_downloaded', {
                    'call_uuid': call_uuid,
                    'filename': download_result['filename'],
                    'file_size': download_result['file_size'],
                    'message': 'Recording downloaded successfully'
                })
                
                # Trigger analysis
                self.socketio.emit('start_analysis', {
                    'call_uuid': call_uuid,
                    'file_path': download_result['local_path'],
                    'filename': download_result['filename']
                })
            else:
                self.socketio.emit('recording_download_failed', {
                    'call_uuid': call_uuid,
                    'error': download_result['error']
                })
        
        return {'status': 'completed', 'action': 'downloaded', 'recording_sid': recording_sid}
    
    def _handle_recording_failed(self, call_uuid, request_data):
        """Handle recording failed"""
        self.socketio.emit('recording_failed', {
            'call_uuid': call_uuid,
            'message': 'Recording failed'
        })
        return {'status': 'failed', 'action': 'retry_call'}
    
    def _handle_recording_absent(self, call_uuid, request_data):
        """Handle recording absent"""
        self.socketio.emit('recording_absent', {
            'call_uuid': call_uuid,
            'message': 'No recording available for this call'
        })
        return {'status': 'absent', 'action': 'no_recording'}
    
    def _handle_unknown_recording_status(self, call_uuid, status, request_data):
        """Handle unknown recording status"""
        self.logger.warning(f"Unknown recording status: {status} for call {call_uuid}")
        self.socketio.emit('recording_status_unknown', {
            'call_uuid': call_uuid,
            'status': status,
            'message': f'Unknown recording status: {status}'
        })
        return {'status': status, 'action': 'monitor'}
    
    def log_webhook_event(self, event_type, call_uuid, request_data):
        """
        Log webhook events for debugging and monitoring
        
        Args:
            event_type (str): Type of webhook event
            call_uuid (str): Unique call identifier
            request_data (dict): Webhook request data
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'call_uuid': call_uuid,
            'data': request_data
        }
        
        # Log to file (optional)
        log_file = 'webhook_events.log'
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write webhook log: {str(e)}")
        
        # Log to console
        self.logger.info(f"Webhook event logged: {event_type} - {call_uuid}")
    
    def get_webhook_statistics(self):
        """
        Get webhook processing statistics
        
        Returns:
            dict: Webhook statistics
        """
        try:
            # This would typically read from a database or log files
            # For now, return basic info
            return {
                'total_webhooks_processed': 0,  # Would track actual count
                'call_events': 0,
                'recording_events': 0,
                'transcription_events': 0,
                'last_webhook': datetime.now().isoformat(),
                'status': 'active'
            }
        except Exception as e:
            self.logger.error(f"Failed to get webhook statistics: {str(e)}")
            return {'error': str(e)} 