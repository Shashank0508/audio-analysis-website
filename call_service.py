import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime
import uuid
import logging
from dotenv import load_dotenv

# Load environment variables with explicit path
load_dotenv(override=True)

class CallService:
    def __init__(self):
        """Initialize Twilio client with credentials from environment variables"""
        # Load environment variables
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        if not all([self.account_sid, self.auth_token, self.from_number]):
            raise ValueError("Missing Twilio credentials in environment variables")
        
        self.client = Client(self.account_sid, self.auth_token)
            
        self.active_calls = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initiate_call(self, to_number, webhook_base_url):
        """
        Initiate an outbound call
        
        Args:
            to_number (str): Phone number to call (with country code)
            webhook_base_url (str): Base URL for webhooks
            
        Returns:
            dict: Call result with success status and call_sid
        """
        try:
            # Generate unique call ID for tracking
            call_uuid = str(uuid.uuid4())
            
            # Create TwiML URL for call handling
            twiml_url = f"{webhook_base_url}/webhook/call/{call_uuid}"
            status_callback_url = f"{webhook_base_url}/webhook/status/{call_uuid}"
            recording_callback_url = f"{webhook_base_url}/webhook/recording/{call_uuid}"
            
            # Initiate the call
            call = self.client.calls.create(
                to=to_number,
                from_=self.from_number,
                url=twiml_url,
                method='POST',
                status_callback=status_callback_url,
                status_callback_method='POST',
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                record=True,
                recording_channels='dual',  # Record both sides separately
                recording_status_callback=recording_callback_url,
                recording_status_callback_method='POST',
                timeout=30,  # 30 seconds timeout for connection
                machine_detection='Enable',  # Detect answering machines
                machine_detection_timeout=30
            )
            
            # Store call information
            call_info = {
                'call_sid': call.sid,
                'call_uuid': call_uuid,
                'to_number': to_number,
                'from_number': self.from_number,
                'status': 'initiated',
                'start_time': datetime.now(),
                'recording_url': None
            }
            
            self.active_calls[call_uuid] = call_info
            
            self.logger.info(f"Call initiated: {call.sid} to {to_number}")
            
            return {
                'success': True,
                'call_sid': call.sid,
                'call_uuid': call_uuid,
                'status': 'initiated',
                'message': 'Call initiated successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initiate call to {to_number}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to initiate call'
            }
    
    def end_call(self, call_sid):
        """
        End an active call
        
        Args:
            call_sid (str): Twilio call SID
            
        Returns:
            dict: Result of call termination
        """
        try:
            call = self.client.calls(call_sid).update(status='completed')
            
            self.logger.info(f"Call ended: {call_sid}")
            
            return {
                'success': True,
                'call_sid': call_sid,
                'status': 'completed',
                'message': 'Call ended successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to end call {call_sid}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to end call'
            }
    
    def get_call_status(self, call_sid):
        """
        Get current status of a call
        
        Args:
            call_sid (str): Twilio call SID
            
        Returns:
            dict: Call status information
        """
        try:
            call = self.client.calls(call_sid).fetch()
            
            return {
                'success': True,
                'call_sid': call_sid,
                'status': call.status,
                'duration': call.duration,
                'start_time': call.start_time,
                'end_time': call.end_time
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get call status for {call_sid}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to get call status'
            }
    
    def generate_call_twiml(self, call_uuid):
        """Generate TwiML for call handling with transcription"""
        try:
            from twilio.twiml.voice_response import VoiceResponse
            
            response = VoiceResponse()
            
            # Simple greeting
            response.say(
                "Hello! Thank you for calling. Your call is being recorded.",
                voice='Polly.Joanna',
                language='en-US'
            )
            
            # Get webhook base URL
            webhook_base_url = os.getenv('WEBHOOK_BASE_URL', 'https://your-domain.com')
            print(f"🔥 Using webhook base URL: {webhook_base_url}")
            
            # Validate webhook URL
            if 'your-domain.com' in webhook_base_url:
                # Fallback to simple recording without webhooks
                response.record(
                    max_length=3600,
                    play_beep=False,
                    record_on_hangup=True
                )
            else:
                # Full recording with webhooks
                response.record(
                    action=f'{webhook_base_url}/webhook/recording/{call_uuid}',
                    method='POST',
                    max_length=3600,
                    play_beep=False,
                    record_on_hangup=True,
                    transcribe=True,
                    transcribe_callback=f'{webhook_base_url}/webhook/transcription/{call_uuid}',
                    dual_channel=True
                )
            
            # Allow conversation
            response.say(
                "Please speak now. Press any key to end the call.",
                voice='Polly.Joanna',
                language='en-US'
            )
            
            # Add a long pause to allow conversation
            response.pause(length=30)
            
            # End call gracefully
            response.say(
                "Thank you for your call. Goodbye!",
                voice='Polly.Joanna',
                language='en-US'
            )
            
            response.hangup()
            
            # Trigger AWS Transcribe streaming session
            self._trigger_aws_transcribe_streaming(call_uuid)
            
            return str(response)
            
        except Exception as e:
            self.logger.error(f"TwiML generation error: {str(e)}")
            # Fallback TwiML
            from twilio.twiml.voice_response import VoiceResponse
            response = VoiceResponse()
            response.say("Hello! Please hold while we connect your call.")
            response.record(max_length=300)
            return str(response)
    
    def _trigger_aws_transcribe_streaming(self, call_uuid):
        """Trigger AWS Transcribe streaming session for call"""
        try:
            # This would typically be called via WebSocket or API
            # For now, we'll emit a SocketIO event to trigger frontend
            import requests
            
            # Trigger streaming session
            streaming_config = {
                'call_uuid': call_uuid,
                'language': 'en-US',
                'enable_diarization': True,
                'max_speakers': 4
            }
            
            # Send to AWS Transcribe service
            try:
                response = requests.post(
                    'http://localhost:5000/api/aws/transcribe/start',
                    json=streaming_config,
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.logger.info(f"AWS Transcribe streaming triggered for call: {call_uuid}")
                else:
                    self.logger.warning(f"Failed to trigger AWS Transcribe: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Could not trigger AWS Transcribe streaming: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error triggering AWS Transcribe streaming: {str(e)}")
    
    def enable_real_time_transcription(self, call_uuid, language='en-US'):
        """Enable real-time transcription for a call"""
        try:
            if call_uuid not in self.active_calls:
                return {'success': False, 'error': 'Call not found'}
            
            # Update call with transcription settings
            self.active_calls[call_uuid].update({
                'real_time_transcription': True,
                'transcription_language': language,
                'transcription_started': datetime.now(),
                'transcription_session_id': None
            })
            
            # Trigger AWS Transcribe streaming
            self._trigger_aws_transcribe_streaming(call_uuid)
            
            return {
                'success': True,
                'call_uuid': call_uuid,
                'transcription_enabled': True,
                'language': language
            }
            
        except Exception as e:
            self.logger.error(f"Error enabling real-time transcription: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def disable_real_time_transcription(self, call_uuid):
        """Disable real-time transcription for a call"""
        try:
            if call_uuid not in self.active_calls:
                return {'success': False, 'error': 'Call not found'}
            
            # Update call settings
            self.active_calls[call_uuid].update({
                'real_time_transcription': False,
                'transcription_ended': datetime.now()
            })
            
            return {
                'success': True,
                'call_uuid': call_uuid,
                'transcription_enabled': False
            }
            
        except Exception as e:
            self.logger.error(f"Error disabling real-time transcription: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_transcription_status(self, call_uuid):
        """Get transcription status for a call"""
        try:
            if call_uuid not in self.active_calls:
                return {'success': False, 'error': 'Call not found'}
            
            call_data = self.active_calls[call_uuid]
            
            return {
                'success': True,
                'call_uuid': call_uuid,
                'transcription_enabled': call_data.get('real_time_transcription', False),
                'language': call_data.get('transcription_language', 'en-US'),
                'session_id': call_data.get('transcription_session_id'),
                'started': call_data.get('transcription_started'),
                'ended': call_data.get('transcription_ended')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting transcription status: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def update_call_status(self, call_uuid, status, **kwargs):
        """
        Update call status in memory
        
        Args:
            call_uuid (str): Unique call identifier
            status (str): New call status
            **kwargs: Additional call information
        """
        if call_uuid in self.active_calls:
            self.active_calls[call_uuid]['status'] = status
            self.active_calls[call_uuid].update(kwargs)
            
            self.logger.info(f"Call {call_uuid} status updated to {status}")
    
    def get_call_info(self, call_uuid):
        """
        Get call information by UUID
        
        Args:
            call_uuid (str): Unique call identifier
            
        Returns:
            dict: Call information or None if not found
        """
        return self.active_calls.get(call_uuid)
    
    def validate_phone_number(self, phone_number):
        """
        Validate phone number format
        
        Args:
            phone_number (str): Phone number to validate
            
        Returns:
            dict: Validation result
        """
        try:
            # Remove all non-digit characters except +
            cleaned_number = ''.join(c for c in phone_number if c.isdigit() or c == '+')
            
            # Check if number starts with + and has country code
            if not cleaned_number.startswith('+'):
                # Assume US number if no country code
                if len(cleaned_number) == 10:
                    cleaned_number = '+1' + cleaned_number
                else:
                    return {
                        'valid': False,
                        'error': 'Invalid phone number format. Use +1234567890 format.'
                    }
            
            # Basic length validation (minimum 10 digits)
            if len(cleaned_number) < 12:  # +1 + 10 digits minimum
                return {
                    'valid': False,
                    'error': 'Phone number too short'
                }
            
            return {
                'valid': True,
                'formatted_number': cleaned_number
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Phone number validation failed: {str(e)}'
            } 
