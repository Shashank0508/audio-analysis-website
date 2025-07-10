import os
import requests
import tempfile
from datetime import datetime
import logging
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv()

class RecordingService:
    def __init__(self):
        """Initialize recording service with Twilio client"""
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        
        if not all([self.account_sid, self.auth_token]):
            raise ValueError("Missing Twilio credentials for recording service")
        
        self.client = Client(self.account_sid, self.auth_token)
        self.recordings_folder = 'recordings'
        
        # Create recordings directory if it doesn't exist
        os.makedirs(self.recordings_folder, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_recording(self, recording_url, call_uuid, call_sid):
        """
        Download recording from Twilio
        
        Args:
            recording_url (str): Twilio recording URL
            call_uuid (str): Unique call identifier
            call_sid (str): Twilio call SID
            
        Returns:
            dict: Download result with local file path
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"call_{call_uuid}_{timestamp}.wav"
            local_path = os.path.join(self.recordings_folder, filename)
            
            # Download the recording
            response = requests.get(recording_url, auth=(self.account_sid, self.auth_token))
            response.raise_for_status()
            
            # Save to local file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(local_path)
            
            self.logger.info(f"Recording downloaded: {filename} ({file_size} bytes)")
            
            return {
                'success': True,
                'local_path': local_path,
                'filename': filename,
                'file_size': file_size,
                'call_uuid': call_uuid,
                'call_sid': call_sid,
                'download_time': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to download recording for call {call_uuid}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'call_uuid': call_uuid,
                'call_sid': call_sid
            }
    
    def get_recording_info(self, recording_sid):
        """
        Get recording information from Twilio
        
        Args:
            recording_sid (str): Twilio recording SID
            
        Returns:
            dict: Recording information
        """
        try:
            recording = self.client.recordings(recording_sid).fetch()
            
            return {
                'success': True,
                'recording_sid': recording_sid,
                'call_sid': recording.call_sid,
                'status': recording.status,
                'duration': recording.duration,
                'date_created': recording.date_created,
                'date_updated': recording.date_updated,
                'uri': recording.uri,
                'media_url': f"https://api.twilio.com{recording.uri.replace('.json', '.wav')}",
                'channels': recording.channels,
                'source': recording.source
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recording info for {recording_sid}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'recording_sid': recording_sid
            }
    
    def list_call_recordings(self, call_sid):
        """
        List all recordings for a specific call
        
        Args:
            call_sid (str): Twilio call SID
            
        Returns:
            list: List of recordings for the call
        """
        try:
            recordings = self.client.recordings.list(call_sid=call_sid)
            
            recording_list = []
            for recording in recordings:
                recording_list.append({
                    'recording_sid': recording.sid,
                    'call_sid': recording.call_sid,
                    'status': recording.status,
                    'duration': recording.duration,
                    'date_created': recording.date_created,
                    'date_updated': recording.date_updated,
                    'uri': recording.uri,
                    'media_url': f"https://api.twilio.com{recording.uri.replace('.json', '.wav')}",
                    'channels': recording.channels,
                    'source': recording.source
                })
            
            return {
                'success': True,
                'call_sid': call_sid,
                'recordings': recording_list,
                'count': len(recording_list)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list recordings for call {call_sid}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'call_sid': call_sid
            }
    
    def delete_recording(self, recording_sid):
        """
        Delete a recording from Twilio
        
        Args:
            recording_sid (str): Twilio recording SID
            
        Returns:
            dict: Deletion result
        """
        try:
            self.client.recordings(recording_sid).delete()
            
            self.logger.info(f"Recording deleted: {recording_sid}")
            
            return {
                'success': True,
                'recording_sid': recording_sid,
                'message': 'Recording deleted successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to delete recording {recording_sid}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'recording_sid': recording_sid
            }
    
    def get_local_recordings(self):
        """
        Get list of locally stored recordings
        
        Returns:
            list: List of local recording files
        """
        try:
            local_recordings = []
            
            if os.path.exists(self.recordings_folder):
                for filename in os.listdir(self.recordings_folder):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(self.recordings_folder, filename)
                        file_stats = os.stat(file_path)
                        
                        local_recordings.append({
                            'filename': filename,
                            'file_path': file_path,
                            'file_size': file_stats.st_size,
                            'created_time': datetime.fromtimestamp(file_stats.st_ctime),
                            'modified_time': datetime.fromtimestamp(file_stats.st_mtime)
                        })
            
            return {
                'success': True,
                'recordings': local_recordings,
                'count': len(local_recordings),
                'folder': self.recordings_folder
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get local recordings: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cleanup_old_recordings(self, days_old=30):
        """
        Clean up recordings older than specified days
        
        Args:
            days_old (int): Number of days after which to delete recordings
            
        Returns:
            dict: Cleanup result
        """
        try:
            deleted_count = 0
            current_time = datetime.now()
            
            if os.path.exists(self.recordings_folder):
                for filename in os.listdir(self.recordings_folder):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(self.recordings_folder, filename)
                        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        # Check if file is older than specified days
                        if (current_time - file_modified).days > days_old:
                            os.remove(file_path)
                            deleted_count += 1
                            self.logger.info(f"Deleted old recording: {filename}")
            
            return {
                'success': True,
                'deleted_count': deleted_count,
                'message': f'Cleaned up {deleted_count} old recordings'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old recordings: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_recording_access(self, recording_sid):
        """
        Validate if recording exists and is accessible
        
        Args:
            recording_sid (str): Twilio recording SID
            
        Returns:
            dict: Validation result
        """
        try:
            recording = self.client.recordings(recording_sid).fetch()
            
            return {
                'valid': True,
                'recording_sid': recording_sid,
                'status': recording.status,
                'duration': recording.duration
            }
            
        except Exception as e:
            return {
                'valid': False,
                'recording_sid': recording_sid,
                'error': str(e)
            } 