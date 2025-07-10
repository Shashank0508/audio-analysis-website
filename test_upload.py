#!/usr/bin/env python3
"""
Simple test script to verify upload endpoint functionality
"""
import requests
import os
import sys

def test_upload_endpoint():
    """Test the upload endpoint"""
    try:
        # Test file path (use an existing file from uploads)
        test_files = [
            'uploads/20250709_014121_3ce0c55c.mp3',
            'uploads/20250709_015654_a5cdfd6f.mp3'
        ]
        
        # Find an existing test file
        test_file = None
        for file_path in test_files:
            if os.path.exists(file_path):
                test_file = file_path
                break
        
        if not test_file:
            print("❌ No test file found in uploads directory")
            return False
        
        print(f"🧪 Testing upload with file: {test_file}")
        
        # Test upload endpoint
        url = 'http://localhost:5000/upload'
        
        with open(test_file, 'rb') as f:
            files = {'audio': f}
            response = requests.post(url, files=files)
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Body: {response.text}")
        
        if response.status_code == 200:
            print("✅ Upload endpoint working correctly!")
            return True
        else:
            print("❌ Upload endpoint failed!")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server. Make sure it's running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == '__main__':
    print("🚀 Testing upload endpoint...")
    success = test_upload_endpoint()
    sys.exit(0 if success else 1) 