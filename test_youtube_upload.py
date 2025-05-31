#!/usr/bin/env python3
"""
YouTube Upload Test Script

This script is used to test the YouTube upload functionality using a sample video.
It simulates the final product upload without going through the full restoration process.
It can also test thumbnail generation separately.
"""

import os
import sys
from pathlib import Path
import argparse
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Add the parent directory to sys.path to allow importing from YouTubeApi
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))

# Import the YouTube uploader and thumbnail generator
from YouTubeApi.youtube_uploader import YouTubeUploader
from YouTubeApi.secrets_manager import get_client_secrets_file
from YouTubeApi.thumbnail_generator import generate_thumbnail

def upload_test_video(video_path=None, title=None, description=None, privacy="unlisted"):
    """
    Upload a test video to YouTube
    
    Args:
        video_path (str): Path to the video file to upload
        title (str): Title of the video (optional)
        description (str): Description of the video (optional)
        privacy (str): Privacy setting (public, unlisted, private)
    
    Returns:
        str: YouTube video ID if successful, None if failed
    """
    print("[INFO] Starting YouTube upload test...")
    
    # If no video path provided, use the default test video
    if not video_path:
        video_path = Path(__file__).parent / "TestYouTube" / "Dr. Martin Luther King Jr. I have a Dream Speech.mp4"
    else:
        video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"[ERROR] Test video not found at: {video_path}")
        return None
    
    print(f"[INFO] Using test video: {video_path}")
    
    # Get client secrets file from environment variables or file
    client_secret = get_client_secrets_file()
    
    if not client_secret.exists():
        print(f"[ERROR] Client secrets not found. Check your .env file.")
        return None
    
    # Create YouTube uploader
    uploader = YouTubeUploader(client_secret)
    
    # Generate title based on filename if not provided
    if not title:
        title = f"{video_path.stem} - AI Restored and Colorized"
    
    # Default description if not provided
    if not description:
        description = """This historical video has been restored and colorized using AI technology.

The Full Restore project aims to make moments from the past feel more real and accessible through modern technology.

Check out the repository here to do it yourself: https://github.com/MikahNiehaus/full_restore

Check out my LinkedIn to learn more about me:
https://www.linkedin.com/in/mikahniehaus/

Thanks for watching and let's keep history alive together.

#AI #History #VideoRestoration #DeepLearning #AIColorization #HistoricalFootage"""    # Upload to YouTube
    try:
        print("[INFO] Uploading video to YouTube...")
        video_id = uploader.upload_video(
            str(video_path),
            title=title,
            description=description,
            privacy_status=privacy
        )
        
        if video_id:
            print(f"[SUCCESS] Video uploaded to YouTube: https://youtu.be/{video_id}")
            return video_id
        else:
            print("[ERROR] Upload failed")
            return None
            
    except Exception as e:
        print(f"[ERROR] YouTube upload error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YouTube upload functionality.")
    parser.add_argument("--video", "-v", help="Path to video file to upload (defaults to TestYouTube directory video)")
    parser.add_argument("--title", "-t", help="Title for the uploaded video")
    parser.add_argument("--description", "-d", help="Description for the uploaded video")
    parser.add_argument("--privacy", "-p", choices=["public", "unlisted", "private"], 
                       default="unlisted", help="Privacy setting (default: unlisted)")
    
    args = parser.parse_args()
    
    upload_test_video(
        video_path=args.video,
        title=args.title,
        description=args.description,
        privacy=args.privacy
    )
