import asyncio
import logging
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List
import tempfile
import requests
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TARGET_HEIGHT = 1920  # 9:16 aspect ratio for Instagram Reels
TARGET_WIDTH = 1080
MAX_CLIP_DURATION = 60  # seconds
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY environment variable is not set")


def get_video_stream_url(video_id: str) -> str:
    """
    Fetch the video streaming URL using the YouTube Data API.
    """
    api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={YOUTUBE_API_KEY}&part=contentDetails"
    response = requests.get(api_url)
    if response.status_code != 200:
        error_message = response.json().get("error", {}).get("message", "Unknown error")
        logger.error(f"Failed to fetch video details: {error_message}")
        raise RuntimeError(f"Failed to fetch video details: {error_message}")
    
    video_data = response.json()
    if "items" not in video_data or not video_data["items"]:
        raise ValueError("Invalid video ID or no video found.")
    
    streaming_url = f"https://www.youtube.com/watch?v={video_id}"
    return streaming_url


def download_and_process_video(stream_url: str, output_path: str):
    """
    Download and process the video using FFmpeg.
    """
    try:
        ffmpeg_command = [
            "ffmpeg",
            "-i", stream_url,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}",
            output_path
        ]
        logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {str(e)}")
        raise RuntimeError("Failed to download or process video.")


async def process_youtube_video(youtube_url: str, temp_dir: Path, output_path: str):
    """
    Process a YouTube video using the YouTube Data API and FFmpeg.
    """
    temp_video_path = temp_dir / "original.mp4"
    try:
        video_id = youtube_url.split("v=")[-1]
        stream_url = get_video_stream_url(video_id)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: download_and_process_video(stream_url, str(temp_video_path))
        )
        shutil.move(temp_video_path, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error processing YouTube video: {str(e)}")
        raise
    finally:
        if temp_video_path.exists():
            temp_video_path.unlink()
