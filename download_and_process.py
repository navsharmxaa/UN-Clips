import asyncio
import logging
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"  # Replace with your YouTube API key


def get_video_stream_url(video_id: str) -> str:
    """
    Fetch the video streaming URL using the YouTube Data API.
    """
    api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={YOUTUBE_API_KEY}&part=contentDetails"
    response = requests.get(api_url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch video details: {response.text}")
    
    video_data = response.json()
    if "items" not in video_data or not video_data["items"]:
        raise ValueError("Invalid video ID or no video found.")
    
    # Extract the streaming URL (requires additional parsing or use of DASH manifest)
    # For simplicity, assume we have a direct URL (you may need to handle adaptive streams)
    streaming_url = f"https://www.youtube.com/watch?v={video_id}"
    return streaming_url


def download_and_process_video(stream_url: str, output_path: str):
    """
    Download and process the video using FFmpeg.
    """
    try:
        # FFmpeg command to download and process the video
        ffmpeg_command = [
            "ffmpeg",
            "-i", stream_url,  # Input stream URL
            "-c:v", "libx264",  # Video codec
            "-c:a", "aac",      # Audio codec
            "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}",  # Resize to target dimensions
            output_path         # Output file
        ]
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {str(e)}")
        raise RuntimeError("Failed to download or process video.")


async def process_youtube_video(youtube_url: str, temp_dir: Path, output_path: str):
    """
    Process a YouTube video using the YouTube Data API and FFmpeg.
    """
    try:
        # Extract video ID from the URL
        video_id = youtube_url.split("v=")[-1]
        
        # Get the streaming URL
        stream_url = get_video_stream_url(video_id)
        
        # Temporary file path
        temp_video_path = temp_dir / "original.mp4"
        
        # Download and process the video
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: download_and_process_video(stream_url, str(temp_video_path))
        )
        
        # Further processing (e.g., splitting, cropping) can be added here
        shutil.move(temp_video_path, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error processing YouTube video: {str(e)}")
        raise


async def split_video_into_chunks(video_path: str, temp_dir: Path, duration: int = MAX_CLIP_DURATION) -> List[str]:
    """
    Split a video into chunks of specified duration.
    """
    try:
        loop = asyncio.get_event_loop()
        video = await loop.run_in_executor(None, lambda: VideoFileClip(video_path))
        chunk_paths = []
        total_duration = video.duration
        num_chunks = int(total_duration / duration) + (1 if total_duration % duration > 0 else 0)
        for i in range(num_chunks):
            start_time = i * duration
            end_time = min((i + 1) * duration, total_duration)
            chunk = video.subclip(start_time, end_time)
            chunk_path = str(temp_dir / f"chunk_{i+1}.mp4")
            await loop.run_in_executor(None, lambda: chunk.write_videofile(chunk_path, codec="libx264", audio_codec="aac"))
            chunk_paths.append(chunk_path)
            chunk.close()
        video.close()
        return chunk_paths
    except Exception as e:
        logger.error(f"Error splitting video: {str(e)}")
        raise RuntimeError(f"Failed to split video into chunks: {video_path}") from e


async def process_video_chunk(input_path: str, output_path: str) -> bool:
    """
    Process a video chunk with face detection and reframing.
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scale_factor = TARGET_HEIGHT / frame_height
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_video_path = temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
        face_centers = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % (total_frames // 100 or 1) == 0:
                logger.info(f"Processing frame {frame_count}/{total_frames} ({(frame_count / total_frames) * 100:.2f}%)")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                face_centers.append((face_center_x, face_center_y))
            elif face_centers:
                face_center_x, face_center_y = face_centers[-1]
            else:
                face_center_x = frame_width // 2
                face_center_y = frame_height // 2
                face_centers.append((face_center_x, face_center_y))
            crop_width = int(TARGET_HEIGHT * frame_width / frame_height)
            left = max(0, face_center_x - crop_width // 2)
            right = left + crop_width
            if right > frame_width:
                right = frame_width
                left = max(0, right - crop_width)
            cropped_frame = frame[:, left:right]
            resized_frame = cv2.resize(cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT))
            out.write(resized_frame)
        cap.release()
        out.release()
        loop = asyncio.get_event_loop()
        processed_video = await loop.run_in_executor(None, lambda: VideoFileClip(temp_video_path))
        original_video = await loop.run_in_executor(None, lambda: VideoFileClip(input_path))
        if original_video.audio is not None:
            processed_video = processed_video.set_audio(original_video.audio)
        await loop.run_in_executor(None, lambda: processed_video.write_videofile(output_path, codec="libx264", audio_codec="aac"))
        processed_video.close()
        original_video.close()
        os.remove(temp_video_path)
        return True
    except Exception as e:
        logger.error(f"Error processing video chunk: {str(e)}")
        return False


async def generate_hindi_audio(english_audio_path: str, output_path: str) -> str:
    """
    Generate Hindi audio from English audio (placeholder for future implementation).
    """
    shutil.copy(english_audio_path, output_path)
    return output_path
