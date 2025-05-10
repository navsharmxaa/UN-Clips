import asyncio
import logging
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tempfile

# Video processing
import yt_dlp
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TARGET_HEIGHT = 1920  # 9:16 aspect ratio for Instagram Reels
TARGET_WIDTH = 1080
MAX_CLIP_DURATION = 60  # seconds

import yt_dlp

async def process_youtube_video(youtube_url: str, job_id: str, temp_dir: Path, downloads_dir: Path):
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        url: YouTube URL
        temp_dir: Directory to save the downloaded video
        
    Returns:
        Path to the downloaded video file
    """
    try:
        loop = asyncio.get_event_loop()
        
        # Define yt-dlp options for downloading the video
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',  # Select the best video and audio combination
            'outtmpl': str(temp_dir / 'original.mp4'),  # Path to save the video
            'quiet': False,  # Show output for debugging
            'retries': 3,  # Retry logic (number of retries)
            'noplaylist': True,  # Don't download playlist (if URL is playlist)
        }
        
        # Use yt-dlp to download the video asynchronously
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            await loop.run_in_executor(None, lambda: ydl.download([url]))
        
        # Return the path of the downloaded video
        return str(temp_dir / 'original.mp4')
    
    except Exception as e:
        logger.error(f"Error downloading YouTube video with yt-dlp: {str(e)}")
        raise
    
    finally:
        # Clean up temporary files
        if job_temp_dir.exists():
            shutil.rmtree(job_temp_dir)

async def download_youtube_video(url: str, temp_dir: Path) -> str:
    """
    Download a YouTube video using pytube.
    
    Args:
        url: YouTube URL
        temp_dir: Directory to save the downloaded video
        
    Returns:
        Path to the downloaded video file
    """
    try:
        loop = asyncio.get_event_loop()
        yt = await loop.run_in_executor(None, lambda: YouTube(url))
        
        # Get the highest resolution stream with both video and audio
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        
        if not stream:
            # If no progressive stream found, get the highest resolution video stream
            stream = yt.streams.filter(file_extension="mp4").order_by("resolution").desc().first()
        
        if not stream:
            raise ValueError("No suitable video stream found")
        
        # Download the video
        output_path = await loop.run_in_executor(
            None, 
            lambda: stream.download(output_path=str(temp_dir), filename=f"original.mp4")
        )
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {str(e)}")
        raise

async def split_video_into_chunks(video_path: str, temp_dir: Path, duration: int = MAX_CLIP_DURATION) -> List[str]:
    """
    Split a video into chunks of specified duration.
    
    Args:
        video_path: Path to the video file
        temp_dir: Directory to save the video chunks
        duration: Maximum duration of each chunk in seconds
        
    Returns:
        List of paths to the video chunks
    """
    try:
        # Load the video using moviepy
        loop = asyncio.get_event_loop()
        video = await loop.run_in_executor(None, lambda: VideoFileClip(video_path))
        
        chunk_paths = []
        total_duration = video.duration
        
        # Calculate how many chunks we need
        num_chunks = int(total_duration / duration) + (1 if total_duration % duration > 0 else 0)
        
        for i in range(num_chunks):
            start_time = i * duration
            end_time = min((i + 1) * duration, total_duration)
            
            # Extract the subclip
            chunk = video.subclip(start_time, end_time)
            
            # Save the chunk
            chunk_path = str(temp_dir / f"chunk_{i+1}.mp4")
            await loop.run_in_executor(None, lambda: chunk.write_videofile(chunk_path, codec="libx264", audio_codec="aac"))
            
            chunk_paths.append(chunk_path)
            
            # Close the chunk to free memory
            chunk.close()
        
        # Close the original video
        video.close()
        
        return chunk_paths
    
    except Exception as e:
        logger.error(f"Error splitting video: {str(e)}")
        raise

async def process_video_chunk(input_path: str, output_path: str) -> bool:
    """
    Process a video chunk with face detection and reframing.
    
    Args:
        input_path: Path to the input video chunk
        output_path: Path to save the processed video
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Load the pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Open the video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate scaling factor to maintain aspect ratio
        scale_factor = TARGET_HEIGHT / frame_height
        
        # Create a temporary file for the processed frames
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_video_path = temp_file.name
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Process each frame
        face_centers = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 == 0:  # Log progress every 10 frames
                logger.info(f"Processing frame {frame_count}/{total_frames}")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Find the center position for cropping
            if len(faces) > 0:
                # Use the largest face if multiple are detected
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Calculate face center
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                face_centers.append((face_center_x, face_center_y))
            elif face_centers:
                # If no face detected in this frame but we've seen faces before,
                # use the last known face position
                face_center_x, face_center_y = face_centers[-1]
            else:
                # If no face has been detected at all, use the center of the frame
                face_center_x = frame_width // 2
                face_center_y = frame_height // 2
                face_centers.append((face_center_x, face_center_y))
            
            # Calculate cropping coordinates
            # The goal is to create a 9:16 vertical crop centered on the face
            crop_width = int(TARGET_HEIGHT * frame_width / frame_height)
            
            # Ensure the crop stays within the frame
            left = max(0, face_center_x - crop_width // 2)
            right = left + crop_width
            
            if right > frame_width:
                right = frame_width
                left = max(0, right - crop_width)
            
            # Crop and resize the frame
            cropped_frame = frame[:, left:right]
            resized_frame = cv2.resize(cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT))
            
            # Write the frame
            out.write(resized_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Now we need to add the audio back from the original clip
        # We use moviepy for this since it handles audio well
        loop = asyncio.get_event_loop()
        
        # Load the processed video (without audio) and the original video (with audio)
        processed_video = await loop.run_in_executor(None, lambda: VideoFileClip(temp_video_path))
        original_video = await loop.run_in_executor(None, lambda: VideoFileClip(input_path))
        
        # Add the audio from the original video to the processed video
        if original_video.audio is not None:
            processed_video = processed_video.set_audio(original_video.audio)
        
        # Write the final video with audio
        await loop.run_in_executor(None, lambda: processed_video.write_videofile(output_path, codec="libx264", audio_codec="aac"))
        
        # Close the clips
        processed_video.close()
        original_video.close()
        
        # Remove the temporary file
        os.remove(temp_video_path)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing video chunk: {str(e)}")
        return False


# Additional function for future implementation of Hindi audio generation
async def generate_hindi_audio(english_audio_path: str, output_path: str) -> str:
    """
    Generate Hindi audio from English audio (placeholder for future implementation).
    
    In a production system, this would use a speech-to-speech translation service
    or a combination of speech-to-text + translation + text-to-speech.
    
    Args:
        english_audio_path: Path to the English audio file
        output_path: Path to save the Hindi audio file
        
    Returns:
        Path to the Hindi audio file
    """
    # This is a placeholder - in a real implementation, you'd use a
    # speech-to-speech translation service or a pipeline:
    # 1. Speech-to-text (convert English audio to English text)
    # 2. Translation (translate English text to Hindi text)
    # 3. Text-to-speech (convert Hindi text to Hindi audio)
    
    # For now, we'll just copy the original audio as a placeholder
    shutil.copy(english_audio_path, output_path)
    return output_path
