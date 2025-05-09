from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
import uuid
import shutil
from pathlib import Path
import asyncio
import logging
import uvicorn
from download_and_process import process_youtube_video

app = FastAPI(title="YouTube to Instagram Reels Converter")

# Setup templates and static dirs
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create necessary directories
TEMP_DIR = Path("temp")
DOWNLOADS_DIR = Path("downloads")
TEMP_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_video(request: Request, youtube_url: str = Form(...)):
    # Create unique job ID for this processing request
    job_id = str(uuid.uuid4())
    job_dir = DOWNLOADS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    try:
        logger.info(f"Processing YouTube URL: {youtube_url}")
        
        # Process the video asynchronously
        # This is a non-blocking call that will run in the background
        asyncio.create_task(
            process_video_task(youtube_url, job_id)
        )
        
        # Return a page that will show processing status and eventually results
        return templates.TemplateResponse(
            "processing.html", 
            {
                "request": request, 
                "job_id": job_id,
                "youtube_url": youtube_url
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request, 
                "error": str(e)
            }
        )

async def process_video_task(youtube_url: str, job_id: str):
    """Background task to process the video"""
    try:
        output_files = await process_youtube_video(youtube_url, job_id, TEMP_DIR, DOWNLOADS_DIR)
        # Update status file to indicate completion
        status_file = DOWNLOADS_DIR / job_id / "status.json"
        with open(status_file, "w") as f:
            import json
            json.dump({
                "status": "completed", 
                "output_files": [os.path.basename(f) for f in output_files]
            }, f)
    except Exception as e:
        logger.error(f"Background task error: {str(e)}")
        # Update status file to indicate error
        status_file = DOWNLOADS_DIR / job_id / "status.json"
        with open(status_file, "w") as f:
            import json
            json.dump({"status": "error", "message": str(e)}, f)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check the status of a processing job"""
    status_file = DOWNLOADS_DIR / job_id / "status.json"
    
    if not status_file.exists():
        return {"status": "processing"}
    
    with open(status_file, "r") as f:
        import json
        return json.load(f)

@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a processed video file"""
    file_path = DOWNLOADS_DIR / job_id / filename
    
    if not file_path.exists():
        return {"error": "File not found"}
    
    return FileResponse(file_path, filename=filename)

@app.get("/results/{job_id}", response_class=HTMLResponse)
async def view_results(request: Request, job_id: str):
    """View processing results"""
    return templates.TemplateResponse(
        "results.html", 
        {
            "request": request, 
            "job_id": job_id
        }
    )

# Cleanup old files periodically (optional, could be implemented with a background task)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)