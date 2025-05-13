# Vlog2Reels - YouTube to Instagram Reels Converter

This web application automatically transforms YouTube vlog videos into Instagram Reels by:
1. Downloading YouTube videos
2. Splitting them into 60-second clips
3. Converting them to vertical (9:16) format with AI face tracking
4. (Optionally) Replacing English audio with Hindi audio (future enhancement)

## 🎯 Features

- **YouTube Video Download**: Easily download any public YouTube video using the YouTube Data API
- **AI Face Detection & Tracking**: Intelligently reframes videos to keep faces centered
- **Vertical Format Conversion**: Transforms landscape videos to 9:16 aspect ratio for Instagram Reels
- **Smart Clip Splitting**: Automatically splits longer videos into Reels-ready 60-second clips
- **Hindi Audio Support**: (Future enhancement) Convert English audio to Hindi

## 🔧 Tech Stack

- **Frontend**: HTML/CSS with Bootstrap 5 and Jinja2 templates
- **Backend**: Python + FastAPI
- **Video Processing**:
  - YouTube Data API for video metadata and streaming URLs
  - `ffmpeg` for video downloading and processing
  - `moviepy` for video editing and audio synchronization
  - `OpenCV` (cv2) for face detection and tracking
- **Deployment**: Render.com (containerized app)

## 📋 Setup and Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vlog2reels.git
   cd vlog2reels
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the development server**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

### Deployment to Render

1. **Push your code to GitHub**

2. **Create a new Web Service on Render**
   - Connect your GitHub repository
   - Select "Python" as the runtime environment
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add the `YOUTUBE_API_KEY` environment variable in the Render dashboard or `render.yaml`

3. **Deploy**
   Render will automatically build and deploy your application.

## 📁 Project Structure

```
vlog2reels/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── download_and_process.py # Video processing logic
│   ├── templates/              # HTML templates
│   │   ├── index.html          # Homepage with form
│   │   ├── processing.html     # Processing status page
│   │   ├── results.html        # Results page with downloadable videos
│   │   └── error.html          # Error page
│   ├── static/                 # Static assets (CSS, JS)
│   │   ├── css/
│   │   │   └── styles.css
│   │   ├── js/
│   │   │   └── scripts.js
│   │   └── images/
│   │       └── logo.png
├── temp/                       # Temporary storage for processing (created at runtime)
├── downloads/                  # Output directory for processed videos (created at runtime)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── render.yaml                 # Render deployment configuration
└── README.md                   # Project documentation
```

## 🧠 How It Works

1. **User Submits a YouTube URL**
   - The application validates the URL
   - Assigns a unique job ID for processing

2. **Backend Processing**
   - Fetches video metadata and streaming URLs using the YouTube Data API
   - Downloads the video using `ffmpeg`
   - Splits the video into 60-second chunks
   - For each chunk:
     - Detects and tracks faces using OpenCV
     - Reframes the video to vertical format (1080x1920)
     - Keeps faces centered in the frame

3. **Result Delivery**
   - Processed videos are saved to the downloads folder
   - User can view and download the final Reels-ready videos

## 📋 Future Enhancements

- Implement Hindi audio conversion using a Text-to-Speech service
- Add user authentication and video history
- Implement smart clip selection based on content analysis
- Add caption generation and subtitles
- Social media direct sharing functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenCV community for face detection algorithms
- FastAPI for the efficient web framework
- YouTube Data API for video metadata and streaming URLs
- FFmpeg for video processing
