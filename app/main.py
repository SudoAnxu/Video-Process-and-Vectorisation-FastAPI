import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for video upload and frames extraction
os.makedirs("frames", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Import models and services
from app.models import FrameInterval, FrameResponse, SearchQuery, SearchResult
from app.video_processor import VideoProcessor
from app.vector_store import VectorStore

# Global service instances
video_processor = VideoProcessor()
vector_store = VectorStore()

def create_application():
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Video Processing and Vector Search API",
        description="An API for video frame extraction and vector search"
    )

    # Mounting static files for serving frame images
    app.mount("/frames", StaticFiles(directory="frames"), name="frames")

    # Root path handler
    @app.get("/")
    async def root():
        return {
            "message": "Welcome to Video Processing API",
            "endpoints": {
                "process_video": "/videos/process",
                "search_vectors": "/vectors/search",
                "documentation": "/docs"
            }
        }

    # Video Processing Endpoint
    @app.post("/videos/process", response_model=List[FrameResponse])
    async def process_video(
        file: UploadFile = File(...), 
        interval: Optional[float] = 1.0
    ):
        """
        Upload and process a video file:
        1. Save the uploaded video
        2. Extract frames at the specified interval
        3. Compute feature vectors for each frame
        4. Store vectors in the database
        5. Return information about the processed frames
        """
        try:
            # Log file details
            logger.info(f"Received file: {file.filename}")
            logger.info(f"Content type: {file.content_type}")
            
            # Validating file type
            if not file.content_type.startswith("video/"):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.content_type}"
                )
            
            # Uploaded files saved 
            file_path = await video_processor.save_upload(file)
            
            # Extraction frames
            frames_info = video_processor.extract_frames(file_path, interval)
            
            # Computation and storing feature vectors
            vectors_data = []
            for frame in frames_info:

                vector = video_processor.compute_feature_vector(frame["frame_path"])
                logger.info(f"Vector for frame {frame['frame_id']}: {vector}")
                vectors_data.append({
                    "frame_id": frame["frame_id"],
                    "frame_path": frame["frame_path"],
                    "timestamp": frame["timestamp"],
                    "vector": vector
                })
            
            # Store vectors in database
            vector_store.add_vectors(vectors_data)
            
            # Returning frame information
            return [FrameResponse(**frame) for frame in frames_info]
            
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    # Vector Search Endpoint
    @app.post("/vectors/search", response_model=List[SearchResult])
    async def search_similar_frames(
        query: SearchQuery
    ):
        """
        Search for frames with similar feature vectors
        """
        try:
            results = vector_store.search_similar(query.vector, query.top_k)
            return [SearchResult(**result) for result in results]
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")

    return app

# Creating the fastapi application
app = create_application()
