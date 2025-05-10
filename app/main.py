import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from typing import List

from app.models import FrameInterval, FrameResponse, FeatureVector, SearchQuery, SearchResult
from app.video_processor import VideoProcessor
from app.vector_store import VectorStore

app = FastAPI(title="Video Processing and Vector Search API")

# Mount static files for serving frame images
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

# Initialize services
video_processor = VideoProcessor()
vector_store = VectorStore()

@app.post("/videos/process", response_model=List[FrameResponse])
async def process_video(file: UploadFile = File(...), params: FrameInterval = Depends()):
    """
    Upload and process a video file:
    1. Save the uploaded video
    2. Extract frames at the specified interval
    3. Compute feature vectors for each frame
    4. Store vectors in the database
    5. Return information about the processed frames
    """
    # Validate file type (basic check)
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded file
        file_path = await video_processor.save_upload(file)
        
        # Extract frames
        frames_info = video_processor.extract_frames(file_path, params.interval_seconds)
        
        # Compute and store feature vectors
        vectors_data = []
        for frame in frames_info:
            vector = video_processor.compute_feature_vector(frame["frame_path"])
            vectors_data.append({
                "frame_id": frame["frame_id"],
                "frame_path": frame["frame_path"],
                "timestamp": frame["timestamp"],
                "vector": vector
            })
        
        # Store vectors in database
        vector_store.add_vectors(vectors_data)
        
        # Return frame information
        return [FrameResponse(**frame) for frame in frames_info]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/vectors/search", response_model=List[SearchResult])
async def search_similar_frames(query: SearchQuery):
    """
    Search for frames with similar feature vectors
    """
    try:
        results = vector_store.search_similar(query.vector, query.top_k)
        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")