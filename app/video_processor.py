import os
import cv2
import uuid
import numpy as np
from datetime import datetime
from typing import List, Tuple

class VideoProcessor:
    def __init__(self, upload_dir: str = "uploads", frames_dir: str = "frames"):
        self.upload_dir = upload_dir
        self.frames_dir = frames_dir
        
        # Create directories if they don't exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        
    async def save_upload(self, file) -> str:
        """Save uploaded file and return the path"""
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        file_path = os.path.join(self.upload_dir, filename)
        
        # Write file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        return file_path
    
    def extract_frames(self, video_path: str, interval_seconds: float = 1.0) -> List[dict]:
        """Extract frames from video at specified intervals"""
        frames_info = []
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            raise ValueError("Could not open video file")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        
        current_frame = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                timestamp = current_frame / fps
                frame_id = f"{os.path.basename(video_path)}_{current_frame}"
                frame_path = os.path.join(self.frames_dir, f"{frame_id}.jpg")
                
                # Save frame as image
                cv2.imwrite(frame_path, frame)
                
                frames_info.append({
                    "frame_id": frame_id,
                    "frame_path": frame_path,
                    "timestamp": timestamp
                })
                
            current_frame += 1
            
        video.release()
        return frames_info
    
    def compute_feature_vector(self, frame_path: str) -> List[float]:
        """Compute feature vector for a given frame image"""
        # Read image
        img = cv2.imread(frame_path)
        
        # Convert to HSV for better color representation
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Compute color histogram
        h_hist = cv2.calcHist([hsv_img], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
        
        # Combine histograms to form feature vector
        feature_vector = np.concatenate([h_hist, s_hist, v_hist]).tolist()
        
        return feature_vector