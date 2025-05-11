from pydantic import BaseModel
from typing import List, Optional

class FrameInterval(BaseModel):
    interval_seconds: float = 1.0

class FrameResponse(BaseModel):
    frame_id: str
    frame_path: str
    timestamp: float

class SearchQuery(BaseModel):
    vector: List[float]
    top_k: int = 5


class SearchResult(BaseModel):
    frame_id: str
    frame_path: str
    vector: List[float]
    score: Optional[float] = None
