import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VectorStore:
    def __init__(self, collection_name: str = "video_frames"):
        self.collection_name = collection_name
        # Initialize Qdrant client (local instance)
        self.client = QdrantClient(":memory:")  
        self.create_collection()
    
    def create_collection(self, vector_size: int = 48):
        """Create vector collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
    
    def add_vectors(self, vectors: List[Dict[str, Any]]):
        """Add vectors to the collection"""
        points = []
        
        for i, item in enumerate(vectors):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=item["vector"],
                    payload={
                        "frame_id": item["frame_id"],
                        "frame_path": item["frame_path"],
                        "timestamp": item.get("timestamp", 0.0)
                    }
                )
            )
        
        # Inserting vectors in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search_similar(self, vector: List[float], top_k: int = 5):
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            with_vectors=True  
        )
        results = []
        for hit in search_result:
            results.append({
            "frame_id": hit.payload["frame_id"],
            "frame_path": hit.payload["frame_path"],
            "vector": hit.vector,
            "score": hit.score
            })
        return results
