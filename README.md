
---
# 🎬 Video Processing & Vector Search API

A FastAPI-based backend for video frame extraction, feature vectorization using color histograms, and similarity search via a Qdrant vector database. Designed for efficient video data indexing and retrieval.

## 🚀 Features

- **Video Upload & Processing**: Upload videos, extract frames, and compute feature vectors.
- **Vector Storage**: Store and manage extracted feature vectors in Qdrant.
- **Frame Similarity Search**: Query for similar frames using vector similarity.
- **Robust APIs**: Easy-to-use, well-documented endpoints.

## 🏗️ Architecture Overview

```plaintext
[User] <--HTTP--> [FastAPI Backend] <---> [Qdrant Vector DB]
                                     |
                                [OpenCV & Numpy]
```

1. User uploads a video file.
2. FastAPI server processes the video with OpenCV, extracting frames and computing color histogram vectors.
3. Computed vectors are stored in Qdrant.
4. The user can search for similar frames by submitting a vector to the backend, which queries Qdrant and returns the top matches.

## ⚙️ Prerequisites

- **Python** >= 3.8
- **Qdrant Server** (Docker/local instance or remote URL)
- (Optional) [Docker installed](https://docs.docker.com/get-docker/) if you prefer using containers

## 📦 Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Spin up Qdrant with Docker**
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

4. **Update Qdrant connection info in `main.py` if needed**

## 🚦 Usage

- **Start the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```

- **Open API docs:**  
  Go to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

### Example Requests

#### 1. Upload and Process Video

```bash
curl -F "file=@/path/to/video.mp4" http://localhost:8000/videos/process
```

#### 2. Search for Similar Frames

Replace `[vector_values]` with the vector you want to search with:

```bash
curl -X POST "http://localhost:8000/vectors/search" \
     -H "Content-Type: application/json" \
     -d '{"vector": [0.125, 0.675, 0.430, ...], "top_k": 5}'
```

## 📝 API Endpoints

- `POST /videos/process`  
  Upload a video file. Extracts frames, computes vectors, and stores them.

  **Parameters:**  
  - `file` (form-data): Video file to upload

- `POST /vectors/search`  
  Find similar frames based on a provided feature vector.

  **Parameters:**  
  - `vector` (list of float): Feature vector for similarity search
  - `top_k` (int, optional): Number of similar results to retrieve

## 📂 Project Structure

```plaintext
├── app/
│   ├── main.py             # FastAPI app with endpoints
│   ├── models.py           # Pydantic models & data schemas
│   ├── video_processor.py  # Frame extraction & feature computation
│   └── vector_store.py     # Qdrant integration & vector search
├── requirements.txt        # Python dependencies
├── run.py                  # Entry point to run the API server
└── README.md               # Project documentation
```

## 💡 Notes

- Make sure Qdrant server is running before starting the FastAPI app.
- API returns JSON responses, including error information if applicable.
- The feature extraction is based on color histograms; feel free to upgrade with more complex models.

## 👤 Author

- [Your Name or GitHub handle](https://github.com/SudoAnxu)

---

Enjoy using the Video Processing & Vector Search API!🎉
```
