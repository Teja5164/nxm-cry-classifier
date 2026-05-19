# Infant Cry Classifier — Docker Setup
# ======================================
# Run from project root (D:\nxm\ML_pipeline\):
#
#   Build:  docker build -f deployment/Dockerfile -t cry-classifier .
#   Run:    docker run -p 8000:8000 cry-classifier
#   GPU:    docker run -p 8000:8000 --gpus all cry-classifier
#
# Or use docker-compose (from deployment/ folder):
#   docker-compose up --build

FROM python:3.10-slim

# Install system deps for librosa/soundfile + curl for healthcheck
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install PyTorch CPU-only separately (uses PyTorch index, smaller wheel)
# Own layer so other packages are cached independently of torch
RUN pip install --no-cache-dir --timeout 600 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining API dependencies (smaller packages, faster download)
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    "librosa==0.11.0" "soundfile==0.13.1" numpy "scipy==1.15.3" \
    "scikit-learn==1.6.1" "fastapi==0.115.12" "uvicorn[standard]==0.34.0" \
    "python-multipart==0.0.20"

COPY configs/    ./configs/
COPY inference/  ./inference/
COPY api/        ./api/

# Models and norm_stats must be mounted at runtime:
#   -v /host/models:/app/models
#   -v /host/training/features:/app/training/features

EXPOSE 8000

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
