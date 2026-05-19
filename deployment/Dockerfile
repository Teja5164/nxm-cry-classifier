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

# Copy only the API requirements (no training packages like transformers/gdown)
COPY requirements-api.txt .
# Extended timeout for large packages (torch CPU wheel is ~250MB)
RUN pip install --no-cache-dir --timeout 600 --retries 5 -r requirements-api.txt

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
