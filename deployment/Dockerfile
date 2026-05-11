# Infant Cry Classifier — Docker Setup
# ======================================
# Build:  docker build -t cry-classifier .
# Run:    docker run -p 8000:8000 --gpus all cry-classifier
# CPU:    docker run -p 8000:8000 cry-classifier

FROM python:3.10-slim

# Install system deps for librosa/soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only code (not datasets, models, features — mount as volumes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
