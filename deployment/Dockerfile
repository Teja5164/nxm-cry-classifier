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

# Step 1: Install PyTorch CPU-only (uses smaller CPU-only wheel ~175MB vs 532MB from PyPI)
# Shell retry loop handles hash failures caused by corrupted partial downloads on slow networks
RUN for i in 1 2 3 4 5; do \
      pip install --no-cache-dir --timeout 600 \
        torch --index-url https://download.pytorch.org/whl/cpu \
      && break || (echo "torch install attempt $i failed, retrying..."; sleep 10); \
    done

# Step 2: Install remaining API dependencies with retry loop
# (pip --retries only handles connection errors, not hash failures from corrupted downloads;
#  this shell loop retries the entire install on any failure)
RUN pip_install() { \
      for i in 1 2 3 4 5; do \
        pip install --no-cache-dir --timeout 600 "$@" && return 0; \
        echo "pip attempt $i failed, retrying in 10s..."; sleep 10; \
      done; return 1; \
    }; \
    pip_install "numpy==2.2.6" && \
    pip_install "scipy==1.15.3" && \
    pip_install "scikit-learn==1.6.1" && \
    pip_install "soundfile==0.13.1" && \
    pip_install "librosa==0.11.0" && \
    pip_install "fastapi==0.115.12" "uvicorn[standard]==0.34.0" "python-multipart==0.0.20"

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
