FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY videorag ./videorag
ENV VIDEORAG_DATA_DIR=/data HF_HOME=/data/hf-cache
VOLUME ["/data"]
EXPOSE 8000
CMD ["python", "-m", "videorag", "serve", "--host", "0.0.0.0", "--port", "8000"]
