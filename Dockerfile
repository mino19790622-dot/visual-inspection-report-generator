FROM python:3.12-slim

WORKDIR /app

# System libs required by OpenCV (libGL) on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching (code changes won't re-install)
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Application code + model + standards knowledge base
COPY app/ ./app/
COPY data/ ./data/
COPY classes.txt yolov8m.onnx run_pipeline.py run_agent.py ./

EXPOSE 8000

CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
