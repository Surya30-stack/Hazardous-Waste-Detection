FROM python:3.13-slim

WORKDIR /app

# System libraries OpenCV needs on a slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1) CPU-only PyTorch first (skips ~2.5 GB of CUDA packages)
RUN pip install --no-cache-dir --default-timeout=100 --retries 10 \
    torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries 10 \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]