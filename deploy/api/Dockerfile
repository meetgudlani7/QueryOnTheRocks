# Standalone API deployment image — CPU-only torch wheel (avoids the
# ~2-5GB CUDA wheel; there's no GPU to use it on a free-tier box anyway),
# and bakes data/bm25_index.pkl into the image so there's no separate
# data-upload step at deploy time.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY config config
COPY retrieval retrieval
COPY pipeline pipeline
COPY apps/api apps/api
COPY data/bm25_index.pkl data/bm25_index.pkl

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
