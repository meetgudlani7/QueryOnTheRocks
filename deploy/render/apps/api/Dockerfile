# Build context is the repo root (see docker-compose.yml) — the API's
# imports (apps.api, pipeline, retrieval, config) are repo-root-relative,
# not a self-contained apps/api/ package.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch wheel: the default PyPI torch pulls CUDA libraries that are
# dead weight on a CPU-only container and roughly 5x the image size.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY config config
COPY retrieval retrieval
COPY pipeline pipeline
COPY apps/api apps/api

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
