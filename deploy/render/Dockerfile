# Lean runtime image for Render — no torch/transformers/duckdb/pandas
# (requirements-runtime.txt), ONNX embedder + BM25 pickle baked in.
# Requires EMBEDDING_BACKEND=onnx set as an env var on the Render service
# (retrieval/embeddings.py falls back to the torch path — not installed
# here — if that's left unset, so this MUST be set for this image to boot).
FROM python:3.13-slim

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY config config
COPY retrieval retrieval
COPY pipeline pipeline
COPY apps/api apps/api
COPY data/bm25_index.pkl data/bm25_index.pkl
COPY data/onnx_embedder data/onnx_embedder

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
