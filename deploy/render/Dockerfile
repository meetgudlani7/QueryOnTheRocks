# Lean runtime image for Render Free (512MB) — no torch/transformers/
# duckdb/pandas (requirements-runtime.txt), quantized int8 ONNX embedder
# only. No BM25: the in-memory inverted-index pickle alone runs 1.5-2.5GB
# resident once unpickled, which cannot coexist with anything else in a
# 512MB container — dropped entirely for this deployment (dense-only
# retrieval; pipeline/orchestrator.py already treats BM25 failure as a
# non-fatal degradation, see asyncio.gather(..., return_exceptions=True)).
#
# EMBEDDING_BACKEND/EMBEDDING_ONNX_DIR/QDRANT_COLLECTION are baked in as
# defaults below (not left to be configured on the Render dashboard) since
# forgetting any of the three breaks this image in a different way: unset
# EMBEDDING_BACKEND falls back to the torch path (not installed here, hard
# ImportError); wrong QDRANT_COLLECTION points dense search at vectors
# embedded by a different-precision model (see
# scripts/quantize_onnx_embedder.py's docstring). Override on the Render
# dashboard only if you deliberately want a different collection.
FROM python:3.13-slim

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY config config
COPY retrieval retrieval
COPY pipeline pipeline
COPY apps/api apps/api
COPY data/onnx_embedder_quantized data/onnx_embedder_quantized

ENV EMBEDDING_BACKEND=onnx \
    EMBEDDING_ONNX_DIR=data/onnx_embedder_quantized \
    QDRANT_COLLECTION=msmarco_xi_onnx_q8 \
    OMP_NUM_THREADS=1

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
