# Voice-Enabled RAG

HH Goa 2026 — Task 2 implementation scaffold for a multilingual, voice-enabled Retrieval-Augmented Generation (RAG) application.

The project is designed around this flow:

```text
Voice → Groq Whisper STT → query processing → hybrid retrieval → evidence gate → Groq LLM → answer
```

## Current status

This repository is an in-progress implementation. The API, pipeline modules, retrieval abstractions, ingestion flow, web UI, tests, and helper scripts are present, but the production integrations are not complete yet.

| Area | Current state |
| --- | --- |
| FastAPI backend | Runnable scaffold with `/health`, `/api/query`, and `/api/audio` routes |
| Next.js frontend | Basic text and voice-query interface |
| BM25 and RRF | Implemented locally |
| Groq STT/LLM | API clients are present; live calls require configuration and further hardening |
| Dataset ingestion | Scaffold currently uses mock/fallback data; real MSMARCO-XI ingestion is pending |
| Embeddings | Current implementation uses deterministic placeholder vectors; real multilingual embeddings are pending |
| Qdrant | Store abstraction is present; real indexed retrieval is pending |
| Guardrails | Basic evidence and response checks exist; structured-output validation is still being hardened |
| Performance | No benchmark results are claimed yet |

See [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) for the detailed replacement plan and known gaps. The original assignment brief is in [`HH_Goa_2026_Task_2_End_to_End_Implementation_Guide(1).md`](HH_Goa_2026_Task_2_End_to_End_Implementation_Guide(1).md).

## Intended architecture

The system has two pipelines:

### Offline indexing

```text
MSMARCO-XI → normalize → chunk → embed → Qdrant index + BM25 index
```

### Online query processing

```text
text/audio → transcript → dense search + BM25 search → RRF → evidence gate → generation → validation
```

Dense and keyword retrieval are intended to run concurrently. Chunking, document embeddings, and index construction should happen offline so they are not part of request latency.

## Repository layout

```text
apps/api/          FastAPI application and HTTP routes
apps/web/          Next.js frontend
config/            Environment-backed application settings and logging
evaluation/        Benchmark runner and evaluation metrics
ingestion/         Dataset loading, normalization, chunking, embeddings, indexing
pipeline/          STT, retrieval, generation, guardrails, orchestration, metrics
retrieval/         Qdrant store, BM25 store, and rank fusion
scripts/           Index build, startup check, and smoke test utilities
tests/             Unit and pipeline tests
```

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer for the web application
- A Groq API key for live speech-to-text and generation calls
- Qdrant for the intended production retrieval path

The base Python dependencies are listed in [`requirements.txt`](requirements.txt). Dataset, embedding, and Qdrant packages are currently commented as optional because those integrations are still being completed.

## Local setup

The following steps are enough to run the repository locally. The backend can be started without a live Groq request, but real voice transcription and LLM answers require a Groq API key. The complete production retrieval path also requires the optional dataset, embedding, and Qdrant dependencies described below.

### 1. Create the Python environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create local configuration

```bash
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` if you want to exercise live Groq calls. The remaining values have local-development defaults. Do not commit `.env` or API keys.

### 3. Start Qdrant locally (optional for the current scaffold)

If Docker is installed, run Qdrant in a separate terminal:

```bash
docker run --name voice-rag-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant:latest
```

The default `.env` already points to `http://localhost:6333`. Stop and remove this container later with:

```bash
docker stop voice-rag-qdrant
docker rm voice-rag-qdrant
```

The current readiness route does not yet perform complete Qdrant validation, so a healthy container alone does not mean the production index has been built.

### 4. Install and prepare the frontend

In a second terminal, from the repository root:

```bash
cd apps/web
npm install
```

### 5. Build the optional real retrieval stack

The repository currently keeps these larger integrations commented out in `requirements.txt`. When working on the real MSMARCO-XI pipeline, install the required packages explicitly:

```bash
pip install datasets huggingface-hub qdrant-client sentence-transformers torch
```

Then configure `DATASET_NAME`, `DATASET_SPLIT`, `QDRANT_URL`, and the embedding settings in `.env` and run:

```bash
python scripts/build_index.py
```

This step is not required for the basic API or unit-test setup, and the roadmap should be consulted before treating the generated index as production-ready.

## Run the applications

### Backend

From the repository root, with the virtual environment activated:

```bash
uvicorn apps.api.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`. Interactive API documentation is available at `/docs`.

### Frontend

In a second terminal:

```bash
cd apps/web
npm run dev
```

The web application is available at `http://localhost:3000`.

The current frontend calls `/api/query` and `/api/audio` as same-origin paths. In a local split frontend/backend setup, configure a Next.js rewrite/proxy to forward those paths to `http://localhost:8000`, or use the backend directly with the `curl` examples below. Without that proxy, the browser sends requests to port 3000 rather than port 8000.

### Verify the local backend

```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

For a configured local environment, run the basic checks from the repository root:

```bash
python scripts/startup_check.py
python scripts/smoke_test.py
```

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic API metadata |
| `GET` | `/health` | Liveness response |
| `GET` | `/health/ready` | Readiness response; service probes are not fully implemented yet |
| `POST` | `/api/query` | Process a JSON text query |
| `POST` | `/api/audio` | Upload an audio file for transcription and processing |

Example text request:

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who discovered penicillin?","language":"en"}'
```

The audio endpoint expects a multipart upload with an audio file:

```bash
curl -X POST http://localhost:8000/api/audio \
  -F 'file=@recording.wav;type=audio/wav'
```

## Indexing status

The intended indexing command is:

```bash
python scripts/build_index.py
```

At the current stage, do not treat this as a production dataset build: the roadmap identifies dataset loading, real embeddings, Qdrant wiring, and BM25 persistence as unfinished work. Run it only after installing the optional integration dependencies and verifying the local configuration.

## Tests and checks

Run the test suite with:

```bash
pytest
```

Useful development checks include:

```bash
python scripts/startup_check.py
python scripts/smoke_test.py
```

These scripts currently provide basic checks, not a complete 10/10 production readiness gate. Live service checks require the relevant services and credentials.

## Configuration

Common settings are documented in [`.env.example`](.env.example), including:

- Groq STT and LLM models
- Qdrant URL, API key, and collection name
- MSMARCO-XI dataset name, split, cache directory, and optional limit
- Retrieval sizes, fusion settings, context limits, and the latency target
- Multilingual embedding model and vector dimension

The `TARGET_LATENCY_MS=200` value is an architectural target from the assignment, not a measured result. Any performance claim should be based on a benchmark run and should report STT and RAG latency separately.

## Roadmap

The highest-priority remaining work is:

1. Replace mock dataset loading and placeholder embeddings with real MSMARCO-XI and multilingual embedding integrations.
2. Wire the same real embedding model into both Qdrant indexing and query-time search.
3. Add structured JSON generation and robust evidence-ID validation.
4. Harden timeouts, retries, request IDs, readiness checks, and persisted indexes.
5. Expand multilingual and refusal evaluation coverage, then benchmark the complete system.

## License

No license has been added to this repository yet. Add a license file before distributing the project publicly.
