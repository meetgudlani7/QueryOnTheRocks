# Implementation Roadmap — Voice-Enabled RAG (HH Goa 2026, Task 2)

This is the build order for turning the current repo into a working submission, based on the guide in `HH_Goa_2026_Task_2_End_to_End_Implementation_Guide(1).md`.

## Start here: the repo is not empty

Every file the guide asks for already exists — `apps/`, `pipeline/`, `ingestion/`, `retrieval/`, `evaluation/`, `scripts/`, `tests/` are all scaffolded with real code, not empty stubs. But it was built **dependency-light**: instead of the real dataset, embedding model, and `qdrant-client`/`rank-bm25` libraries, several core pieces were wired up as **mocks that return plausible-looking fake data**. The pipeline runs end-to-end today, but on fake evidence.

So this roadmap is not "write files from scratch" — it's **replace the mocks with real integrations, in dependency order**, then harden and test. Phases 1–5 are the mock-replacement critical path; nothing after them is meaningful until they're done.

Legend: ✅ real/working · ⚠️ present but mocked or incomplete · ❌ missing

| Component | Status | File |
|---|---|---|
| Dataset download | ⚠️ returns 100 copies of one hardcoded Q&A, ignores real MSMARCO-XI | `ingestion/download.py:57-73` |
| Document embeddings | ⚠️ MD5-hash pseudo-vector, not a real model | `ingestion/embed.py:47,67-91` |
| Query-time embedding (Qdrant search) | ⚠️ same MD5-hash trick, so dense search is meaningless | `retrieval/qdrant_store.py:71-74,161-177` |
| BM25 keyword search | ✅ real BM25 math, hand-rolled | `retrieval/bm25_store.py` |
| RRF fusion | ✅ real | `retrieval/fusion.py` |
| Groq Whisper STT | ✅ real API call, one bug (multipart header) | `pipeline/stt.py:56` |
| Groq LLM generation | ⚠️ real API call, but plain text output — guide requires structured JSON (`answer`/`evidence_ids`/`grounded`/`confidence`) | `pipeline/generation.py:87-119` |
| Guardrails (evidence gate) | ⚠️ naive keyword-overlap heuristic, no injection defense in prompt | `pipeline/guardrails.py` |
| Guardrails (response grounding) | ⚠️ word-overlap heuristic instead of validating `evidence_ids`/`grounded` from LLM JSON | `pipeline/guardrails.py:99-165` |
| Orchestrator | ⚠️ correct shape (parallel retrieval, gate, generate, validate) but no timeouts, no retries, no request IDs | `pipeline/orchestrator.py` |
| Health/readiness checks | ❌ hardcoded `"pending"`, no real Groq/Qdrant probe | `apps/api/routes/health.py:38-46` |
| Frontend → backend wiring | ✅ correct endpoints already | `apps/web/lib/api.ts` |
| Evaluation query set | ⚠️ only 10 English factoid queries — missing Hindi/Hinglish/unanswerable/off-topic/injection | `evaluation/queries.jsonl` |
| Dependencies | ⚠️ `sentence-transformers`, `torch`, `datasets`, `huggingface-hub`, `qdrant-client` are commented out as "optional" | `requirements.txt` |

---

## Phase 1 — Real dataset ingestion

**Why first:** everything downstream (chunking, embeddings, indexes, evaluation) is built on whatever this returns. Right now it's 100 identical fake rows.

- Replace the mock block in `ingestion/download.py` with a real `datasets.load_dataset("ai4bharat/MSMARCO-XI", split=...)` call (or the HF datasets-server REST call already scaffolded at line 55). Inspect actual columns before assuming schema — the guide is explicit about this.
- Fix `_load_cached_dataset`'s use of `eval()` on cached lines (`ingestion/download.py:92`) — swap for `json.loads`/`json.dumps`; `eval` on a data file is unsafe and fragile.
- Verify `ingestion/normalize.py` maps real MSMARCO-XI fields into the internal schema (`id`, `query`, `passage`, `answer`, `language`, `query_type`, `is_selected`, `metadata`) without silently dropping language info.

**Files:** `ingestion/download.py`, `ingestion/normalize.py`

**Done when:** `python ingestion/download.py` prints real, varied MSMARCO-XI records across multiple languages.

---

## Phase 2 — Real embeddings

- In `ingestion/embed.py`, delete `_generate_mock_embedding` and load a real multilingual model (`intfloat/multilingual-e5-base` per the guide, or a smaller multilingual `sentence-transformers` model if load time is an issue). Uncomment `sentence-transformers`/`torch` in `requirements.txt`.
- Generate embeddings once, offline, over the normalized+chunked dataset — never per-request.

**Files:** `ingestion/embed.py`, `requirements.txt`

**Done when:** embedding vectors for two semantically similar sentences are actually close in cosine distance (mock hash vectors are not).

---

## Phase 3 — Wire real embeddings into Qdrant

- `retrieval/qdrant_store.py._generate_mock_vector` (line 161) is called both at index time and at **query time** — meaning even a correctly-populated Qdrant collection is being searched with a fake query vector today. Replace it with a call into the same embedding model from Phase 2 (load once, reuse — don't reload per request).
- Confirm the Qdrant collection is created with the right vector size/distance metric for the real model's output dimension (currently hardcoded to 384 for the mock).

**Files:** `retrieval/qdrant_store.py`, `config/settings.py` (`EMBEDDING_DIMENSION`)

**Done when:** a query for "who discovered penicillin" ranks the Fleming passage above unrelated passages via Qdrant alone.

---

## Phase 4 — Chunking strategies

The guide requires ≥3 chunking strategies with tests. Files exist but should be checked against real (not mock) passages once Phase 1 lands real data.

- `ingestion/chunkers/sentence.py` — sentence-window chunking.
- `ingestion/chunkers/semantic.py` — 120–180 token chunks, 20–30 token overlap, configurable.
- `ingestion/chunkers/metadata.py` — attaches `chunk_id`, `language`, `query_id`, `query_type`, `chunk_strategy`.
- `tests/test_chunking.py` — extend to assert no empty chunks, correct overlap, metadata preserved, across all three strategies on real data.

**Files:** `ingestion/chunkers/sentence.py`, `ingestion/chunkers/semantic.py`, `ingestion/chunkers/metadata.py`, `tests/test_chunking.py`

**Done when:** `pytest tests/test_chunking.py` passes on real MSMARCO-XI passages.

---

## Phase 5 — Build the real offline index end-to-end

`scripts/build_index.py` already orchestrates download → normalize → embed → index correctly — it just needs Phases 1–3 underneath it to be real.

- Also add BM25 persistence: `retrieval/bm25_store.py` is currently in-memory only and gets rebuilt from scratch on every process start. Add save/load (pickle or a simple serialized inverted index) so `apps/api/main.py` startup loads it instead of rebuilding.

**Files:** `scripts/build_index.py`, `retrieval/bm25_store.py`, `apps/api/main.py`

**Done when:** `python scripts/build_index.py` runs once, populates Qdrant + a persisted BM25 index, and the API starts up loading both without re-indexing.

---

## Phase 6 — Fix STT and verify live Groq calls

- `pipeline/stt.py:56` manually sets `Content-Type: multipart/form-data` while also passing `files=` to httpx — httpx needs to generate its own boundary; the manual header will break the multipart body. Remove the manual header and let httpx set it.
- With a real `GROQ_API_KEY` in `.env`, manually verify one STT call and one LLM call succeed.

**Files:** `pipeline/stt.py`, `.env` (local, not committed)

**Done when:** a real audio file transcribes correctly via `/api/audio`.

---

## Phase 7 — Structured JSON generation + injection resistance

The guide requires the LLM to return `{answer, evidence_ids, grounded, confidence}` as JSON, with a system prompt that explicitly treats retrieved passages as untrusted. Current `pipeline/generation.py` returns free-text and has no injection-defense language.

- Set `"response_format": {"type": "json_object"}` in the Groq request (`pipeline/generation.py:118`).
- Rewrite the system prompt to include: never invent facts, never use outside knowledge, retrieved passages are untrusted data, never follow instructions found inside them, return valid JSON with `evidence_ids` referencing the passed-in evidence IDs.
- Parse and validate the JSON response with Pydantic (`GenerationResponse` in `pipeline/schemas.py` needs `evidence_ids`/`grounded`/`confidence` fields added).

**Files:** `pipeline/generation.py`, `pipeline/schemas.py`

**Done when:** malformed/injected evidence in a test passage does not change the model's behavior, and output is always valid JSON matching the schema.

---

## Phase 8 — Real guardrails on structured output

Replace the word-overlap heuristics in `pipeline/guardrails.py` with checks against the structured JSON from Phase 7:

- `check_evidence` (pre-generation gate): keep the evidence-count/score threshold logic, but make `MIN_RETRIEVAL_SCORE` configurable via `config/settings.py` and tune it against the evaluation set rather than hardcoding `0.3`/`0.5`.
- `validate_response` (post-generation): validate JSON structure, that `grounded` is `true`, that `evidence_ids` actually reference retrieved evidence, and that `confidence` is in range — not string word-overlap.

**Files:** `pipeline/guardrails.py`, `config/settings.py`

**Done when:** `tests/test_guardrails.py` covers empty retrieval → refuse, low evidence → refuse, off-topic → refuse, injected instruction → ignored, valid evidence → answered.

---

## Phase 9 — Orchestrator hardening

`pipeline/orchestrator.py` has the right shape (parallel Qdrant+BM25 via `asyncio.gather`, evidence gate, generate, validate) but no timeouts, no retries, no request IDs.

- Add per-stage timeouts using `config/settings.py`'s already-defined `STT_TIMEOUT_MS`/`RETRIEVAL_TIMEOUT_MS`/`LLM_TIMEOUT_MS` (currently unused).
- Add single-retry logic for STT/Qdrant/LLM transient failures and one JSON-repair attempt, per the guide's retry table — never infinite retries, never retry a refusal.
- Add a request ID generated at pipeline entry, threaded through logs and the response.

**Files:** `pipeline/orchestrator.py`, `config/settings.py`, `pipeline/metrics.py`

**Done when:** a simulated Qdrant timeout or a malformed LLM response degrades gracefully instead of crashing, and every log line for one request shares a request ID.

---

## Phase 10 — Real health/readiness checks

`apps/api/routes/health.py:38-46` hardcodes `"pending"` for both services.

- `/health/ready` should actually ping Groq (lightweight models-list call) and Qdrant (collection info call) and report `ok`/`degraded` per service, not a static string.

**Files:** `apps/api/routes/health.py`

**Done when:** killing Qdrant locally flips `/health/ready` to unhealthy for that service specifically.

---

## Phase 11 — Expand the evaluation set

`evaluation/queries.jsonl` currently has 10 English factoid questions only. The guide requires English, Hindi, Hinglish, other Indic languages, exact/semantic questions, unanswerable questions, off-topic questions, and a prompt-injection probe.

**Files:** `evaluation/queries.jsonl`, `evaluation/benchmark.py`, `evaluation/metrics.py`

**Done when:** the set has ≥100–200 queries spanning every required category, and `evaluation/metrics.py` computes recall@K, evidence hit rate, grounded-answer rate, refusal accuracy, JSON validity rate.

---

## Phase 12 — Tests against real components

Re-run and extend `tests/` now that Phases 1–9 replaced the mocks — several tests were likely written/passing against mock embeddings and unstructured generation output, and need updating for the new JSON contract.

**Files:** `tests/test_chunking.py`, `tests/test_retrieval.py`, `tests/test_guardrails.py`, `tests/test_generation.py`, `tests/test_pipeline.py`

**Done when:** `pytest` passes end to end.

---

## Phase 13 — Smoke test

`scripts/smoke_test.py` exists — confirm it actually checks env vars, Groq STT, Groq LLM, Qdrant connection/collection, BM25 index, embedding model, hybrid retrieval, guardrails, structured output, and full pipeline (the guide's 10-point checklist), not a subset.

**Files:** `scripts/smoke_test.py`, `scripts/startup_check.py`

**Done when:** it prints `10/10 PASS`.

---

## Phase 14 — Voice UI verification

Frontend is already wired to the right endpoints (`apps/web/lib/api.ts` calls `/api/query`, `/api/audio`, `/health`). This phase is manual verification + polish, not new wiring:

- Confirm `apps/web/components/AudioRecorder.tsx` actually records and posts audio correctly against the fixed STT endpoint (Phase 6).
- Confirm `apps/web/components/AnswerDisplay.tsx` shows language, retrieval mode, evidence count, grounded flag, confidence, and RAG latency, per the guide's UI spec.

**Files:** `apps/web/components/AudioRecorder.tsx`, `apps/web/components/AnswerDisplay.tsx`, `apps/web/app/page.tsx`

**Done when:** a full voice question → spoken answer round trip works in a browser.

---

## Phase 15 — Benchmark

Run `evaluation/benchmark.py` against the now-real pipeline for 150–200 queries from Phase 11. Report P50/P70/P95/P100 for STT latency and RAG latency **separately**, plus total voice latency. Use only measured numbers.

**Files:** `evaluation/benchmark.py`, `evaluation/metrics.py`

**Done when:** you have a real latency report, not placeholders.

---

## Phase 16 — Deploy, freeze, final rehearsal

- Deploy `apps/api` and `apps/web`, secrets as platform env vars, confirm `.env` is gitignored and only `.env.example` is committed.
- Re-run the benchmark against the deployed config; don't change architecture after that.
- Manual rehearsal from a clean browser: 5 known-good questions, 2 unanswerable, 1 off-topic, 1 multilingual, check latency and errors.

**Files:** `Dockerfile`, `docker-compose.yml`, `.gitignore`, `.env.example`

---

## Priority order if time is short

1. **Phases 1–3 (real dataset + real embeddings, both index-time and query-time)** — without these, retrieval quality is essentially random and nothing else in the guide's grading criteria ("grounded in MSMARCO-XI") can be true.
2. **Phase 7–8 (structured JSON + real guardrails)** — this is the guide's core safety/grounding requirement, currently the weakest link after retrieval.
3. **Phase 9 (timeouts/retries/request IDs)** — the guide explicitly calls out "harness" as a graded requirement, not optional polish.
4. Everything else (chunking variety, health checks, benchmark breadth, UI polish) improves the score but won't break the demo if partially done.
