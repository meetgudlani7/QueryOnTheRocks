# Optimization Roadmap — Phases 17–25

Continues from [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md), which ends at Phase 16 with a
working, fully-real (no mocks) end-to-end system. This roadmap starts from that working baseline and
addresses the gaps identified in a full-codebase review: retrieval quality, latency, concurrency
correctness, safety, and operability.

**Ground rule for every phase below:** nothing here changes an existing public contract (function
signature, API response shape, config default) without (a) a feature flag defaulting to the *current*
behavior, (b) a benchmark run proving the new behavior is neutral-or-better before the flag's default
flips, and (c) all 64 existing tests still passing unmodified. Each phase should be its own PR/commit,
independently revertible.

---

## Phase 17 — Fix the accidental request-serialization bug

**Why first:** everything downstream about concurrency and latency is measuring a system that has a real
bug in it today. `retrieval/bm25_store.py::search()` is declared `async def` but contains zero `await`
statements (confirmed by inspection) — it's a synchronous O(`total_docs`) scan (`for doc_id in range(self.total_docs): ...`) running directly on FastAPI's single event loop. Every concurrent request
stalls behind whichever one is currently mid-scan, and the scan itself is doing far more work than it
needs to.

- Move the scan off the event loop: wrap it in `asyncio.to_thread`, mirroring the exact pattern already
  used in `retrieval/embeddings.py:124`.
- Fix the scan itself: iterate `self.inverted_index[term]` postings for the query's terms and accumulate
  scores in a `defaultdict`, instead of looping `range(self.total_docs)` and checking membership per
  document. Same BM25 math, same output ranking — pure algorithmic fix.
- Add a regression test asserting identical top-k ordering and scores before/after the rewrite on a
  fixed small corpus, so this is provably a pure performance change, not a scoring change.

**Files:** `retrieval/bm25_store.py`, `tests/test_retrieval.py`

**Done when:** `pytest` passes unchanged; a manual concurrency test (fire 10 concurrent `/api/query`
requests) shows latency no longer serializes around BM25.

---

## Phase 18 — Calibrate the guardrail thresholds for real

`MIN_RETRIEVAL_SCORE` and `MIN_GROUNDEDNESS_SIMILARITY` are marked in `config/settings.py` as
"provisional... pending real tuning against an evaluation set." That evaluation set (`evaluation/queries.jsonl`)
now exists — this phase actually does the tuning.

- New `scripts/calibrate_guardrails.py`: sweeps both thresholds across a grid, runs the benchmark at each
  combination, reports refusal accuracy / grounded-answer rate / average F1 per combination.
- Pick the combination that maximizes a combined score (refusal accuracy weighted at least equally with
  grounded-answer rate — a system that answers confidently and wrongly is worse than one that refuses too
  often).
- Update `.env.example` defaults with the measured values, and replace the "provisional" comment in
  `config/settings.py` with the actual measured tradeoff.

**Files:** `scripts/calibrate_guardrails.py` (new), `config/settings.py`, `.env.example`

**Done when:** benchmark shows measured, justified threshold values, not placeholders.

---

## Phase 19 — Wire the language filter that already exists but is unused

`QdrantStore.search()` accepts a `filter` parameter (`retrieval/qdrant_store.py:173`) that `pipeline/retrieval.py`
never passes — a Hindi question currently searches the full multilingual index instead of being scoped to
Hindi + English.

- Thread the request's language (explicit `language` field, or STT's detected language for voice) through
  `pipeline/retrieval.py::search_qdrant` into a Qdrant payload filter on `{language: [detected, "en"]}`.
- Feature flag: `RETRIEVAL_LANGUAGE_FILTER` (default **off**). Turn on only after an A/B benchmark run
  (filtered vs. unfiltered) shows it doesn't hurt recall — a wrong language *detection* filtering out the
  right passage would be a worse regression than the status quo.

**Files:** `pipeline/retrieval.py`, `config/settings.py`

**Done when:** A/B benchmark on `evaluation/queries.jsonl` (grouped by language) shows filtered recall@5 ≥
unfiltered; flag flips to default-on only then.

---

## Phase 20 — Reranking

RRF fusion is a positional heuristic; nothing today re-scores the actual top candidates against the
actual query with a learned relevance signal. This is usually the single biggest lever on answer quality
in a RAG system.

- New `retrieval/reranker.py`, same lazy-singleton/device-auto-detect pattern as `retrieval/embeddings.py`,
  wrapping a multilingual cross-encoder (e.g. `bge-reranker-v2-m3`).
- Retrieve a wider candidate set from fusion (e.g. top 20 instead of top 5), rerank, keep top
  `FUSION_K` after reranking, before the evidence gate.
- Feature flag `RERANKING_ENABLED` (default **off**), and it must **fail open**: if the reranker errors
  or times out, fall back to the pre-rerank RRF order rather than failing the request — consistent with
  the existing resilience posture (Qdrant/BM25 already fail independently without failing the request).

**Files:** `retrieval/reranker.py` (new), `pipeline/retrieval.py`, `config/settings.py`

**Done when:** benchmark shows measurable recall@5/groundedness improvement with reranking on, and a
measured (not assumed) added-latency number, before flipping the default.

---

## Phase 21 — Streaming answers

The API currently returns one JSON blob only after STT (if voice) → retrieval → fusion → guardrail →
generation → validation all finish serially. The 200ms target in `config/settings.py` is explicitly
"architectural, not measured" — streaming is the highest-leverage fix for *perceived* latency, independent
of any backend speed work.

- Switch `pipeline/generation.py` to Groq's streaming chat completions endpoint.
- Add an additive streaming path (e.g. `POST /api/query?stream=true` over SSE) — the existing
  non-streaming `QueryResponse` contract stays the default and is untouched, since `evaluation/benchmark.py`
  and existing tests depend on a single complete response.
- **Real tradeoff to be explicit about:** guardrails (evidence gate before generation, groundedness
  validation after) inherently need the complete answer before they can pass/fail it. A streamed answer is
  therefore provisional until validation completes — the frontend needs a visible "verifying…" state that
  resolves to a grounded/confidence badge, and must be able to visually flag or retract a streamed answer
  that fails post-hoc validation. This isn't a free win; it's shipping useful partial information ahead of
  a trust signal that arrives slightly later.
- Frontend: `AnswerDisplay.tsx` / `page.tsx` updated to render streamed tokens with that provisional state.

**Files:** `pipeline/generation.py`, `apps/api/routes/query.py`, `apps/web/components/AnswerDisplay.tsx`, `apps/web/lib/api.ts`

**Done when:** time-to-first-visible-token drops sharply in manual testing; the existing non-streaming
path and all its tests are provably untouched.

---

## Phase 22 — Concurrency control (bounded, in-process — not an external queue)

Builds directly on Phase 17. See the queuing discussion above for the reasoning; summary of what actually
gets built:

- `asyncio.Semaphore` bounding concurrent Groq LLM/STT calls, sized to a new `GROQ_MAX_CONCURRENT`
  setting matched to your actual Groq plan limits — turns a burst into "briefly waits" instead of "trips
  429s."
- Micro-batching inside `retrieval/embeddings.py`: coalesce `embed_query` calls arriving within a short
  window (~10–20ms) into one batched `model.encode()` call. Purely internal to the module — the
  `embed_texts`/`embed_query` public interface is unchanged, so every caller (ingestion, retrieval,
  groundedness check) needs zero changes.
- A capacity ceiling that returns a clean `503` past a configured limit, instead of accepting unbounded
  concurrent requests thapt degrade everyone's latency silently.
- Explicitly **not** in scope here: an external broker (Celery/Redis/SQS). Revisit only if there's an
  actual need for durability across restarts or multi-machine horizontal scaling — premature at the
  current single-process scale.

**Files:** `pipeline/generation.py`, `pipeline/stt.py`, `retrieval/embeddings.py`, `apps/api/main.py`, `config/settings.py`

**Done when:** a load test (e.g. `hey`/`locust` against `/api/query`) shows bounded, predictable latency
under burst load instead of unbounded degradation; single-request behavior is unchanged.

---

## Phase 23 — Safety and abuse hardening

Two gaps found in review: the "unsafe content" check is 5 regex keywords
(`_UNSAFE_PATTERNS` in `pipeline/guardrails.py`) — trivially bypassed and prone to false-positiving on
legitimate content (e.g. a history question mentioning "violence") — and the API has no auth or rate
limiting at all.

- Replace the regex list with a real moderation call (Groq's moderation endpoint, or a small classifier),
  behind the same `GuardrailsResult` contract so `orchestrator.py` doesn't change.
- Add API-key auth (a header check via FastAPI `Depends()`) and per-key rate limiting (e.g. `slowapi` or
  a simple token bucket) as additive middleware — route logic itself doesn't change.
- Tighten CORS from `allow_origins=["*"]` + credentials (`apps/api/main.py`) to an explicit configured
  allow-list.

**Files:** `pipeline/guardrails.py`, `apps/api/main.py`, `config/settings.py`, `scripts/smoke_test.py` (add a moderation check)

**Done when:** `smoke_test.py` covers the new moderation path; manual verification that unauthenticated
or over-limit requests are rejected with a clear error, not a silent hang or 500.

---

## Phase 24 — CI and continuous evaluation

There is currently no CI of any kind (`.github/` doesn't exist) — the 64 tests exist but nothing enforces
they run before a merge, and quality regressions in retrieval/guardrails would only be caught manually.

- `.github/workflows/ci.yml`: run `pytest` and `scripts/smoke_test.py` on every PR.
- Scheduled (nightly or per-merge-to-main) run of `evaluation/benchmark.py` against
  `evaluation/queries.jsonl`, with results appended to a tracked log/dashboard — so a quality regression
  (e.g. from a threshold or model change) is caught automatically, not discovered by a user.
- OpenTelemetry spans per pipeline stage (STT/retrieval/fusion/guardrails/generation/validation), exported
  to a configurable backend, replacing "grep the local log" as the only way to diagnose latency.

**Files:** `.github/workflows/ci.yml` (new), `config/logging.py`, new `evaluation/` tracking script

**Done when:** a PR with an intentionally broken test fails CI; a dashboard/log shows the last N benchmark
runs' quality metrics over time.

---

## Phase 25 — Multi-turn memory and TTS

Two product-level gaps: every question is handled independently (no follow-up support), and the "voice"
system only listens, never speaks back.

- Add optional session-scoped state (`session_id`, a short rolling history of prior Q/A turns) so a
  follow-up like "and in French?" resolves against the previous turn's context. Purely additive: existing
  callers that don't pass a `session_id` get today's single-turn behavior unchanged.
- Add TTS for voice answers (Groq or another TTS API) to close the voice loop end to end.

**Files:** `pipeline/orchestrator.py`, `pipeline/schemas.py`, `apps/api/routes/query.py`, `apps/api/routes/audio.py`, `apps/web/components/AudioRecorder.tsx`

**Done when:** a manual 2–3 turn conversation resolves follow-ups correctly; a full voice round trip
(speak question → hear spoken answer) works end to end.

---

## Suggested execution order

Phases 17–19 are corrections to existing behavior (bug fix, calibration, wiring up dead code) — lowest
risk, do these regardless of what else gets prioritized. Phases 20–22 are the real quality/speed
step-changes and depend on 17–19 being done first (reranking and concurrency work are best measured
against a system that isn't already bottlenecked by the Phase 17 bug). Phases 23–25 can run in parallel
with 20–22 once 17–19 are merged, since they touch different parts of the system (safety/ops vs.
retrieval/latency vs. product features).
