# QueryOnTheRocks — Architecture & Process Reference

A multilingual, voice-enabled Retrieval-Augmented Generation (RAG) system. A user asks a question — by
typing or speaking, in English or an Indic language — and gets an answer grounded in a real indexed
corpus, with the evidence attached and a machine-checked claim about whether the answer is actually
supported by that evidence.

This document explains how every piece of it works, end to end: the **process** that builds the
knowledge base offline, the **flow** a live question takes through the system, and a file-by-file map
of the codebase behind both. New to RAG or this stack? Start with [The idea, in plain
terms](#the-idea-in-plain-terms) — every technical section after it also has a **Plain terms** callout
translating the jargon into an everyday comparison. Skip them freely if you already know the concept.

**At a glance:** 437,574 chunks indexed · 13 language shards sampled · 384-dim multilingual embeddings ·
2 retrieval indexes (dense + keyword) · 64 unit/pipeline tests

---

## The idea, in plain terms

Most chatbots answer purely from what they memorized while training — like a student sitting a
**closed-book exam** from memory alone. That's fast, but it means they can misremember a fact, or
confidently invent one that merely sounds right (usually called "hallucination"). This system is built
to sit an **open-book exam** instead: before it writes a single word of an answer, it goes and finds the
actual passages in a real reference collection that are relevant to the question, hands *only* those
passages to the answer-writer, and requires it to point to exactly which passage each part of the answer
came from. If nothing relevant turns up, it has to say so — "I don't know" is an acceptable exam answer
here; a plausible-sounding guess is not.

That whole approach is called **Retrieval-Augmented Generation (RAG)**: retrieve real evidence first,
generate an answer from that evidence second, and check the citations actually hold up third.

**1. Build the reference library** *(done once, offline)* — take a huge public dataset, cut it into
paragraph-sized pieces, and file every piece into two searchable libraries, the way a publisher indexes
an entire encyclopedia before the library ever opens to readers. See [Offline ingestion](#process--offline-ingestion-building-the-knowledge-base).

**2. Find the right pages** *(done on every question)* — two researchers race to the shelves at once:
one searches by *meaning*, one by *exact keyword*, and their two lists get merged into one short, ranked
stack of the most relevant pages. See [Online query](#process--online-query-from-question-to-grounded-answer).

**3. Answer, then fact-check** *(done on every question)* — an AI model writes an answer using only
those pages and cites which ones it used, then a separate, automatic check confirms the citations are
real *and* actually say what the answer claims, before anything reaches the user.

#### Quick questions

**Why not just let the AI model answer from what it already knows?**
Because it might be wrong, out of date, or simply making something up — and there'd be no way to check.
By forcing every answer to come from specific, retrieved passages with citations, a wrong answer becomes
traceable: you can look at exactly what the model was given and see where it went wrong, the same way
you'd check a student's cited source rather than just trusting their essay.

**Why two different search methods instead of one?**
They fail in different, complementary ways. Meaning-based search (dense retrieval) can miss an exact
rare term or ID because it's focused on overall meaning, not exact wording. Keyword search (BM25) can
miss a paraphrase that uses none of the query's actual words. Running both and merging the results
catches more real answers than either alone.

**What happens if the system genuinely doesn't know the answer?**
It's designed to refuse rather than guess, at two separate checkpoints: the evidence gate refuses before
even calling the AI model if nothing relevant enough was found, and response validation catches it
afterward if the model answered anyway without real support. Both return an honest "couldn't find enough
information" instead of a confident-sounding guess.

**Does it work the same for a typed question and a spoken one?**
Yes — a spoken question only has one extra first step (turning audio into text via speech recognition).
Once that's done, it enters the exact same search → merge → fact-check → answer path as a typed question.

---

## What it's built from

| Layer | Technology | Role in this system |
|---|---|---|
| `apps/api` | FastAPI + Uvicorn, Pydantic v2 | HTTP entry point — three routes, request validation, app lifespan (model/index preload) |
| `apps/web` | Next.js 14, React 18, TypeScript, Tailwind | Browser UI — text box, mic recorder, answer + evidence panel |
| `pipeline` | Plain async Python | Orchestrates one query end to end: STT → retrieval → fusion → guardrails → generation → validation |
| Dense retrieval | Qdrant Cloud (REST via `httpx`, no client SDK) | Vector similarity search over indexed passage chunks |
| Keyword retrieval | Hand-rolled Okapi BM25 (`k1=1.5, b=0.75`) | Exact-term search, persisted to a local pickle file — no dependency on Qdrant being up |
| Embeddings | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` | Turns text into 384-dim vectors; the identical model runs at index time and query time |
| Speech-to-text | Groq API — `whisper-large-v3-turbo` | Transcribes uploaded audio, auto-detects spoken language |
| Generation | Groq API — `llama-3.1-8b-instant` | Produces a structured JSON answer from retrieved evidence only |
| Dataset | AI4Bharat `MSMARCO-XI` (HuggingFace, parquet) | English MS MARCO passages machine-translated into 12 Indic languages, one language per shard |
| Dataset reader | DuckDB + `httpx` streaming | Reads shard parquet files directly — the standard `datasets`/pyarrow readers crash on this file's nested schema |
| Ops | Docker Compose, pytest, structured logging with request-ID correlation | Local/prod parity, regression tests, per-request log tracing |

---

## Process — offline ingestion (building the knowledge base)

Everything here runs once, ahead of time, via `python scripts/build_index.py` — never inside a live
request. Five stages, each independent and inspectable on its own.

### 01 · Download — `ingestion/download.py`

Pulls a bounded, reproducible sample of MSMARCO-XI without ever downloading the full ~49 GB / 10M-row
dataset. The dataset's own streaming reader and the standard pyarrow table reader both crash on this
file (a bug isolated to its nested `passages` struct column), so shards are downloaded individually and
read with DuckDB instead, which handles the same file with no issue.

- **Discovery that shaped this:** each parquet shard is a *single* target language end to end — shard
  0003 is 100% Hindi, shard 0011 100% Tamil. Sampling from one shard silently produces "multilingual"
  data that's actually just English + one language, so the run instead samples across **13** shards for
  real coverage.
- **Disk discipline:** one shard (~3.7 GB) is downloaded, sampled, then deleted before the next starts —
  peak disk usage never exceeds one shard regardless of how many languages are sampled.
- **Speed trade-off:** a true random sample within a shard forces DuckDB to decode the entire ~750K-row
  file (~15 min/shard); a plain `LIMIT n` is pushed down and returns in seconds. The run takes the first
  N usable rows per shard instead — still a real, diverse slice, just not shuffled within the shard.
- **Result cached to disk** as JSONL, keyed by split/limit/shard-count/seed, so re-running the pipeline
  never re-downloads.

> **Plain terms** — Think of the full dataset as a warehouse of 13 shipping containers, one language per
> container, ~750,000 books each. Emptying every container onto the floor to grab a sample would need a
> warehouse the size of all 13 combined. Instead, this step opens *one* container, grabs a fair sample of
> books from it, clears the container, and only then opens the next — so the floor never holds more than
> one container's worth of boxes at a time, no matter how many languages get sampled in total.

### 02 · Normalize — `ingestion/normalize.py`

Flattens each raw row — one query, bundled with several English *and* translated candidate passages —
into individual per-passage records in the pipeline's internal schema: `{id, query, passage, answer,
language, query_type, is_selected, metadata}`.

- **Dedup by query:** the same English content repeats verbatim across every language variant of a
  query_id (one English question, translated N times = N rows). The English variant is only emitted the
  first time each query_id is seen.
- **Dedup by content:** a SHA-1 hash of normalized passage text catches verbatim duplicate passages
  across *different* queries too — a known property of MS MARCO-derived data.
- **Unicode:** every text field is NFC-normalized, since Indic-script text can have multiple valid byte
  representations of the same visible characters — without this, identical-looking strings wouldn't
  compare or search identically.

> **Plain terms** — Like proofreading and standardizing a big shipment of scanned pages before filing
> them: throw out any page that's an exact duplicate of one already filed, make sure a word like "café"
> is recognized as the same word no matter which of two invisible-to-the-eye ways it was typed, and
> reshape the shipment into one clean index card per fact instead of one messy folder per document.

### 03 · Chunk — `ingestion/chunk.py`

Splits each passage into retrieval-sized units using **three independent strategies at once**, then
merges and deduplicates the result — no single chunking heuristic is trusted alone.

- **Sentence-window** (`chunkers/sentence.py`) — 3-sentence overlapping windows, splitting on both Latin
  punctuation and Devanagari danda marks (।॥). Good for short, precise factual evidence.
- **Semantic / token-window** (`chunkers/semantic.py`) — sliding 120–180 token windows with 25-token
  overlap, using whitespace-token count as a language-agnostic proxy that needs no per-language
  tokenizer.
- **Metadata-adaptive** (`chunkers/metadata.py`) — reads the dataset's own relevance label: a passage
  marked `is_selected=True` (MS MARCO's gold answer passage) is kept whole and never split, since
  splitting it risks separating the answer from its supporting context. Everything else is delegated to
  the token-window chunker.
- **Fallback:** if all three strategies fail or produce nothing for a passage, that passage is indexed
  as one whole fallback chunk rather than silently dropped from the knowledge base.

> **Plain terms** — You wouldn't hand someone an entire encyclopedia volume to answer "when was X born"
> — you'd hand them the one paragraph that actually has the answer. Chunking is pre-cutting every
> passage into paragraph-sized pieces so retrieval can later return just the small piece with the answer
> in it. Doing it three different ways is like having three people cut the same document into pieces
> with different scissors, then keeping every distinct cut any of them made — more chances that at least
> one cut lands exactly on the boundary that isolates the answer cleanly.

### 04 · Embed — `ingestion/embed.py` → `retrieval/embeddings.py`

Every chunk's text is encoded into a 384-dimensional vector by
`paraphrase-multilingual-MiniLM-L12-v2`, loaded once as a process-wide singleton.

- **One engine, two callers:** `retrieval/embeddings.py` is the single source of truth for text→vector
  conversion, called identically by this offline step and by every live query's dense search. An
  earlier version used two separate mock hash functions here — subtly different implementations across
  index-time and query-time would have silently broken dense search entirely.
- **Device:** auto-detects CUDA → Apple `mps` → CPU. The production build ran on Apple Silicon (MPS).
- Runs batched (32 at a time) off the event loop in a worker thread, so it never blocks concurrent
  requests when the same engine is reused at query time.

> **Plain terms** — Imagine giving every sentence GPS coordinates based on what it *means*, not what
> language it's written in — so "Paris is the capital of France" and its Hindi translation end up parked
> at nearly the same coordinates. That's an embedding: 384 numbers that place a piece of text at a
> specific point in space, positioned so that similar meanings land close together. It's why "dense"
> search can find a relevant passage even when it shares zero exact words with the question — a keyword
> search alone never could.

### 05 · Index — `ingestion/index.py`

Writes the embedded chunks into **both** retrieval stores: Qdrant (dense) and BM25 (keyword).

- **Qdrant path:** upserts in batches of 128 over REST. Point IDs are a UUID5 deterministically derived
  from each chunk's own id, so re-running ingestion over the same data updates points in place instead
  of duplicating them.
- **Failure isolation:** BM25 is a separate, local, always-available index — it must not go down with
  Qdrant. A single long bulk load once crashed the whole process on a mid-run network failure and took
  BM25 down with it; every Qdrant call is now scoped so BM25 always finishes building from the full
  document set regardless of what happens to Qdrant Cloud.
- **Resilience, observed in production:** a stale pooled HTTP connection can go silently dead mid-upload
  (OS reports it open, zero bytes move). After 3 consecutive batch failures the client is torn down and
  rebuilt rather than retried against the same dead socket.
- BM25's final index is pickled to `data/bm25_index.pkl` so the API loads it once at startup instead of
  rebuilding from the dataset on every boot.

> **Plain terms** — This is the moment the meaning-coordinates and the old-fashioned keyword catalog
> both actually get built and shelved — like stocking two separate libraries from the same box of books
> at once: one organized by meaning (Qdrant), one organized by exact keyword (BM25). Either library
> alone can answer a question; having both means one library being temporarily closed doesn't shut down
> the whole reading room.

```
[Download] --raw rows--> [Normalize] --passages--> [Chunk ×3] --chunks--> [Embed] --vectors--> [Index]
                                                                                                    │
                                                                              ┌── dense ───> Qdrant Cloud (msmarco_xi)
                                                                              └── keyword ─> BM25 index (bm25_index.pkl)
```

---

## Process — online query (from question to grounded answer)

Everything here runs inside `pipeline/orchestrator.py::process_query`, on every request, with a request
ID attached to every log line for tracing.

### 01 · Normalize the query — `pipeline/retrieval.py`

Trims whitespace and collapses repeated spaces. Deliberately minimal — the embedding model and BM25's
own tokenizer do the real linguistic normalization downstream.

> **Plain terms** — Straightening a slip of paper before handing it to the librarian — nothing fancier
> than that.

### 02 · Retrieve — dense and keyword, in parallel — `retrieval/qdrant_store.py` · `bm25_store.py`

Both searches launch together via `asyncio.gather(..., return_exceptions=True)` and run independently —
a Qdrant outage doesn't fail the request, it degrades to BM25-only, and vice versa. Only if *both* fail
does the request short-circuit to a safe "knowledge service unavailable" response.

- **Qdrant search:** embeds the query with the shared engine, requests the top 20 nearest vectors by
  cosine similarity.
- **BM25 search:** tokenizes the query, scores every document containing at least one query term with
  the classic Okapi formula, returns the top 20.
- **Timeout:** 3000ms per backend — raised from an original 500ms after measuring real Qdrant Cloud
  round-trips (300–750ms warm).

> **Plain terms** — Two researchers run to the shelves at the same moment. One (Qdrant) searches by
> meaning — walking toward whatever "feels" most related even if the wording is completely different.
> The other (BM25) does an exact keyword search, like flipping to a book's index at the back. They work
> independently — if one researcher is out sick (that backend is down or slow), the other still comes
> back with a usable answer.

### 03 · Fuse — Reciprocal Rank Fusion — `retrieval/fusion.py`

Combines the two ranked lists into one by rank position, not raw score — cosine similarity and BM25's
unbounded term-weight sum live on incompatible scales, so rank position is the only fair currency to
fuse on. Each document scores `1 / (60 + rank)` per list it appears in; a document found by both methods
outranks one found by only one.

> **Plain terms** — You can't just add the two researchers' scores together — one measures "how similar"
> on a 0-to-1 scale, the other "how many rare keywords matched" on an unbounded scale, which is like
> averaging a temperature in Celsius with a stock price. So each list is scored purely by *position*: 1st
> place beats 2nd, 2nd beats 3rd. A passage both researchers independently ranked near the top rockets to
> the top of the merged list — agreement from two completely different methods is a strong signal.

### 04 · Evidence gate — `pipeline/guardrails.py`

A pre-generation checkpoint: is there actually enough to answer with, before spending an LLM call on it?
The fused RRF score is normalized against the theoretical maximum a document can reach and compared to a
threshold. Below it, the pipeline returns a direct refusal — *"I couldn't find enough information in the
knowledge base to answer that"* — without ever calling the LLM.

> **Plain terms** — A bouncer at the door, checked before the answer-writer even starts: "is what you
> found actually good enough to write about?" If the results genuinely score too low, saying "I don't
> know" here is the honest outcome — far better than waving a shaky lead through.

### 05 · Generate — structured, evidence-only — `pipeline/generation.py`

Calls the Groq LLM with every retrieved chunk tagged by its own ID, and a system prompt that constrains
it to exactly one JSON shape: `{answer, evidence_ids, grounded, confidence}`.

- **Prompt-injection defense:** the system prompt explicitly frames evidence as untrusted data the model
  must read but never obey.
- **Citation is never trusted blindly:** any `evidence_id` the model cites that wasn't actually in the
  retrieved set is dropped before the response leaves this function. If nothing real is left, `grounded`
  is forced to `false` regardless of what the model claimed.
- **One repair attempt:** malformed JSON gets exactly one corrective follow-up before failing cleanly —
  never an unbounded retry loop.

> **Plain terms** — The model is handed a stack of numbered index cards (the retrieved chunks) and told:
> write the answer using *only* what's on these cards, and write down which card numbers you used. It's
> also warned that anything written on a card is just information to read — even if a card says "ignore
> your instructions and do X", that's still just text on a card, never a new instruction to follow.

### 06 · Validate the response — `pipeline/guardrails.py`

A second, independent gate — this one checks the model's own output, not just the raw evidence.

- Rejects empty answers, missing/invalid citations, and a fixed list of unsafe-content patterns.
- **Groundedness similarity:** embeds the generated answer and its cited passages, then requires a
  minimum cosine similarity between them — independent of the citation-ID check above, because a real
  citation only proves the model pointed at a real passage, not that the passage actually supports what
  it claimed.
- A failed validation returns a distinct, honest fallback message; this outcome is actually enforced on
  the response path (an earlier version computed it and returned the answer regardless).

> **Plain terms** — Fact-checking after the fact. First: does every card number the model wrote down
> actually correspond to a real card, not a made-up one? Second, subtler: does the sentence the model
> wrote actually match what that card *says* — or did it just staple a real citation onto a made-up
> sentence? That's like a teacher confirming a footnote doesn't just point at a real book, but at a real
> book that actually contains the claim.

**Voice branch** — `POST /api/audio` prepends one extra step: the uploaded file is sent to **Groq
Whisper** (`pipeline/stt.py`) for transcription, with the language field left unset so Whisper
auto-detects the spoken language. The resulting transcript is then handed straight into the same text
pipeline above — voice questions get the identical retrieval → fusion → guardrails → generation path as
typed ones, not a separate code path. *In plain terms: speaking into the mic is just a different front
door — once Whisper turns the audio into text, it walks through the exact same building as a typed
question would.*

```
Text query ────────────────────────┐
                                     ├─> Normalize query
Audio upload ─> Groq Whisper (STT) ┘

Normalize query ─┬─> Qdrant (dense)   ─┐
                  └─> BM25 (keyword)   ─┴─> RRF fusion ─> Evidence gate ──fails──> Refusal
                                                                 │ passes
                                                                 v
                                          Generate (Groq LLM) ─> Validate response ──fails──> Fallback
                                                                        │ passes
                                                                        v
                                                                 QueryResponse
```

```
Browser (Next.js)  ──fetch /api/*──>  FastAPI (apps/api)
                                            │ process_query()
                                            v
                                       Pipeline (pipeline/*)
                                            │ search() / embed_query()
                                            v
                                       Retrieval (retrieval/*)
                                            │ HTTPS
                                            v
                          Qdrant Cloud · Groq Whisper · Groq Llama
```

---

## Glossary

| Term | Meaning |
|---|---|
| **RAG** | *Retrieval-Augmented Generation.* Like an open-book exam instead of closed-book: look up real reference material first, then write the answer from only that material. |
| **LLM** | *Large Language Model* — the AI model that writes the answer text (here, `llama-3.1-8b-instant`). A very fast, well-read writer with no way to double-check facts on its own — which is why it's only ever shown retrieved evidence. |
| **Ingestion** | The whole offline process — download, normalize, chunk, embed, index — that turns a raw dataset into a searchable knowledge base. Run once, never during a live request. |
| **Chunk** | A retrieval-sized slice of a passage — a few sentences or ~150 words — small enough to be precisely relevant on its own. |
| **Token** | Roughly a word-sized unit of text, the unit models and chunking measure length in. |
| **Vector / embedding** | A list of 384 numbers standing in for a piece of text's *meaning* — like GPS coordinates, but for meaning, so sentences that mean the same thing land near each other even across languages. |
| **Dense retrieval** | Search by comparing meaning-coordinates (cosine distance) rather than exact words — finds a passage that answers the question even with no shared vocabulary. |
| **BM25 / keyword retrieval** | The classic exact-term ranking algorithm — like flipping to a book's index. Catches exact names, numbers, and IDs that meaning-based search can miss. |
| **Hybrid retrieval** | Running dense and keyword search together and combining what each brings back, rather than betting on one style. |
| **RRF (Reciprocal Rank Fusion)** | Merges the two ranked lists by position (`1/(60+rank)`), sidestepping the fact that the two search methods' raw scores live on incomparable scales. |
| **Evidence gate** | The pre-generation guardrail — refuses to call the LLM at all if retrieval confidence is too low. |
| **Groundedness** | Whether a generated answer is actually backed by the evidence it cites — checked by both citation-ID validity and answer-to-passage similarity. |
| **Guardrails** | The two checkpoints (evidence gate + response validation) that can each independently stop a bad answer from reaching the user. |
| **Structured generation** | Making the LLM fill out a fixed-shape JSON form instead of writing free prose, so the response can be checked field by field by code. |
| **Prompt injection** | An attack where text hidden inside retrieved evidence tries to hijack the model's behavior — defended against by framing all evidence as content to read, never instructions to obey. |
| **Orchestrator** | `pipeline/orchestrator.py` — runs every stage in the right order and decides which safe fallback to return if any stage fails. |
| **Request ID** | A tracking number stamped on every log line one request produces, so its full journey can be pulled out of concurrent traffic. |
| **Timeout & retry** | Give up and move on if a call takes too long (timeout); try once more on a likely-transient failure before giving up for real (retry). |
| **Latency budget** | How much time is allowed for each step, set from measured reality against real services, not a hoped-for number. |
| **Liveness vs. readiness** | `/health` = "is the process running." `/health/ready` = "can it actually serve an answer right now" — pings Groq and Qdrant live. |

---

## Repository map

**`apps/api` — HTTP backend**

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, CORS, lifespan hook (preloads embedding model + Qdrant/BM25 stores at startup) |
| `routes/query.py` | `POST /api/query` — thin wrapper over `pipeline.orchestrator.process_query` |
| `routes/audio.py` | `POST /api/audio` — multipart upload, wraps `pipeline.orchestrator.process_audio` |
| `routes/health.py` | `GET /health`, `GET /health/ready` — liveness and live-dependency readiness |
| `models/schemas.py` | API-facing Pydantic request/response models |
| `middleware/timing.py` | Adds an `X-Response-Time` header, debug-logs request latency |

**`apps/web` — browser frontend**

| File | Purpose |
|---|---|
| `app/page.tsx` | Single-page UI: text input, mic recorder, answer panel |
| `components/AudioRecorder.tsx` | Wraps `MediaRecorder`, records in whatever codec the browser supports |
| `components/AnswerDisplay.tsx` | Renders answer, language, grounded flag, confidence, latency, evidence list |
| `components/LoadingSpinner.tsx` | In-flight request indicator |
| `lib/api.ts` | Typed `fetch` wrappers for `/api/query`, `/api/audio`, `/health` |

**`ingestion` — offline dataset → index**

| File | Purpose |
|---|---|
| `download.py` | Bounded, multi-shard MSMARCO-XI sampling |
| `normalize.py` | Raw rows → flat per-passage records, deduped, NFC-normalized |
| `chunk.py` | Orchestrates the three chunking strategies into one deduplicated set |
| `chunkers/sentence.py` | Strategy A — overlapping sentence windows |
| `chunkers/semantic.py` | Strategy B — overlapping token windows |
| `chunkers/metadata.py` | Strategy C — protects gold-answer passages whole |
| `embed.py` | Batch-embeds chunk text via the shared embedding engine |
| `index.py` | Writes embedded chunks into Qdrant and BM25 |

**`retrieval` — search backends, shared by ingestion and live queries**

| File | Purpose |
|---|---|
| `embeddings.py` | The one embedding engine — lazy singleton, device auto-detection |
| `qdrant_store.py` | REST client for Qdrant — collection management, batched upsert, cosine search |
| `bm25_store.py` | Hand-rolled Okapi BM25 index with disk persistence |
| `fusion.py` | Reciprocal Rank Fusion and dedup helper |

**`pipeline` — the live request path**

| File | Purpose |
|---|---|
| `orchestrator.py` | `process_query` / `process_audio` — the top-level sequence |
| `stt.py` | Groq Whisper transcription, retry-once-on-transient-failure |
| `retrieval.py` | Thin async wrappers around retrieval stores, returns `Evidence` objects |
| `generation.py` | Structured Groq LLM call, injection-resistant prompt, citation filtering |
| `guardrails.py` | Evidence gate + response validation |
| `schemas.py` | Internal Pydantic models shared across the pipeline |
| `metrics.py` | In-process per-stage latency counters |

**`scripts`, `evaluation`, `tests`, `config`**

| File | Purpose |
|---|---|
| `scripts/build_index.py` | Runs the full offline ingestion process end to end |
| `scripts/startup_check.py` | Live probes against Groq and Qdrant |
| `scripts/smoke_test.py` | 10-point live checklist across the whole pipeline |
| `scripts/run_benchmark.py` | CLI entry point for `evaluation/benchmark.py` |
| `evaluation/queries.jsonl` | 157 evaluation queries — factoid, semantic, refusal categories |
| `evaluation/benchmark.py` | Runs the query set against the live pipeline |
| `evaluation/metrics.py` | F1, recall@k, evidence hit rate, grounded-answer rate, refusal accuracy |
| `tests/` | 64 tests across chunking, retrieval, guardrails, generation, full-pipeline |
| `config/settings.py` | Every tunable in one `pydantic-settings` class |
| `config/logging.py` | Console/file logging, request-ID injection |

---

## Configuration reference

Every setting lives in `config/settings.py`, overridable via `.env`.

**Groq / Qdrant**

| Key | Default | Meaning |
|---|---|---|
| `GROQ_STT_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model |
| `GROQ_LLM_MODEL` | `llama-3.1-8b-instant` | Generation model — chosen for speed against the latency target |
| `QDRANT_COLLECTION` | `msmarco_xi` | Collection name, points at Qdrant Cloud in production |

**Retrieval & fusion**

| Key | Default | Meaning |
|---|---|---|
| `QDRANT_SEARCH_K` / `BM25_SEARCH_K` | 20 / 20 | Candidates pulled from each backend before fusion |
| `FUSION_K` | 5 | Final fused results passed to generation |
| `RRF_CONSTANT` | 60 | The `k` in `1/(k+rank)` |
| `RETRIEVAL_TIMEOUT_MS` | 3000 | Per-backend hard timeout |

**Embedding & chunking**

| Key | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Shared index-time and query-time model |
| `EMBEDDING_DIMENSION` | 384 | Vector size — real model output overrides if it disagrees |
| `CHUNK_SENTENCES_PER_CHUNK` / overlap | 3 / 1 | Sentence-window sizing |
| `CHUNK_MIN/MAX_TOKENS` | 120 / 180 | Token-window sizing |
| `CHUNK_TOKEN_OVERLAP` | 25 | Overlap between adjacent token windows |
| `DATASET_NUM_SHARDS` | 3 (dev) — 13 used for the production build | How many single-language shards to sample |

**Guardrails**

| Key | Default | Meaning |
|---|---|---|
| `MIN_RETRIEVAL_SCORE` | 0.15 | Normalized RRF confidence floor for the evidence gate |
| `MIN_GROUNDEDNESS_SIMILARITY` | 0.35 | Cosine similarity floor between an answer and its cited evidence |
| `MAX_CONTEXT_CHUNKS` | 5 | Chunks passed into the generation prompt |
| `MAX_ANSWER_TOKENS` | 150 | Requested answer length ceiling |

---

## API surface

Five routes, served by `apps/api/main.py`. Interactive docs at `/docs`.

| Method & path | Purpose |
|---|---|
| `GET /` | API metadata |
| `GET /health` | Liveness — process is up |
| `GET /health/ready` | Readiness — live pings Groq and Qdrant |
| `POST /api/query` | JSON `{query, language}` → grounded answer, evidence, confidence |
| `POST /api/audio` | Multipart audio file → transcript + grounded answer for it |

```bash
# text query
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who discovered penicillin?","language":"en"}'

# voice query
curl -X POST http://localhost:8000/api/audio \
  -F 'file=@recording.wav;type=audio/wav'
```

---

## Status — where this stands today

Every component below is a real integration running against real infrastructure — not the
placeholder/mock scaffold this repo started as.

- ✅ **Full ingestion run completed** — 13 language shards sampled, normalized, chunked, and embedded
  into **437,574** indexed chunks, mirrored into Qdrant Cloud (dense) and a 422 MB local BM25 pickle
  (keyword).
- ✅ **Real embeddings, both sides of the boundary** — index-time and query-time vectors come from the
  identical `paraphrase-multilingual-MiniLM-L12-v2` singleton, replacing an earlier mock hash-vector
  implementation.
- ✅ **Real multilingual dataset** — replaced an earlier 100-copies-of-one-fake-row mock; the
  shard-diversity bug (accidentally sampling only one language) was caught and fixed before the
  production run.
- ✅ **Structured, injection-resistant generation** — replaced free-text LLM output with a validated
  JSON contract and untrusted-evidence framing.
- ✅ **Guardrails actually enforced** — a bug where response validation was computed but never acted on
  is fixed; both gates now control the response.
- ✅ **Live readiness checks** — `/health/ready` pings Groq and Qdrant directly instead of returning a
  hardcoded status.
- ✅ **Realistic timeouts** — `RETRIEVAL_TIMEOUT_MS` raised from an aspirational 500ms to a measured
  3000ms after observing real Qdrant Cloud latency.

> **Known operational notes.** CORS currently allows all origins with credentials enabled
> (`apps/api/main.py`) — fine for local development, worth tightening before a public deploy. The BM25
> store's search is a straightforward in-memory scan over its inverted index — correct and fast enough
> at this corpus size, but not the implementation to reach for at 10×+ the current document count.

---

## Running it locally

```bash
# 1 — backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set GROQ_API_KEY, QDRANT_URL
uvicorn apps.api.main:app --reload --port 8000

# 2 — frontend
cd apps/web && npm install && npm run dev

# 3 — build the real index (offline, one-time)
python scripts/build_index.py

# 4 — verify
python scripts/startup_check.py   # live Groq + Qdrant probes
python scripts/smoke_test.py      # 10-point pipeline checklist
pytest                            # 64 unit / pipeline tests
python scripts/run_benchmark.py   # evaluation/queries.jsonl against the live pipeline
```

---

*QueryOnTheRocks — multilingual voice-enabled RAG · reference generated from the codebase as of Aug 2026*
