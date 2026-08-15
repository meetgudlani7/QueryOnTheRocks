"""
Smoke Test Script

Runs the guide's 10-point checklist against live components: env vars, Groq
STT, Groq LLM, Qdrant, BM25 index, embedding model, hybrid retrieval,
guardrails, structured output, and the full pipeline. Each check is
independent — one failing does not stop the rest from running.
"""

import asyncio
import io
import logging
import struct
import wave
from typing import Awaitable, Callable, List, Tuple

from config import configure_logging, settings
from pipeline import process_query, QueryRequest, Evidence
from pipeline import generation, guardrails, stt
from pipeline import retrieval as pipeline_retrieval
from retrieval import bm25_store, embeddings, qdrant_store

logger = logging.getLogger(__name__)


def _silent_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """A minimal valid WAV file (silence) — enough to exercise a real STT round trip."""
    n_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


async def _check_env_vars() -> str:
    missing = [
        name for name, value in [
            ("GROQ_API_KEY", settings.GROQ_API_KEY),
            ("QDRANT_URL", settings.QDRANT_URL),
            ("QDRANT_COLLECTION", settings.QDRANT_COLLECTION),
        ] if not value
    ]
    if missing:
        raise RuntimeError(f"missing: {', '.join(missing)}")
    return "GROQ_API_KEY, QDRANT_URL, QDRANT_COLLECTION are set"


async def _check_groq_stt() -> str:
    transcript, _ = await stt.transcribe_audio(_silent_wav_bytes(), format="audio/wav", language="en")
    return f"reached Groq Whisper (transcript={transcript!r})"


async def _check_groq_llm(state: dict) -> str:
    result = await generation.generate_answer(
        query="Where is the Eiffel Tower?",
        context="[e1] The Eiffel Tower is located in Paris, France.",
        valid_evidence_ids=["e1"],
        language="en",
    )
    state["generation_result"] = result
    return f"model={result.model}, latency={result.generation_latency_ms:.0f}ms"


async def _check_structured_output(state: dict) -> str:
    result = state.get("generation_result")
    if result is None:
        raise RuntimeError("Groq LLM check did not produce a result to validate")
    assert isinstance(result.grounded, bool), "grounded must be bool"
    assert 0.0 <= result.confidence <= 1.0, "confidence out of [0,1] range"
    assert all(isinstance(i, str) for i in result.evidence_ids), "evidence_ids must be strings"
    return "response matches {answer, evidence_ids, grounded, confidence} schema"


async def _check_qdrant() -> str:
    if not await qdrant_store.ping():
        raise RuntimeError(f"collection '{settings.QDRANT_COLLECTION}' unreachable")
    return f"collection '{settings.QDRANT_COLLECTION}' reachable"


async def _check_bm25() -> str:
    store = bm25_store.get_store()
    if not store._initialized or store.total_docs == 0:
        raise RuntimeError("index not built — run scripts/build_index.py")
    return f"{store.total_docs} documents indexed"


async def _check_embedding_model() -> str:
    vector = await embeddings.embed_query("smoke test")
    dim = embeddings.get_embedding_dimension()
    if len(vector) != dim:
        raise RuntimeError(f"vector length {len(vector)} != model dimension {dim}")
    return f"model produces {dim}-dim vectors"


async def _check_hybrid_retrieval() -> str:
    normalized = pipeline_retrieval.normalize_query("who discovered penicillin")
    (qdrant_evidence, _), (bm25_evidence, _) = await asyncio.gather(
        pipeline_retrieval.search_qdrant(normalized),
        pipeline_retrieval.search_bm25(normalized),
    )
    fused = pipeline_retrieval.fuse_results(qdrant_evidence, bm25_evidence, k=settings.MAX_CONTEXT_CHUNKS)
    return f"qdrant={len(qdrant_evidence)}, bm25={len(bm25_evidence)}, fused={len(fused)} results"


async def _check_guardrails() -> str:
    empty_gate = await guardrails.check_evidence(query="x", evidence=[])
    if empty_gate.passed:
        raise RuntimeError("evidence gate should refuse empty evidence")

    strong_evidence = [Evidence(
        passage="Alexander Fleming discovered penicillin in 1928.",
        score=0.05, source="fused", document_id="e1", language="en",
    )]
    strong_gate = await guardrails.check_evidence(query="who discovered penicillin", evidence=strong_evidence, min_confidence=0.0)
    if not strong_gate.passed:
        raise RuntimeError("evidence gate should accept strong evidence")

    return "refuses on empty evidence, accepts on strong evidence"


async def _check_full_pipeline() -> str:
    response = await process_query(QueryRequest(query="Who discovered penicillin?", language="en"))
    if not response.answer:
        raise RuntimeError("pipeline returned an empty answer")
    return f"grounded={response.grounded}, confidence={response.confidence:.2f}, latency={response.latency_ms:.0f}ms"


async def main() -> int:
    configure_logging("INFO")
    logger.info("Running smoke tests...\n")

    state: dict = {}
    checks: List[Tuple[str, Callable[[], Awaitable[str]]]] = [
        ("1. Environment variables", _check_env_vars),
        ("2. Groq STT", _check_groq_stt),
        ("3. Groq LLM generation", lambda: _check_groq_llm(state)),
        ("4. Qdrant connection/collection", _check_qdrant),
        ("5. BM25 index", _check_bm25),
        ("6. Embedding model", _check_embedding_model),
        ("7. Hybrid retrieval", _check_hybrid_retrieval),
        ("8. Guardrails", _check_guardrails),
        ("9. Structured output", lambda: _check_structured_output(state)),
        ("10. Full pipeline", _check_full_pipeline),
    ]

    passed = 0
    for name, check in checks:
        try:
            detail = await check()
            logger.info(f"PASS  {name}: {detail}")
            passed += 1
        except Exception as e:
            logger.error(f"FAIL  {name}: {e}")

    logger.info(f"\n{passed}/{len(checks)} PASS")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
