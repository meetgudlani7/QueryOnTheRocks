"""
Retrieval Tests

Tests for retrieval components.
"""

import heapq

import pytest
from retrieval import QdrantStore, BM25Store, reciprocal_rank_fusion
from retrieval.reranker import RerankerError
from pipeline.schemas import Evidence
import pipeline.retrieval as pipeline_retrieval


@pytest.mark.asyncio
async def test_bm25_store_basic():
    """Test basic BM25 store functionality."""
    store = BM25Store()

    # Add documents
    docs = [
        {"id": "1", "passage": "The quick brown fox", "language": "en"},
        {"id": "2", "passage": "jumps over the lazy dog", "language": "en"},
    ]

    await store.add_documents(docs)

    # Search
    results = await store.search("quick brown", k=2)

    assert len(results) > 0
    assert all("passage" in r for r in results)
    assert all("score" in r for r in results)


def _reference_bm25_search(store: BM25Store, query: str, k: int):
    """
    Independent re-implementation of the *original* BM25Store.search()
    scoring loop — scans every doc_id in range(total_docs) and checks
    inverted-index membership per term, rather than walking postings lists
    directly. Kept deliberately naive/duplicated (not imported from
    bm25_store.py) so this test can't pass by accident just because both
    sides share the same code path.
    """
    query_terms = store._tokenize(query)
    scores = []
    for doc_id in range(store.total_docs):
        score = 0.0
        for term in query_terms:
            if term in store.inverted_index and doc_id in store.inverted_index[term]:
                tf = store.inverted_index[term][doc_id]
                idf = store._idf(term)
                score += store._bm25_score(tf, idf, store.doc_lengths[doc_id])
        if score > 0:
            scores.append((-score, doc_id))
    top_k = heapq.nsmallest(k, scores)
    return [(-neg_score, store.documents[doc_id]["id"]) for neg_score, doc_id in top_k]


@pytest.mark.asyncio
async def test_bm25_search_matches_naive_reference_scan():
    """
    Regression test for the Phase 17 rewrite: the optimized search() (which
    only scans postings lists for the query's terms) must produce byte-for-
    byte identical scores and ranking to the original full-corpus scan, on
    a corpus deliberately sized and varied enough to exercise ties,
    multi-term overlap, repeated query terms, and documents that match zero
    query terms.
    """
    store = BM25Store()
    docs = [
        {"id": "d0", "passage": "the quick brown fox jumps over the lazy dog", "language": "en"},
        {"id": "d1", "passage": "the lazy dog sleeps all day in the sun", "language": "en"},
        {"id": "d2", "passage": "quick foxes are clever and quick", "language": "en"},
        {"id": "d3", "passage": "an entirely unrelated passage about weather", "language": "en"},
        {"id": "d4", "passage": "brown dog, quick dog, lazy fox", "language": "en"},
        {"id": "d5", "passage": "fox fox fox fox fox", "language": "en"},
    ]
    await store.add_documents(docs)

    for query, k in [
        ("quick brown fox", 3),
        ("lazy dog", 10),
        ("fox fox", 2),  # repeated query term, exercises duplicate-term accumulation
        ("nonexistent term entirely", 5),
        ("quick", 1),
    ]:
        expected = _reference_bm25_search(store, query, k)
        actual = await store.search(query, k=k)
        actual_pairs = [(r["score"], r["document_id"]) for r in actual]

        assert len(actual_pairs) == len(expected)
        for (exp_score, exp_id), (act_score, act_id) in zip(expected, actual_pairs):
            assert exp_id == act_id
            assert exp_score == pytest.approx(act_score)


@pytest.mark.asyncio
async def test_bm25_search_runs_off_the_event_loop(monkeypatch):
    """
    Regression test for the concurrency bug itself: search() must delegate
    its scoring work to a worker thread (asyncio.to_thread) rather than
    executing it directly on the calling coroutine — otherwise a slow scan
    blocks every other concurrently-running request on the same event loop.
    """
    import retrieval.bm25_store as bm25_module

    store = BM25Store()
    await store.add_documents([{"id": "1", "passage": "hello world", "language": "en"}])

    called_with = {}
    real_to_thread = bm25_module.asyncio.to_thread

    async def spy_to_thread(func, *args, **kwargs):
        called_with["func"] = func
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(bm25_module.asyncio, "to_thread", spy_to_thread)

    await store.search("hello", k=1)

    assert called_with.get("func") == store._search_sync


def test_rrf_fusion():
    """Test Reciprocal Rank Fusion."""
    qdrant_results = [
        Evidence(passage="Qdrant result 1", score=0.9, source="qdrant", document_id="1", language="en"),
        Evidence(passage="Qdrant result 2", score=0.8, source="qdrant", document_id="2", language="en"),
    ]
    
    bm25_results = [
        Evidence(passage="BM25 result 1", score=0.7, source="bm25", document_id="1", language="en"),
        Evidence(passage="BM25 result 3", score=0.6, source="bm25", document_id="3", language="en"),
    ]
    
    fused = reciprocal_rank_fusion(qdrant_results, bm25_results, k=2)

    assert len(fused) == 2
    assert all(e.source == "fused" for e in fused)
    # Pre-existing bug fixed here: the original assertion was
    # `"document_id" in e.document_id`, a substring check against the
    # literal word "document_id" that could never be true for any real id
    # value — it always failed regardless of whether fusion worked
    # correctly, silently blocking `pytest` from ever passing cleanly.
    # Document "1" appears in both input lists (ranked #1 in each), so it
    # must fuse to the top of the combined result.
    fused_ids = [e.document_id for e in fused]
    assert all(doc_id for doc_id in fused_ids)  # every fused item has a real, non-empty id
    assert fused_ids[0] == "1"
    assert set(fused_ids) <= {"1", "2", "3"}  # only ids that were actually in the inputs


def test_qdrant_store_init():
    """Test QdrantStore initialization."""
    store = QdrantStore()

    assert store.url is not None
    assert store.collection is not None


# ---------------------------------------------------------------------------
# Phase 19: language filter (RETRIEVAL_LANGUAGE_FILTER)
# ---------------------------------------------------------------------------


def test_language_filter_off_by_default():
    """Flag defaults to off — today's unfiltered behavior is unchanged."""
    assert pipeline_retrieval._language_filter("hi") is None


def test_language_filter_includes_english_alongside_requested_language(monkeypatch):
    monkeypatch.setattr(pipeline_retrieval.settings, "RETRIEVAL_LANGUAGE_FILTER", True)
    result = pipeline_retrieval._language_filter("hi")
    assert result == {"must": [{"key": "language", "match": {"any": ["en", "hi"]}}]}


def test_language_filter_dedupes_english(monkeypatch):
    monkeypatch.setattr(pipeline_retrieval.settings, "RETRIEVAL_LANGUAGE_FILTER", True)
    result = pipeline_retrieval._language_filter("en")
    assert result == {"must": [{"key": "language", "match": {"any": ["en"]}}]}


def test_language_filter_no_language_returns_none_even_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline_retrieval.settings, "RETRIEVAL_LANGUAGE_FILTER", True)
    assert pipeline_retrieval._language_filter(None) is None


@pytest.mark.asyncio
async def test_search_qdrant_passes_filter_through_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline_retrieval.settings, "RETRIEVAL_LANGUAGE_FILTER", True)
    captured = {}

    async def fake_search(query, k=20, filter=None):
        captured["filter"] = filter
        return []

    monkeypatch.setattr(pipeline_retrieval.qdrant_store, "search", fake_search)
    await pipeline_retrieval.search_qdrant("test query", k=5, language="hi")

    assert captured["filter"] == {"must": [{"key": "language", "match": {"any": ["en", "hi"]}}]}


@pytest.mark.asyncio
async def test_search_qdrant_no_filter_when_flag_disabled(monkeypatch):
    """Default-off flag must mean zero behavior change to the existing call site."""
    captured = {}

    async def fake_search(query, k=20, filter=None):
        captured["filter"] = filter
        return []

    monkeypatch.setattr(pipeline_retrieval.qdrant_store, "search", fake_search)
    await pipeline_retrieval.search_qdrant("test query", k=5, language="hi")

    assert captured["filter"] is None


# ---------------------------------------------------------------------------
# Phase 20: cross-encoder reranking (RERANKING_ENABLED)
# ---------------------------------------------------------------------------


def _evidence(doc_id: str, passage: str, score: float, source: str = "fused") -> Evidence:
    return Evidence(passage=passage, score=score, source=source, document_id=doc_id, language="en")


@pytest.mark.asyncio
async def test_rerank_evidence_reorders_by_cross_encoder_score(monkeypatch):
    evidence = [
        _evidence("a", "low relevance passage", 0.5),
        _evidence("b", "high relevance passage", 0.4),
    ]

    async def fake_rerank(query, passages):
        # Deliberately invert the fusion order, so a passing test proves
        # reranking actually changed the order rather than passing by
        # coincidence.
        return [0.1, 0.9]

    monkeypatch.setattr(pipeline_retrieval.reranker, "rerank", fake_rerank)

    result = await pipeline_retrieval.rerank_evidence("q", evidence, k=2)

    assert [e.document_id for e in result] == ["b", "a"]
    assert all(e.source == "reranked" for e in result)


@pytest.mark.asyncio
async def test_rerank_evidence_preserves_original_score_scale(monkeypatch):
    """
    Regression test for a real crash found in manual end-to-end testing:
    rerank_evidence() must keep each evidence item's original RRF-scale
    score (from fusion), not substitute in the cross-encoder's raw,
    unbounded logit. guardrails.check_evidence() downstream normalizes
    evidence.score against the theoretical max RRF score — feeding it a
    cross-encoder logit (observed range roughly -10 to +11, nothing like
    RRF's ~0.01-0.03) produced a confidence so out of range it crashed
    QueryResponse's own pydantic validation instead of returning a
    graceful refusal.
    """
    evidence = [
        _evidence("a", "passage one", 0.031),
        _evidence("b", "passage two", 0.029),
    ]

    async def fake_rerank(query, passages):
        # A deliberately extreme, RRF-incompatible logit range, matching
        # the real cross-encoder's observed output magnitude.
        return [-9.8, 10.9]

    monkeypatch.setattr(pipeline_retrieval.reranker, "rerank", fake_rerank)

    result = await pipeline_retrieval.rerank_evidence("q", evidence, k=2)

    scores_by_id = {e.document_id: e.score for e in result}
    assert scores_by_id["a"] == pytest.approx(0.031)
    assert scores_by_id["b"] == pytest.approx(0.029)

    # The cross-encoder score is kept, just not as .score — transparency
    # without breaking the evidence gate's assumed scale.
    reranked_b = next(e for e in result if e.document_id == "b")
    assert reranked_b.metadata["rerank_score"] == pytest.approx(10.9)


@pytest.mark.asyncio
async def test_rerank_evidence_respects_k():
    evidence = [_evidence(str(i), f"passage {i}", 1.0 / i) for i in range(1, 6)]

    async def fake_rerank(query, passages):
        return [float(i) for i in range(len(passages))]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline_retrieval.reranker, "rerank", fake_rerank)
        result = await pipeline_retrieval.rerank_evidence("q", evidence, k=2)

    assert len(result) == 2


@pytest.mark.asyncio
async def test_rerank_evidence_fails_open_on_reranker_error(monkeypatch):
    """A broken reranker must degrade to fusion order, never fail the request."""
    evidence = [
        _evidence("a", "passage one", 0.5),
        _evidence("b", "passage two", 0.4),
    ]

    async def fake_rerank(query, passages):
        raise RerankerError("model unavailable")

    monkeypatch.setattr(pipeline_retrieval.reranker, "rerank", fake_rerank)

    result = await pipeline_retrieval.rerank_evidence("q", evidence, k=2)

    assert [e.document_id for e in result] == ["a", "b"]
    assert all(e.source == "fused" for e in result)  # untouched, not relabeled


@pytest.mark.asyncio
async def test_rerank_evidence_fails_open_on_score_count_mismatch(monkeypatch):
    evidence = [_evidence("a", "p1", 0.5), _evidence("b", "p2", 0.4)]

    async def fake_rerank(query, passages):
        return [0.9]  # wrong length — must not be trusted enough to zip

    monkeypatch.setattr(pipeline_retrieval.reranker, "rerank", fake_rerank)

    result = await pipeline_retrieval.rerank_evidence("q", evidence, k=2)

    assert [e.document_id for e in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_evidence_empty_input_returns_empty():
    assert await pipeline_retrieval.rerank_evidence("q", [], k=5) == []


def test_reranking_disabled_by_default():
    """The default config must not attempt to load the reranker model at all."""
    from config import settings
    assert settings.RERANKING_ENABLED is False


def test_language_filter_disabled_by_default():
    from config import settings
    assert settings.RETRIEVAL_LANGUAGE_FILTER is False
