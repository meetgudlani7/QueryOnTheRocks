"""
Chunking Tests

Covers all three strategies (sentence, semantic/token, metadata-adaptive)
individually plus the hybrid orchestrator that combines them.
"""

import pytest

from ingestion.chunk import ChunkingError, chunk_documents
from ingestion.chunkers import MetadataChunker, SemanticChunker, SentenceChunker


# ---------------------------------------------------------------------------
# Strategy A: SentenceChunker
# ---------------------------------------------------------------------------

class TestSentenceChunker:
    def test_groups_sentences_with_overlap(self):
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=1)
        text = "One. Two. Three. Four."
        chunks = chunker.chunk(text)

        assert len(chunks) == 3  # windows: [One,Two] [Two,Three] [Three,Four]
        assert chunks[0]["text"] == "One. Two."
        assert chunks[1]["text"] == "Two. Three."
        assert chunks[2]["text"] == "Three. Four."
        assert all(c["type"] == "sentence" for c in chunks)

    def test_no_overlap_when_configured_zero(self):
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=0)
        chunks = chunker.chunk("One. Two. Three. Four.")
        assert [c["text"] for c in chunks] == ["One. Two.", "Three. Four."]

    def test_empty_text_returns_no_chunks(self):
        chunker = SentenceChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_without_sentence_punctuation_falls_back_to_whole_text(self):
        """A run-on clause with no terminators must not silently vanish."""
        chunker = SentenceChunker()
        chunks = chunker.chunk("no punctuation here just words")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "no punctuation here just words"

    def test_indic_danda_terminator_is_recognized(self):
        chunker = SentenceChunker(sentences_per_chunk=1, overlap_sentences=0)
        chunks = chunker.chunk("यह पहला वाक्य है। यह दूसरा वाक्य है।")
        assert len(chunks) == 2

    def test_never_drops_short_or_long_sentences(self):
        """Old buggy version silently dropped sentences outside a length range."""
        chunker = SentenceChunker(sentences_per_chunk=1, overlap_sentences=0)
        chunks = chunker.chunk("Yes. " + ("word " * 100).strip() + ".")
        assert len(chunks) == 2

    def test_rejects_overlap_greater_or_equal_to_window(self):
        with pytest.raises(ValueError):
            SentenceChunker(sentences_per_chunk=2, overlap_sentences=2)

    def test_no_empty_chunks_in_output(self):
        chunker = SentenceChunker()
        chunks = chunker.chunk("One. Two. Three. Four. Five.")
        assert all(c["text"].strip() for c in chunks)


# ---------------------------------------------------------------------------
# Strategy B: SemanticChunker (token windows)
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def test_short_text_returns_single_chunk(self):
        chunker = SemanticChunker(min_tokens=120, max_tokens=180, overlap_tokens=25)
        chunks = chunker.chunk("just a few words here")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "just a few words here"

    def test_long_text_windows_with_overlap(self):
        chunker = SemanticChunker(min_tokens=10, max_tokens=20, overlap_tokens=5)
        text = " ".join(f"word{i}" for i in range(50))
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        # verify actual overlap: last 5 tokens of chunk N == first 5 tokens of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i]["text"].split()[-5:]
            head = chunks[i + 1]["text"].split()[:5]
            assert tail == head

    def test_windows_respect_max_tokens(self):
        chunker = SemanticChunker(min_tokens=10, max_tokens=20, overlap_tokens=5)
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunker.chunk(text)
        assert all(len(c["text"].split()) <= 20 for c in chunks)

    def test_empty_text_returns_no_chunks(self):
        chunker = SemanticChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_rejects_overlap_greater_or_equal_to_max_tokens(self):
        with pytest.raises(ValueError):
            SemanticChunker(min_tokens=10, max_tokens=20, overlap_tokens=20)

    def test_rejects_min_greater_than_max(self):
        with pytest.raises(ValueError):
            SemanticChunker(min_tokens=200, max_tokens=100)

    def test_no_empty_chunks_in_output(self):
        chunker = SemanticChunker(min_tokens=5, max_tokens=10, overlap_tokens=2)
        text = " ".join(f"word{i}" for i in range(40))
        chunks = chunker.chunk(text)
        assert all(c["text"].strip() for c in chunks)


# ---------------------------------------------------------------------------
# Strategy C: MetadataChunker (adaptive)
# ---------------------------------------------------------------------------

class TestMetadataChunker:
    def test_selected_passage_is_never_split(self):
        chunker = MetadataChunker(semantic_chunker=SemanticChunker(min_tokens=2, max_tokens=3, overlap_tokens=1))
        long_passage = " ".join(f"word{i}" for i in range(20))
        docs = [{"id": "1", "passage": long_passage, "language": "en", "is_selected": True, "metadata": {}}]

        chunks = chunker.chunk(docs)

        assert len(chunks) == 1
        assert chunks[0]["text"] == long_passage
        assert chunks[0]["chunk_strategy"] == "metadata_protected"

    def test_non_selected_passage_is_delegated_and_split(self):
        chunker = MetadataChunker(semantic_chunker=SemanticChunker(min_tokens=2, max_tokens=3, overlap_tokens=1))
        long_passage = " ".join(f"word{i}" for i in range(20))
        docs = [{"id": "1", "passage": long_passage, "language": "en", "is_selected": False, "metadata": {}}]

        chunks = chunker.chunk(docs)

        assert len(chunks) > 1
        assert all(c["chunk_strategy"] == "metadata_delegated" for c in chunks)

    def test_missing_is_selected_defaults_to_splittable(self):
        chunker = MetadataChunker()
        docs = [{"id": "1", "passage": "short passage", "language": "en", "metadata": {}}]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1  # short enough to not actually split

    def test_all_chunks_tagged_type_metadata(self):
        chunker = MetadataChunker()
        docs = [
            {"id": "1", "passage": "Passage 1", "language": "en", "metadata": {"key": "value1"}},
            {"id": "2", "passage": "Passage 2", "language": "en", "metadata": {"key": "value2"}},
        ]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 2
        assert all(c["type"] == "metadata" for c in chunks)

    def test_empty_passage_is_skipped_not_crashed(self):
        chunker = MetadataChunker()
        docs = [
            {"id": "1", "passage": "", "language": "en", "metadata": {}},
            {"id": "2", "passage": "   ", "language": "en", "metadata": {}},
            {"id": "3", "passage": "real content", "language": "en", "metadata": {}},
        ]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"].startswith("3")

    def test_metadata_is_preserved(self):
        chunker = MetadataChunker()
        docs = [{
            "id": "1", "passage": "content", "language": "hi", "query": "q?",
            "answer": "a", "is_selected": True, "query_type": "factoid",
            "metadata": {"source_lang": "en", "target_lang": "hi"},
        }]
        chunks = chunker.chunk(docs)
        meta = chunks[0]["metadata"]
        assert meta["source_lang"] == "en"
        assert meta["target_lang"] == "hi"
        assert meta["query"] == "q?"
        assert meta["is_selected"] is True
        assert meta["query_type"] == "factoid"


# ---------------------------------------------------------------------------
# Hybrid orchestrator: chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def _doc(self, doc_id, passage, **overrides):
        base = {
            "id": doc_id, "passage": passage, "query": "q", "answer": "a",
            "language": "en", "query_type": "factoid", "is_selected": False,
            "metadata": {"query_id": doc_id},
        }
        base.update(overrides)
        return base

    def test_empty_input_returns_empty_output(self):
        assert chunk_documents([]) == []

    def test_short_passage_produces_deduplicated_single_chunk(self):
        """All three strategies agree trivially on a short passage — must
        collapse to ONE indexed chunk, not three near-identical copies."""
        docs = [self._doc("p1", "A short passage with one clear idea.")]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1
        strategies = chunks[0]["metadata"]["chunk_strategies"]
        assert len(strategies) >= 2  # multiple strategies agreed on this boundary

    def test_long_passage_produces_multiple_distinct_chunks(self):
        long_text = " ".join(f"sentence number {i} has some content." for i in range(60))
        docs = [self._doc("p1", long_text)]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1

    def test_no_empty_chunks_ever(self):
        docs = [self._doc("p1", "Some real content here that is not empty.")]
        chunks = chunk_documents(docs)
        assert all(c["passage"].strip() for c in chunks)

    def test_chunk_ids_are_unique(self):
        long_text = " ".join(f"word{i}" for i in range(300))
        docs = [self._doc("p1", long_text), self._doc("p2", long_text + " different tail content here")]
        chunks = chunk_documents(docs)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved_through_to_output(self):
        docs = [self._doc("p1", "Some content.", language="hi", query_type="descriptive")]
        chunks = chunk_documents(docs)
        assert chunks[0]["language"] == "hi"
        assert chunks[0]["query_type"] == "descriptive"
        assert chunks[0]["metadata"]["query_id"] == "p1"
        assert "chunk_strategy" in chunks[0]["metadata"]
        assert "source_passage_id" in chunks[0]["metadata"]

    def test_selected_passage_stays_whole_even_when_long(self):
        long_text = " ".join(f"word{i}" for i in range(300))
        docs = [self._doc("p1", long_text, is_selected=True)]
        chunks = chunk_documents(docs)
        # at least one chunk must be the untouched, protected whole passage
        assert any(c["passage"] == long_text for c in chunks)

    def test_documents_missing_id_are_skipped_not_crashed(self):
        docs = [{"passage": "no id here", "language": "en", "metadata": {}}, self._doc("p1", "valid doc")]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0]["id"].startswith("p1")

    def test_empty_and_whitespace_passages_are_skipped(self):
        docs = [self._doc("p1", ""), self._doc("p2", "   "), self._doc("p3", "real content here")]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source_passage_id"] == "p3"

    def test_max_chunks_per_passage_is_enforced(self):
        long_text = " ".join(f"word{i}" for i in range(5000))
        docs = [self._doc("p1", long_text)]
        chunks = chunk_documents(docs, max_chunks_per_passage=3)
        assert len(chunks) <= 3

    def test_global_dedup_across_different_passages(self):
        """Identical text appearing under two different passage ids must
        only be indexed once (protects against redundant bloat)."""
        docs = [self._doc("p1", "the exact same repeated sentence."), self._doc("p2", "the exact same repeated sentence.")]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1

    def test_raises_when_all_input_is_unusable(self):
        docs = [self._doc("p1", ""), self._doc("p2", "   ")]
        with pytest.raises(ChunkingError):
            chunk_documents(docs)

    def test_one_strategy_failing_does_not_lose_the_passage(self, monkeypatch):
        """If sentence chunking throws for some pathological input, the
        other strategies must still produce output for that passage."""
        import ingestion.chunk as chunk_module

        original_chunk = chunk_module.SentenceChunker.chunk

        def broken_chunk(self, text, metadata=None):
            raise RuntimeError("simulated chunker failure")

        monkeypatch.setattr(chunk_module.SentenceChunker, "chunk", broken_chunk)
        docs = [self._doc("p1", "Some perfectly normal content.")]
        chunks = chunk_documents(docs)
        assert len(chunks) >= 1
        monkeypatch.setattr(chunk_module.SentenceChunker, "chunk", original_chunk)
