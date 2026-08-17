"""
Data Normalization Module

Converts raw MSMARCO-XI rows (nested, one row per query+language, holding
parallel English/translated passage arrays) into flat, per-passage records
in the pipeline's internal schema:

    {id, query, passage, answer, language, query_type, is_selected, metadata}

Design notes (see IMPLEMENTATION_ROADMAP.md Phase 1 for the full reasoning):

- Each raw row bundles several candidate passages for one query, in both
  English and one target Indic language. We explode each row into one
  record per passage, since retrieval operates over passages, not queries.
- The same underlying English content is repeated verbatim across every
  language-variant row of the same query_id (the same English question
  translated into Hindi, Tamil, Bengali... each becomes its own row in the
  dataset). Without dedup, sampling N language variants would duplicate
  every English passage N times in the index. We track seen query_ids and
  only emit the English variant the first time we see each one.
- Indic-script text can have multiple valid Unicode representations of the
  same visual characters; every text field is NFC-normalized so identical-
  looking strings compare and search identically.
- Exact-duplicate passages (verbatim repeats, which MS MARCO-derived data
  is known to contain) are dropped via a content hash, independent of dedup
  by query_id — this catches duplicates across different queries too.
"""

import hashlib
import logging
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class NormalizationError(Exception):
    """Custom exception for normalization errors."""
    pass


def normalize_data(
    records: List[Dict[str, Any]],
    preserve_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Args:
        records: raw rows as returned by ingestion.download.download_dataset
        preserve_metadata: keep original translation metadata (source_lang,
            target_lang, query_id, translation generation params) on each
            output record

    Returns:
        Flat list of per-passage records in the internal schema.

    Raises:
        NormalizationError: if every input row was malformed / produced no
            usable passages — a real, working dataset should never do this,
            so it signals an upstream problem rather than getting silently
            swallowed.
    """
    if not records:
        return []

    normalized: List[Dict[str, Any]] = []
    seen_english_query_ids: Set[Any] = set()
    seen_content_hashes: Set[Tuple[str, str]] = set()

    stats = {"rows_skipped": 0, "passages_emitted": 0, "passages_deduped": 0, "passages_empty": 0}

    for record in records:
        try:
            emitted = _normalize_row(record, preserve_metadata, seen_english_query_ids, seen_content_hashes, stats)
            normalized.extend(emitted)
        except Exception as e:
            stats["rows_skipped"] += 1
            logger.warning(f"Skipping malformed row (query_id={record.get('query_id')}): {e}")

    logger.info(
        f"Normalized {len(records)} rows -> {len(normalized)} passage records "
        f"(rows_skipped={stats['rows_skipped']}, exact_duplicates_dropped={stats['passages_deduped']}, "
        f"empty_dropped={stats['passages_empty']})"
    )

    if not normalized:
        raise NormalizationError("Normalization produced zero usable passage records from the input")

    return normalized


def _normalize_row(
    record: Dict[str, Any],
    preserve_metadata: bool,
    seen_english_query_ids: Set[Any],
    seen_content_hashes: Set[Tuple[str, str]],
    stats: Dict[str, int],
) -> List[Dict[str, Any]]:
    query_id = record.get("query_id")
    if query_id is None:
        raise ValueError("missing query_id")

    query_type = _clean_text(record.get("query_type") or "unknown").lower() or "unknown"
    source_lang = _clean_text(record.get("source_lang") or "en").lower() or "en"
    target_lang = _clean_text(record.get("target_lang") or "").lower()

    passages = record.get("passages") or {}
    english_passages = passages.get("English_passages") or []
    translated_passages = passages.get("Translated_passages") or []
    is_selected_flags = passages.get("is_selected") or []

    # These three parallel arrays are supposed to be the same length; don't
    # assume the dataset is perfectly clean — truncate to whatever's safe
    # rather than raising and losing the whole row over one short array.
    lengths = {len(english_passages), len(translated_passages)}
    if is_selected_flags:
        lengths.add(len(is_selected_flags))
    if len(lengths) > 1:
        logger.debug(f"query_id={query_id}: passage array length mismatch {lengths}, truncating to shortest")
    n = min(len(english_passages), len(translated_passages))

    base_metadata: Dict[str, Any] = {}
    if preserve_metadata:
        base_metadata = {
            "query_id": query_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translation_meta": record.get("meta") or {},
        }

    emit_english = query_id not in seen_english_query_ids
    eng_query = _clean_text(record.get("Eng_Query") or "")
    eng_answer = _clean_text(record.get("Eng_Answer") or "")
    target_query = _clean_text(record.get("query") or eng_query)
    target_answer = _clean_text(record.get("Answer") or eng_answer)

    out: List[Dict[str, Any]] = []

    for i in range(n):
        is_selected = bool(is_selected_flags[i]) if i < len(is_selected_flags) else False

        if emit_english:
            rec = _build_passage_record(
                doc_id=f"{query_id}_{source_lang}_{i}",
                query=eng_query or target_query,
                passage=_clean_text(english_passages[i]),
                answer=eng_answer or target_answer,
                language=source_lang,
                query_type=query_type,
                is_selected=is_selected,
                metadata={**base_metadata, "passage_index": i, "content_origin": "english"},
            )
            _collect(rec, out, seen_content_hashes, stats)

        if target_lang and target_lang != source_lang:
            rec = _build_passage_record(
                doc_id=f"{query_id}_{target_lang}_{i}",
                query=target_query,
                passage=_clean_text(translated_passages[i]),
                answer=target_answer,
                language=target_lang,
                query_type=query_type,
                is_selected=is_selected,
                metadata={**base_metadata, "passage_index": i, "content_origin": "translated"},
            )
            _collect(rec, out, seen_content_hashes, stats)

    if emit_english:
        seen_english_query_ids.add(query_id)

    return out


def _collect(
    rec: Optional[Dict[str, Any]],
    out: List[Dict[str, Any]],
    seen_content_hashes: Set[Tuple[str, str]],
    stats: Dict[str, int],
) -> None:
    if rec is None:
        stats["passages_empty"] += 1
        return
    if _is_duplicate(rec, seen_content_hashes):
        stats["passages_deduped"] += 1
        return
    out.append(rec)
    stats["passages_emitted"] += 1


def _build_passage_record(
    doc_id: str,
    query: str,
    passage: str,
    answer: str,
    language: str,
    query_type: str,
    is_selected: bool,
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not passage or len(passage) < 3:
        return None
    return {
        "id": doc_id,
        "query": query,
        "passage": passage,
        "answer": answer,
        "language": language or "unknown",
        "query_type": query_type,
        "is_selected": is_selected,
        "metadata": metadata,
    }


def _is_duplicate(rec: Dict[str, Any], seen: Set[Tuple[str, str]]) -> bool:
    key_text = " ".join(rec["passage"].lower().split())
    text_hash = hashlib.sha1(key_text.encode("utf-8")).hexdigest()
    key = (rec["language"], text_hash)
    if key in seen:
        return True
    seen.add(key)
    return False


def _clean_text(text: Any) -> str:
    """NFC-normalize, collapse whitespace, strip. Non-string input becomes ''."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()
