"""
Dataset Download Module

Fetches a bounded sample of AI4Bharat MSMARCO-XI spanning several Indic
languages, without ever attempting a full download of the dataset (~10M
rows / ~49GB across 13 parquet shards, each ~3.7GB).

Four things about this specific dataset drove the design below, confirmed
by hand before writing this code (see IMPLEMENTATION_ROADMAP.md Phase 1):

1. Both HuggingFace's own row-preview API AND the `datasets` library's
   streaming reader (`load_dataset(..., streaming=True)`) crash on this
   dataset with `ArrowNotImplementedError: Nested data conversions not
   implemented for chunked array outputs`. Streaming row-by-row is
   therefore not an option here.

2. That bug turns out to be pyarrow-specific, not something wrong with the
   file: `pyarrow.parquet.read_table` (and even `read_row_group`) hit the
   identical error, isolated specifically to the nested `passages` column
   (a struct of three parallel lists) — every scalar column reads fine.
   DuckDB's independent parquet reader handles the same file, same column,
   with no issue, so it's used here instead of pyarrow for reading rows.

3. Each shard turns out to be a SINGLE target language end-to-end (shard
   0000 is 100% Assamese, 0003 is 100% Hindi, 0011 is 100% Tamil, etc — one
   shard = the full English MS MARCO train set translated into one Indic
   language). Sampling rows from only one shard, as an earlier version of
   this module did, silently produces a "multilingual" dataset that's
   actually just English + one language. Getting real language diversity
   requires downloading and sampling from *several* shards.

4. Each shard has exactly one giant parquet row group, so random/reservoir
   sampling (`USING SAMPLE reservoir(...)`) forces DuckDB to decode the
   entire ~750K-row, text-heavy file to fill the reservoir — observed
   taking ~15 minutes per shard. A plain `LIMIT n` is pushed down and
   returns in seconds, so this reads the first N usable rows of each
   chosen shard instead of a true random sample within it. That's a
   deliberate speed/randomness trade-off: still a real, diverse slice of
   that language's data, just not shuffled within the shard.
"""

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Custom exception for download errors."""
    pass


REQUIRED_FIELDS = ("query_id", "source_lang", "target_lang", "passages")
PARQUET_LIST_URL = "https://datasets-server.huggingface.co/parquet"
OVERSAMPLE_FACTOR = 3  # accounts for rows _row_is_usable filters out


def _row_is_usable(row: Dict[str, Any]) -> bool:
    """Defensive check so one malformed row can't crash the whole ingest run."""
    if not all(field in row and row[field] is not None for field in REQUIRED_FIELDS):
        return False
    passages = row.get("passages")
    if not isinstance(passages, dict):
        return False
    if not passages.get("Translated_passages") and not passages.get("English_passages"):
        return False
    return True


async def _list_shards(dataset_name: str, split: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    resp = await client.get(
        PARQUET_LIST_URL, params={"dataset": dataset_name, "config": "default", "split": split}
    )
    if resp.status_code != 200:
        raise DownloadError(f"Failed to list parquet shards for {dataset_name}/{split}: {resp.status_code} - {resp.text}")
    all_shards = resp.json().get("parquet_files", [])
    # The API's `split` query param is not reliable — observed returning every
    # split's shards regardless of what was requested — so filter client-side
    # too rather than trusting the server did it.
    shards = [s for s in all_shards if s.get("split") == split]
    if not shards:
        raise DownloadError(f"No parquet shards found for {dataset_name}/{split} (got {len(all_shards)} shards across other splits)")
    return shards


async def _download_shard(url: str, dest: Path, client: httpx.AsyncClient, expected_size: Optional[int]) -> None:
    """
    Streams one shard to disk in chunks (never holds the whole file in
    memory) and skips the download entirely if a complete copy is already
    cached locally with the expected size.
    """
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        logger.info(f"Shard already cached at {dest} ({dest.stat().st_size / 1e9:.2f} GB), skipping download")
        return

    tmp = dest.with_suffix(".part")
    downloaded = 0
    last_log = time.perf_counter()

    async with client.stream("GET", url) as response:
        if response.status_code != 200:
            raise DownloadError(f"Failed to download shard {url}: {response.status_code}")
        with open(tmp, "wb") as f:
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if now - last_log > 5:
                    logger.info(f"Downloading shard: {downloaded / 1e9:.2f} GB so far...")
                    last_log = now

    tmp.replace(dest)  # atomic on POSIX: no half-downloaded shard mistaken for a complete one
    logger.info(f"Shard download complete: {dest} ({downloaded / 1e9:.2f} GB)")


def _read_shard_rows(shard_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """
    Blocking, CPU/IO-bound work — must run off the event loop. Uses DuckDB
    (not pyarrow — see module docstring) to read the first rows of the
    local shard file. `limit` here is already the per-shard, over-sampled
    count computed by the caller.
    """
    import duckdb

    # Quoting: shard_path is always one of our own "_shard_{split}_{filename}"
    # paths built from an API-provided filename, never user input, but guard
    # against the theoretical case of a stray quote anyway.
    safe_path = str(shard_path).replace("'", "''")

    con = duckdb.connect()
    try:
        query = f"SELECT * FROM read_parquet('{safe_path}')"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    if df.empty:
        return []
    return [row for row in df.to_dict("records") if _row_is_usable(row)]


async def download_dataset(
    output_dir: Path = Path("data/msmarco_xi"),
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train",
    limit: Optional[int] = 500,
    seed: int = 42,
    num_language_shards: int = 3,
    max_retries: int = 3,
    shard_download_timeout_s: float = 1800.0,
) -> List[Dict[str, Any]]:
    """
    Fetch a bounded sample of MSMARCO-XI spanning several target languages.

    Args:
        output_dir: local cache directory (both raw shards and the combined
            sample are cached here)
        dataset_name: HuggingFace dataset id
        split: "train" or "validation"
        limit: total rows to return across all sampled shards combined.
            None means "take everything from every sampled shard" — still
            bounded to num_language_shards shards, not the whole dataset,
            but hundreds of thousands of rows, so it's logged loudly.
        seed: selects which shards (languages) are sampled and makes the
            selection reproducible across runs
        num_language_shards: how many different language shards to pull
            from. Each shard is ~3.7GB to download (one-time, cached) — the
            default of 3 means ~11GB total for English + 2 Indic languages.
            Raise this for broader language coverage at the cost of more
            bandwidth/disk; the full dataset has 13 train-split shards.
        max_retries: retries on transient network failures, per shard
        shard_download_timeout_s: hard cap on each shard's one-time
            download; a hung download must still fail cleanly rather than
            block forever.

    Raises:
        DownloadError: network/dataset failure after retries, or zero
            usable rows were found after filtering malformed ones.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_cache = (
        output_dir
        / f"{split}_sample_limit{limit if limit is not None else 'full'}_shards{num_language_shards}_seed{seed}.jsonl"
    )

    cached = _load_cached_dataset(sample_cache)
    if cached:
        logger.info(f"Loaded {len(cached)} cached records from {sample_cache} (delete this file to force a re-fetch)")
        return cached

    if limit is None:
        logger.warning(
            f"DATASET_LIMIT is unset — this will load {num_language_shards} entire "
            f"~750K-row shards of {dataset_name}/{split} into memory. Set DATASET_LIMIT "
            "in .env unless that's actually what you want."
        )

    per_shard_limit = None if limit is None else max(1, -(-limit // num_language_shards) * OVERSAMPLE_FACTOR)

    # HuggingFace's `resolve` URLs 302-redirect to the actual CDN location —
    # without follow_redirects, the streaming download would just receive
    # the redirect response itself instead of file bytes.
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                shards = await _list_shards(dataset_name, split, client)
                rng = random.Random(seed)
                chosen = rng.sample(shards, min(num_language_shards, len(shards)))

                logger.info(
                    f"Attempt {attempt}/{max_retries}: sampling from {len(chosen)} shard(s) "
                    f"({', '.join(s['filename'] for s in chosen)}), "
                    f"~{sum(s.get('size', 0) for s in chosen) / 1e9:.1f} GB total to download"
                )

                # Cap each shard's contribution to its fair share BEFORE combining —
                # concatenating uncapped per-shard results and slicing [:limit] at the
                # end would let an early shard alone satisfy the whole limit and
                # silently drop every later (differently-languaged) shard's rows.
                per_shard_final_cap = None if limit is None else -(-limit // len(chosen))

                all_records: List[Dict[str, Any]] = []
                for shard in chosen:
                    shard_path = output_dir / f"_shard_{split}_{shard['filename']}"
                    await asyncio.wait_for(
                        _download_shard(shard["url"], shard_path, client, shard.get("size")),
                        timeout=shard_download_timeout_s,
                    )
                    shard_records = await asyncio.to_thread(_read_shard_rows, shard_path, per_shard_limit)
                    if per_shard_final_cap is not None:
                        shard_records = shard_records[:per_shard_final_cap]
                    logger.info(f"Using {len(shard_records)} usable rows from {shard['filename']} (target_lang={shard_records[0].get('target_lang') if shard_records else 'n/a'})")
                    all_records.extend(shard_records)

                if not all_records:
                    raise DownloadError(
                        f"Sampled shards yielded zero usable rows after filtering malformed "
                        "ones — this would be unusual and suggests a schema change."
                    )

                logger.info(f"Sampled {len(all_records)} total usable rows across {len(chosen)} shard(s)")
                _save_cached_dataset(sample_cache, all_records)
                return all_records

            except asyncio.TimeoutError:
                last_error = DownloadError(f"A shard download did not finish within {shard_download_timeout_s}s")
                logger.warning(str(last_error))
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                backoff = min(2 ** attempt, 30)
                logger.info(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

    raise DownloadError(f"Failed to download {dataset_name}/{split} after {max_retries} attempts: {last_error}")


def _load_cached_dataset(cache_file: Path) -> List[Dict[str, Any]]:
    if not cache_file.exists():
        return []
    try:
        records = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping corrupt cache line {line_no} in {cache_file}")
        return records
    except OSError as e:
        logger.warning(f"Failed to read cache {cache_file}: {e}")
        return []


def _save_cached_dataset(cache_file: Path, data: List[Dict[str, Any]]) -> None:
    tmp_file = cache_file.with_suffix(".tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_file.replace(cache_file)  # atomic: no half-written cache file on crash
        logger.info(f"Cached {len(data)} records to {cache_file}")
    except OSError as e:
        logger.error(f"Failed to write cache {cache_file}: {e}")
        tmp_file.unlink(missing_ok=True)
