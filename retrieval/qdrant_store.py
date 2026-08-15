"""
Qdrant Vector Store

Real vector search/upsert against Qdrant (Cloud or self-hosted) over REST,
using the shared embedding engine (retrieval.embeddings) so query vectors
are produced by the exact same model as the vectors stored at index time.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from config import settings
from retrieval.embeddings import EmbeddingError, embed_query, get_embedding_dimension

logger = logging.getLogger(__name__)


class QdrantError(Exception):
    """Custom exception for Qdrant errors."""
    pass


def _to_point_id(doc_id: str) -> str:
    """
    Qdrant point IDs must be an unsigned integer or a UUID — arbitrary
    strings like our doc ids ("12345_hi_2") are rejected outright. We
    deterministically derive a UUID from the string instead, which has a
    useful side effect: re-running ingestion over the same source data
    always maps to the same point ids, so re-indexing upserts in place
    rather than duplicating points.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class QdrantStore:
    """Qdrant vector database client with connection reuse and retries."""

    def __init__(self, timeout_s: Optional[float] = None, max_retries: int = 2):
        timeout_s = timeout_s if timeout_s is not None else settings.RETRIEVAL_TIMEOUT_MS / 1000
        self.url = settings.QDRANT_URL.rstrip("/")
        self.api_key = settings.QDRANT_API_KEY
        self.collection = settings.QDRANT_COLLECTION
        self.max_retries = max_retries

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        # One client for the store's lifetime — avoids a fresh TCP/TLS
        # handshake on every request, which matters for the latency budget.
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers)

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self.url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = await self._client.request(method, url, **kwargs)
                if response.status_code >= 500 and attempt <= self.max_retries:
                    logger.warning(
                        f"Qdrant {method} {path} returned {response.status_code}, "
                        f"retrying ({attempt}/{self.max_retries})"
                    )
                    await asyncio.sleep(0.5 * attempt)
                    continue
                return response
            except httpx.TransportError as e:
                last_error = e
                if attempt <= self.max_retries:
                    logger.warning(
                        f"Qdrant {method} {path} transport error, "
                        f"retrying ({attempt}/{self.max_retries}): {e}"
                    )
                    await asyncio.sleep(0.5 * attempt)
                    continue
        raise QdrantError(f"Qdrant {method} {path} failed after retries: {last_error}")

    async def ensure_collection(self, vector_size: Optional[int] = None, distance: str = "Cosine") -> None:
        """
        Idempotent: creates the collection if it doesn't exist. If it does
        exist, verifies the vector size matches — a silent mismatch here
        produces a confusing "search always returns nothing" bug much later,
        so we fail loudly and immediately instead.
        """
        vector_size = vector_size or get_embedding_dimension()

        info_resp = await self._request("GET", f"/collections/{self.collection}")

        if info_resp.status_code == 200:
            existing = info_resp.json().get("result", {})
            vectors_config = existing.get("config", {}).get("params", {}).get("vectors", {})
            existing_size = vectors_config.get("size") if isinstance(vectors_config, dict) else None
            if existing_size and existing_size != vector_size:
                raise QdrantError(
                    f"Collection '{self.collection}' already exists with vector size "
                    f"{existing_size}, but the current embedding model produces size "
                    f"{vector_size}. Either delete the collection or point QDRANT_COLLECTION "
                    f"at a new name before re-indexing."
                )
            logger.info(f"Collection '{self.collection}' already exists (vector_size={existing_size})")
            return

        if info_resp.status_code != 404:
            raise QdrantError(
                f"Failed to check collection '{self.collection}': "
                f"{info_resp.status_code} - {info_resp.text}"
            )

        logger.info(f"Creating collection '{self.collection}' (vector_size={vector_size}, distance={distance})")
        create_resp = await self._request(
            "PUT",
            f"/collections/{self.collection}",
            json={"vectors": {"size": vector_size, "distance": distance}},
        )
        if create_resp.status_code not in (200, 201):
            raise QdrantError(
                f"Failed to create collection '{self.collection}': "
                f"{create_resp.status_code} - {create_resp.text}"
            )

    async def ping(self) -> bool:
        """Lightweight readiness probe: does the collection respond?"""
        try:
            response = await self._request("GET", f"/collections/{self.collection}")
            return response.status_code == 200
        except QdrantError:
            return False

    async def search(
        self,
        query: str,
        k: int = 20,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in Qdrant.

        Raises:
            QdrantError: if embedding the query or the Qdrant request fails.
        """
        start_time = time.perf_counter()

        try:
            vector = await embed_query(query)
        except EmbeddingError as e:
            raise QdrantError(f"Could not embed query for Qdrant search: {e}") from e

        payload: Dict[str, Any] = {
            "vector": vector,
            "limit": k,
            "with_payload": True,
            "with_vector": False,
        }
        if filter:
            payload["filter"] = filter

        response = await self._request("POST", f"/collections/{self.collection}/points/search", json=payload)
        if response.status_code != 200:
            raise QdrantError(f"Qdrant search failed: {response.status_code} - {response.text}")

        result = response.json().get("result", [])
        transformed = []
        for item in result:
            payload_data = item.get("payload", {}) or {}
            transformed.append({
                "passage": payload_data.get("passage", ""),
                "score": item.get("score", 0.0),
                "document_id": payload_data.get("chunk_id") or str(item.get("id", "")),
                "language": payload_data.get("language", "unknown"),
                "metadata": payload_data.get("metadata", {}),
            })

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Qdrant search completed in {latency_ms:.2f}ms, {len(transformed)} results")
        return transformed

    async def upsert(self, documents: List[Dict[str, Any]], batch_size: int = 128) -> None:
        """
        Upsert documents into Qdrant, chunked into batches — a single
        request with tens of thousands of points can time out or be
        rejected outright by Qdrant Cloud.

        Each document is expected to look like:
            {"id": str, "vector": List[float], "payload": Dict}
        """
        if not documents:
            return

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            points = []
            for i, doc in enumerate(batch):
                vector = doc.get("vector")
                if not vector:
                    logger.warning(f"Skipping document without a vector: id={doc.get('id')}")
                    continue
                original_id = doc.get("id", str(start + i))
                payload = dict(doc.get("payload", {}))
                payload.setdefault("chunk_id", original_id)
                points.append({
                    "id": _to_point_id(str(original_id)),
                    "vector": vector,
                    "payload": payload,
                })
            if not points:
                continue

            response = await self._request(
                "PUT", f"/collections/{self.collection}/points", json={"points": points}
            )
            if response.status_code not in (200, 201):
                raise QdrantError(f"Qdrant upsert batch failed: {response.status_code} - {response.text}")
            total += len(points)
            logger.debug(f"Upserted batch: {len(points)} points ({total}/{len(documents)} total)")

        logger.info(f"Upserted {total} documents to Qdrant collection '{self.collection}'")


# --- module-level singleton -------------------------------------------------
# pipeline/retrieval.py calls `qdrant_store.search(...)` as a plain module
# function (not a class method) — these top-level functions provide that
# interface, backed by one shared, connection-reusing QdrantStore instance
# for the process's lifetime.

_default_store: Optional[QdrantStore] = None


def get_store() -> QdrantStore:
    global _default_store
    if _default_store is None:
        _default_store = QdrantStore()
    return _default_store


async def search(query: str, k: int = 20, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
    return await get_store().search(query, k=k, filter=filter)


async def ping() -> bool:
    return await get_store().ping()


async def upsert(documents: List[Dict[str, Any]], batch_size: int = 128) -> None:
    await get_store().upsert(documents, batch_size=batch_size)


async def ensure_collection(vector_size: Optional[int] = None, distance: str = "Cosine") -> None:
    await get_store().ensure_collection(vector_size=vector_size, distance=distance)


async def close() -> None:
    global _default_store
    if _default_store is not None:
        await _default_store.close()
        _default_store = None
