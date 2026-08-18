"""
Torch-free query encoder: onnxruntime + tokenizers only.

Runs the same weights as retrieval/embeddings.py's LocalEmbeddingProvider
(sentence-transformers/SentenceTransformer), exported once offline via
scripts/export_onnx_embedder.py, through a much lighter runtime. Mirrors
the original model's pipeline exactly — mean pooling over token embeddings
(attention-mask weighted) then L2 normalization — see
data/onnx_embedder/../../1_Pooling/config.json (pooling_mode_mean_tokens)
and embeddings.py's model.encode(..., normalize_embeddings=True).

Not wired into the app yet — retrieval/embeddings.py's public interface is
unchanged until scripts/verify_onnx_embedder.py confirms this module's
output matches the original model's output.
"""

import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_session: Optional[ort.InferenceSession] = None
_tokenizer: Optional[Tokenizer] = None
_lock = threading.Lock()


class OnnxEmbedderError(Exception):
    pass


def _load(model_dir: str, max_seq_length: int) -> None:
    global _session, _tokenizer
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        model_path = Path(model_dir)
        tokenizer_path = model_path / "tokenizer.json"
        onnx_path = model_path / "model.onnx"
        if not tokenizer_path.exists() or not onnx_path.exists():
            raise OnnxEmbedderError(
                f"ONNX embedder files not found at {model_dir} — run "
                "scripts/export_onnx_embedder.py first."
            )

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=max_seq_length)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        # enable_cpu_mem_arena=False: measured (scripts/verify_onnx_embedder.py's
        # sibling memory probes) to give meaningfully lower and more stable RSS
        # than ORT's default arena allocator for this model/session, at the cost
        # of the arena's reuse-across-calls speed benefit — an acceptable trade
        # for a query-encoding workload that isn't latency-critical at the
        # microsecond level.
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session = ort.InferenceSession(
            str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        _tokenizer = tokenizer
        _session = session


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Attention-mask-weighted mean over the token axis — identical to
    sentence-transformers' Pooling module with pooling_mode_mean_tokens."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def preload(model_dir: str, max_seq_length: int) -> None:
    """Force the tokenizer/session to load now instead of on the first request."""
    _load(model_dir, max_seq_length)


def get_dimension(model_dir: str, max_seq_length: int) -> int:
    """Reads the real output dimension from the loaded ONNX session (last_hidden_state's
    hidden-size axis — pooling doesn't change this), rather than trusting a config value.

    Takes max_seq_length (rather than a placeholder) because _load() is an idempotent
    singleton — the first call to _load() from *any* function in this module wins for
    the lifetime of the process, so calling this with a throwaway value would silently
    and permanently truncate every real encode() call that follows.
    """
    _load(model_dir, max_seq_length)
    assert _session is not None
    hidden_size = _session.get_outputs()[0].shape[-1]
    if not isinstance(hidden_size, int):
        raise OnnxEmbedderError(
            f"ONNX model's output hidden size is not a fixed int (got {hidden_size!r}); "
            "cannot determine embedding dimension from the session alone."
        )
    return hidden_size


def _encode_batch(texts: List[str]) -> np.ndarray:
    assert _tokenizer is not None and _session is not None

    encodings = _tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

    try:
        outputs = _session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
    except Exception as e:
        raise OnnxEmbedderError(f"ONNX inference failed: {e}") from e

    last_hidden_state = outputs[0]
    pooled = _mean_pool(last_hidden_state, attention_mask)
    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), a_min=1e-9, a_max=None)
    return pooled / norms


def encode(
    texts: List[str],
    model_dir: str,
    max_seq_length: int,
    batch_size: int = 32,
) -> List[List[float]]:
    """Tokenize -> ONNX forward pass -> mean-pool -> L2-normalize. Preserves input order.

    Chunks into batch_size-sized groups rather than running the whole input through
    one forward pass — a single-shot batch scales activation memory linearly with
    input size (e.g. thousands of chunks during bulk ingestion), which has been
    observed to exhaust available memory and get the process silently killed by the
    OS (no Python exception, since it's an OOM kill, not a caught error). Mirrors the
    torch path (retrieval/embeddings.py's model.encode(..., batch_size=...)), which
    already batches internally.
    """
    if not texts:
        return []

    _load(model_dir, max_seq_length)

    pooled_batches = [_encode_batch(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    normalized = np.concatenate(pooled_batches, axis=0) if len(pooled_batches) > 1 else pooled_batches[0]
    return normalized.tolist()
