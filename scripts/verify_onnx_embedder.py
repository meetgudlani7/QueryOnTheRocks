"""
Verifies retrieval/onnx_embedder.py produces (near-)identical vectors to
the original sentence-transformers/torch model already used to build the
Qdrant index. This is the gate before wiring the ONNX path into the app —
see retrieval/embeddings.py's module docstring on why index-time and
query-time embeddings must match.

Compares on a real, varied sentence set (English + Hindi + short queries +
long passages + edge cases) using both cosine similarity (retrieval cares
about direction, not magnitude) and max absolute per-dimension difference.

Usage:
    python scripts/verify_onnx_embedder.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config import settings  # noqa: E402
from retrieval import onnx_embedder  # noqa: E402

logging.basicConfig(level=logging.WARNING)  # keep sentence-transformers/torch quiet

ONNX_DIR = str(Path(__file__).resolve().parent.parent / "data" / "onnx_embedder")

TEST_SENTENCES = [
    "Who discovered penicillin?",
    "Alexander Fleming discovered penicillin in 1928 when he noticed mould killing bacteria in a petri dish.",
    "पेनिसिलिन की खोज किसने की थी?",
    "What is the capital of France?",
    "",  # embeddings.py replaces true-empty with a single space before encoding
    " ",
    "A",
    "supercalifragilisticexpialidocious " * 50,  # long input, forces truncation at max_seq_length
    "Water boils at 100 degrees Celsius at sea level atmospheric pressure.",
    "मुंबई भारत का सबसे बड़ा शहर है और इसे देश की आर्थिक राजधानी माना जाता है।",
]


def main() -> int:
    print(f"Loading original SentenceTransformer('{settings.EMBEDDING_MODEL}') ...")
    from sentence_transformers import SentenceTransformer

    original_model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
    original_model.max_seq_length = settings.EMBEDDING_MAX_SEQ_LENGTH

    cleaned = [t if t.strip() else " " for t in TEST_SENTENCES]

    print("Encoding with original PyTorch model...")
    original_vectors = original_model.encode(
        cleaned, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )

    print("Encoding with ONNX model...")
    onnx_vectors = np.array(
        onnx_embedder.encode(cleaned, model_dir=ONNX_DIR, max_seq_length=settings.EMBEDDING_MAX_SEQ_LENGTH)
    )

    if original_vectors.shape != onnx_vectors.shape:
        print(f"FAIL: shape mismatch — original {original_vectors.shape} vs onnx {onnx_vectors.shape}")
        return 1

    cosine_sims = np.sum(original_vectors * onnx_vectors, axis=1)  # both already L2-normalized
    max_abs_diffs = np.max(np.abs(original_vectors - onnx_vectors), axis=1)
    onnx_norms = np.linalg.norm(onnx_vectors, axis=1)

    print(f"\n{'text':<60} {'cos_sim':>10} {'max_abs_diff':>14} {'onnx_norm':>10}")
    worst_cos_sim = 1.0
    worst_diff = 0.0
    for text, cs, mad, norm in zip(cleaned, cosine_sims, max_abs_diffs, onnx_norms):
        label = text[:57] + "..." if len(text) > 60 else text
        print(f"{label:<60} {cs:>10.6f} {mad:>14.6f} {norm:>10.6f}")
        worst_cos_sim = min(worst_cos_sim, cs)
        worst_diff = max(worst_diff, mad)

    print(f"\nWorst cosine similarity: {worst_cos_sim:.6f}")
    print(f"Worst max-abs-diff:      {worst_diff:.6f}")

    # Thresholds: fp32 ONNX vs fp32 PyTorch should agree to numerical-noise
    # level (float32 op-ordering differences), not approximation level.
    # 0.9999 cosine / 0.01 max-abs-diff is a real bar, not a rubber stamp.
    ok = worst_cos_sim >= 0.9999 and worst_diff <= 0.01
    print("\n" + ("PASS" if ok else "FAIL") + " — " + (
        "ONNX output is numerically equivalent to the original model."
        if ok else
        "ONNX output diverges too much from the original model — DO NOT wire this in yet."
    ))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
