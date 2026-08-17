"""
One-time offline export: converts the configured EMBEDDING_MODEL to ONNX
(fp32, unquantized) so the runtime API can encode queries with
onnxruntime + tokenizers instead of torch + sentence-transformers +
transformers.

Deliberately NOT quantized: quantization changes the actual embedding
values, which would make query-time vectors numerically different from
the vectors already stored in Qdrant (embedded by the original fp32
PyTorch model) — see retrieval/embeddings.py's module docstring on why
index-time and query-time must match. An fp32 ONNX export runs the exact
same weights through a different (much lighter) runtime, so vectors stay
numerically equivalent (verified separately by scripts/verify_onnx_embedder.py,
not assumed).

This script is offline-only tooling — it imports optimum/torch, which are
NOT part of the runtime API's dependencies.

Usage:
    python scripts/export_onnx_embedder.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "onnx_embedder"


def main() -> None:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    model_name = settings.EMBEDDING_MODEL
    logger.info(f"Exporting '{model_name}' to ONNX (fp32) at {OUTPUT_DIR} ...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    model.save_pretrained(OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(OUTPUT_DIR)

    onnx_files = list(OUTPUT_DIR.glob("*.onnx"))
    logger.info(f"Exported: {[f.name for f in onnx_files]}")
    for f in onnx_files:
        logger.info(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    logger.info("Done. Run scripts/verify_onnx_embedder.py next to confirm numerical equivalence.")


if __name__ == "__main__":
    main()
