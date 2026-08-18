"""
One-time offline step: dynamic int8 quantization of the fp32 ONNX embedder
(scripts/export_onnx_embedder.py's output) into a separate directory.

Kept separate from data/onnx_embedder/ (fp32) rather than overwriting it —
that export is reproducible and already numerically verified against the
original PyTorch model (scripts/verify_onnx_embedder.py), so there's no
reason to destroy it. Quantization trades a small amount of embedding
precision for a much smaller model file (int8 weights vs float32), needed
to fit the whole API process inside a small memory budget (e.g. Render's
512MB free tier) — see scripts/verify_onnx_embedder.py's sibling check for
how much precision is actually lost.

IMPORTANT: because quantization changes the embedding values, anything
indexed with these quantized vectors must be queried with the same
quantized model, and vice versa — mixing fp32-indexed vectors with
quantized-query vectors (or the reverse) silently degrades retrieval. Any
Qdrant collection built for this model must be re-ingested with
EMBEDDING_ONNX_DIR pointing at this quantized directory.

Usage:
    python scripts/quantize_onnx_embedder.py
"""

import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCE_DIR = Path(__file__).resolve().parent.parent / "data" / "onnx_embedder"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "onnx_embedder_quantized"

# Non-.onnx files needed alongside the model (tokenizer + config) — these
# aren't touched by quantization, just copied through as-is.
SIDECAR_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]


def main() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source_model = SOURCE_DIR / "model.onnx"
    if not source_model.exists():
        raise SystemExit(f"No fp32 model found at {source_model} — run scripts/export_onnx_embedder.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_model = OUTPUT_DIR / "model.onnx"

    logger.info(f"Quantizing {source_model} -> {target_model} (dynamic int8)...")
    quantize_dynamic(
        model_input=str(source_model),
        model_output=str(target_model),
        weight_type=QuantType.QInt8,
    )

    for name in SIDECAR_FILES:
        src = SOURCE_DIR / name
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / name)

    before_mb = source_model.stat().st_size / 1e6
    after_mb = target_model.stat().st_size / 1e6
    logger.info(f"fp32 model:      {before_mb:.1f} MB")
    logger.info(f"quantized model: {after_mb:.1f} MB ({after_mb / before_mb * 100:.1f}% of original)")
    logger.info("Done. Run scripts/verify_onnx_embedder.py --quantized next to check embedding drift.")


if __name__ == "__main__":
    main()
