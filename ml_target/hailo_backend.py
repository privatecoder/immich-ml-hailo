"""Backwards-compatibility shim — all code has moved to separate modules.

See: config.py, models.py, preprocessing.py, decoders.py, tokenizer.py, pipeline.py
"""

from ml_target.pipeline import init_pipeline, run_inference  # noqa: F401
