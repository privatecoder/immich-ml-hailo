import io
import json
import logging
import time
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from ml_target.pipeline import init_pipeline, run_inference

LOG = logging.getLogger("ml_target.app")

app = FastAPI()


def _setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for name in ("ml_target", "ml_target.pipeline", "ml_target.models",
                  "ml_target.decoders", "ml_target.preprocessing",
                  "ml_target.ocr", "ml_target.app"):
        logging.getLogger(name).setLevel(logging.INFO)


@app.on_event("startup")
def _startup() -> None:
    _setup_logging()
    LOG.info("startup: init_pipeline()")
    init_pipeline()
    LOG.info("startup: OK")

@app.get("/")
def root():
    return {"message": "Immich ML"}


@app.get("/ping")
def ping():
    return PlainTextResponse("pong")


@app.post("/predict")
async def predict(
    entries: str = Form(...),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
) -> JSONResponse:
    t0 = time.time()
    LOG.info("=== /predict called ===")
    LOG.info("Content-Type: multipart/form-data")
    LOG.info("entries (raw) = %s", entries)

    try:
        parsed_entries = json.loads(entries)
    except Exception as e:
        LOG.warning("entries JSON parse failed: %s", e)
        return JSONResponse({"error": f"entries parse failed: {e}"}, status_code=400)

    LOG.info("entries (parsed) = %s", parsed_entries)

    image_rgb: Optional[np.ndarray] = None
    if image is not None:
        img_bytes = await image.read()
        LOG.info("image received: %d bytes", len(img_bytes))
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_rgb = np.asarray(pil, dtype=np.uint8)
        except Exception:
            LOG.exception("failed decoding image")
            return JSONResponse({"error": "failed to decode image"}, status_code=400)

    try:
        response = run_inference(entries=parsed_entries, image_rgb=image_rgb, text=text)
        dt = (time.time() - t0) * 1000.0

        if isinstance(response, dict):
            LOG.info("response top-level keys: %s", list(response.keys()))
            fr = response.get("facial-recognition")
            if isinstance(fr, dict):
                dets = fr.get("detections") or []
                embs = fr.get("embeddings") or []
                LOG.info("facial-recognition: detections=%d embeddings=%d", len(dets), len(embs))

        LOG.info("/predict OK (%.2f ms)", dt)
        return JSONResponse(response, status_code=200)
    except Exception:
        LOG.exception("Error during /predict")
        return JSONResponse({"error": "internal error"}, status_code=500)
