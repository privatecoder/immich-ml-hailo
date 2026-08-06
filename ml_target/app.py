import io
import json
import logging
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from ml_target.pipeline import (
    BadRequest,
    init_pipeline,
    note,
    request_trace,
    run_inference,
    stage,
)

LOG = logging.getLogger("ml_target.app")

app = FastAPI()


def _setup_logging() -> None:
    """Configure logging. Level comes from LOG_LEVEL (default INFO).

    At INFO each request emits a single structured summary line. Set
    LOG_LEVEL=DEBUG for the per-stage detail — no rebuild needed, it is read
    from the container environment at startup.
    """
    requested = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    level = logging.getLevelName(requested)
    unknown = None
    if not isinstance(level, int):
        unknown, level = requested, logging.INFO

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    root.setLevel(level)

    # Child loggers (ml_target.pipeline, .models, .ocr, …) inherit from this one,
    # so there is no list of names here to fall out of date.
    logging.getLogger("ml_target").setLevel(level)

    if unknown:
        LOG.warning("LOG_LEVEL=%r not recognized — using INFO", unknown)


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
    # request_trace() emits exactly one INFO summary line on every exit path,
    # including the early 400s and the 500 below.
    with request_trace():
        LOG.debug("entries (raw) = %s", entries)

        try:
            parsed_entries = json.loads(entries)
        except Exception as e:
            LOG.warning("entries JSON parse failed: %s", e)
            note(status=400, error="entries-parse")
            return JSONResponse({"error": f"entries parse failed: {e}"}, status_code=400)

        LOG.debug("entries (parsed) = %s", parsed_entries)

        image_rgb: Optional[np.ndarray] = None
        if image is not None:
            img_bytes = await image.read()
            LOG.debug("image received: %d bytes", len(img_bytes))
            try:
                with stage("decode_image"):
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    image_rgb = np.asarray(pil, dtype=np.uint8)
            except Exception:
                LOG.exception("failed decoding image")
                note(status=400, error="image-decode")
                return JSONResponse({"error": "failed to decode image"}, status_code=400)

        try:
            response = run_inference(entries=parsed_entries, image_rgb=image_rgb, text=text)
        except BadRequest as e:
            # Malformed request rather than a worker fault — 400, not 500.
            LOG.warning("bad request: %s", e)
            note(status=400, error="bad-request")
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception:
            LOG.exception("Error during /predict")
            note(status=500, error="internal")
            return JSONResponse({"error": "internal error"}, status_code=500)

        LOG.debug("response top-level keys: %s", list(response.keys())
                  if isinstance(response, dict) else type(response).__name__)
        note(status=200)
        return JSONResponse(response, status_code=200)
