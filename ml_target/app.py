import io
import json
import logging
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from ml_target.config import REQUEST_MODE, REQUEST_THREADS
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
async def _startup() -> None:
    _setup_logging()

    if REQUEST_MODE == "threadpool":
        # Bound the pool. Starlette's default is 40 threads; with one device
        # behind a global lock they would mostly queue on that lock while each
        # holds a decoded frame. Must run on the event loop — the limiter is an
        # anyio RunVar — which is why this handler is async.
        try:
            import anyio.to_thread
            anyio.to_thread.current_default_thread_limiter().total_tokens = REQUEST_THREADS
            LOG.info("request mode: threadpool, %d worker threads", REQUEST_THREADS)
        except Exception as exc:
            LOG.warning(
                "request mode: threadpool, but the pool could not be bounded (%s) — "
                "running with the framework default", exc,
            )
    else:
        LOG.info(
            "request mode: serial (one request at a time). "
            "Set REQUEST_MODE=threadpool to overlap CPU work with device time."
        )

    LOG.info("startup: init_pipeline()")
    init_pipeline()
    LOG.info("startup: OK")


@app.get("/")
def root():
    return {"message": "Immich ML"}


@app.get("/ping")
def ping():
    return PlainTextResponse("pong")


def _predict_impl(
    entries: str,
    img_bytes: Optional[bytes],
    text: Optional[str],
) -> JSONResponse:
    """The whole request. Runs on the event loop in serial mode, or in a worker
    thread in threadpool mode — identical either way.

    Reads the upload as bytes rather than an UploadFile so the two entry points
    below can each obtain them in the manner their execution context allows.
    """
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
        if img_bytes is not None:
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


# Two entry points, one implementation. FastAPI dispatches on the function type:
# `def` goes to the threadpool, `async def` runs on the event loop. Registering
# the right one at import time makes REQUEST_MODE=serial a genuine revert to the
# previously measured behaviour, not an approximation of it.
if REQUEST_MODE == "threadpool":

    @app.post("/predict")
    def predict(
        entries: str = Form(...),
        image: Optional[UploadFile] = File(None),
        text: Optional[str] = Form(None),
    ) -> JSONResponse:
        img_bytes: Optional[bytes] = None
        if image is not None:
            # UploadFile.read() is a coroutine and cannot be awaited here. The
            # underlying SpooledTemporaryFile is synchronous; seek first because
            # nothing guarantees the position after multipart parsing.
            image.file.seek(0)
            img_bytes = image.file.read()
        return _predict_impl(entries, img_bytes, text)

else:

    @app.post("/predict")
    async def predict(
        entries: str = Form(...),
        image: Optional[UploadFile] = File(None),
        text: Optional[str] = Form(None),
    ) -> JSONResponse:
        img_bytes: Optional[bytes] = None
        if image is not None:
            img_bytes = await image.read()
        return _predict_impl(entries, img_bytes, text)
