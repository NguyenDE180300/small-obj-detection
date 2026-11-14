"""FastAPI backend for the plain HTML/CSS/JavaScript inference studio."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
SOURCE_DIR = Path(os.environ.get("CUSTOM_CODE_ROOT", APP_DIR.parent / "Levir_ship_training_YOLO"))
sys.path.insert(0, str(SOURCE_DIR))

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", APP_DIR / "checkpoints")).resolve()
CALIBRATION_ROOT = Path(os.environ.get("CALIBRATION_ROOT", APP_DIR.parent / "processed_coco_exports")).resolve()
INT8_CACHE_ROOT = Path(os.environ.get("INT8_CACHE_ROOT", APP_DIR / "int8_cache")).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 32 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 100_000_000))
JETSON_BACKEND_URL = os.environ.get("JETSON_BACKEND_URL", "").rstrip("/")
SUPPORTED_PRECISIONS = tuple(
    item.strip().lower()
    for item in os.environ.get("SUPPORTED_PRECISIONS", "fp32,fp16,int8").split(",")
    if item.strip()
)

from catalog import load_manifest  # noqa: E402
from inference import annotate, png_bytes  # noqa: E402
from worker import InferenceWorker  # noqa: E402


ENTRIES = load_manifest(MODEL_ROOT)
ENTRY_BY_ID = {
    hashlib.sha256(entry["local_path"].encode("utf-8")).hexdigest()[:20]: entry
    for entry in ENTRIES
}
WORKER = InferenceWorker(calibration_root=CALIBRATION_ROOT, cache_root=INT8_CACHE_ROOT)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    WORKER._executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Inference Studio API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/assets/fpt-logo.webp", include_in_schema=False)
def fpt_logo() -> FileResponse:
    logo = APP_DIR.parent / "Logo_Trường_Đại_học_FPT.svg.webp"
    if not logo.is_file():
        raise HTTPException(status_code=404, detail="Logo asset is unavailable")
    return FileResponse(logo, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/health")
def health() -> dict:
    if JETSON_BACKEND_URL:
        try:
            import httpx

            response = httpx.get(f"{JETSON_BACKEND_URL}/api/health", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Jetson backend unavailable: {exc}") from exc
    return {"status": "ok", "checkpoints": len(ENTRIES), "cuda_available": torch.cuda.is_available()}


@app.get("/api/catalog")
def catalog() -> dict:
    if JETSON_BACKEND_URL:
        try:
            import httpx

            response = httpx.get(f"{JETSON_BACKEND_URL}/api/catalog", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not load Jetson model catalog: {exc}") from exc
    datasets = {
        "levir_ship": "LEVIR-Ship",
        "varroa": "Varroa",
        "tinyperson": "TinyPerson",
    }
    checkpoints = []
    for checkpoint_id, entry in ENTRY_BY_ID.items():
        checkpoints.append({
            "id": checkpoint_id,
            "dataset": entry["dataset"],
            "dataset_label": datasets.get(entry["dataset"], entry["dataset"]),
            "case": entry["case"],
            "run_tag": entry["run_tag"],
            "seed": entry["seed"],
            "family": entry["family"],
        })
    return {
        "checkpoints": checkpoints,
        "devices": ([{"id": "0", "label": torch.cuda.get_device_name(0)}] if torch.cuda.is_available() else [])
        + [{"id": "cpu", "label": "CPU"}],
        "precisions": [
            {"id": "fp32", "label": "FP32"},
            {"id": "fp16", "label": "FP16"},
            *([{"id": "int8", "label": "INT8 / FP32 hybrid (CPU · ONNX Runtime)"}]
              if "int8" in SUPPORTED_PRECISIONS else []),
        ],
    }


@app.post("/api/infer")
async def infer(
    image_file: UploadFile = File(...),
    checkpoint_id: str = Form(...),
    imgsz: int = Form(...),
    conf: float = Form(...),
    iou: float = Form(...),
    device: str = Form("cpu"),
    precision: str = Form("fp32"),
    tiled: bool = Form(False),
    tile_size: int = Form(640),
    tile_stride: int = Form(512),
) -> dict:
    if JETSON_BACKEND_URL:
        if not 256 <= imgsz <= 2048 or imgsz % 32:
            raise HTTPException(status_code=422, detail="Image size must be a multiple of 32 between 256 and 2048.")
        if not 0.01 <= conf <= 0.95 or not 0.10 <= iou <= 0.95:
            raise HTTPException(status_code=422, detail="Confidence or NMS IoU is outside the supported range.")
        if precision not in SUPPORTED_PRECISIONS:
            raise HTTPException(status_code=422, detail=f"Precision {precision} is unavailable on the active inference backend.")
        if not 256 <= tile_size <= 2048 or not 128 <= tile_stride <= tile_size:
            raise HTTPException(status_code=422, detail="Tile size or stride is invalid.")
        payload = await image_file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.")
        started = time.perf_counter()
        try:
            import httpx

            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{JETSON_BACKEND_URL}/api/infer",
                    data={
                        "checkpoint_id": checkpoint_id,
                        "imgsz": str(imgsz),
                        "conf": str(conf),
                        "iou": str(iou),
                        "device": device,
                        "precision": precision,
                        "tiled": str(tiled).lower(),
                        "tile_size": str(tile_size),
                        "tile_stride": str(tile_stride),
                    },
                    files={"image_file": (image_file.filename or "image", payload, image_file.content_type or "application/octet-stream")},
                )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Jetson inference request failed: {exc}") from exc
        try:
            result = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Jetson returned a non-JSON response.") from exc
        if response.is_error:
            raise HTTPException(status_code=response.status_code, detail=result.get("detail", "Jetson inference failed."))
        result.setdefault("metrics", {})["ui_gateway_roundtrip_ms"] = (time.perf_counter() - started) * 1000
        return result

    entry = ENTRY_BY_ID.get(checkpoint_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown checkpoint. Refresh the checkpoint list and try again.")
    if not 256 <= imgsz <= 2048 or imgsz % 32:
        raise HTTPException(status_code=422, detail="Image size must be a multiple of 32 between 256 and 2048.")
    if not 0.01 <= conf <= 0.95 or not 0.10 <= iou <= 0.95:
        raise HTTPException(status_code=422, detail="Confidence or NMS IoU is outside the supported range.")
    if precision not in {"fp32", "fp16", "int8"}:
        raise HTTPException(status_code=422, detail="Unsupported precision.")
    if device != "cpu" and (not torch.cuda.is_available() or device != "0"):
        raise HTTPException(status_code=422, detail="The selected GPU device is unavailable.")
    if precision == "fp16" and device == "cpu":
        raise HTTPException(status_code=422, detail="FP16 requires CUDA in this inference app.")
    if not 256 <= tile_size <= 2048 or not 128 <= tile_stride <= tile_size:
        raise HTTPException(status_code=422, detail="Tile size or stride is invalid.")

    decode_started = time.perf_counter()
    payload = image_file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.")
    try:
        image = Image.open(io.BytesIO(payload))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions exceed the configured pixel limit.")
        image.load()
        image = image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc
    input_decode_ms = (time.perf_counter() - decode_started) * 1000

    checkpoint = (MODEL_ROOT / entry["local_path"]).resolve()
    if not checkpoint.is_relative_to(MODEL_ROOT) or not checkpoint.is_file():
        raise HTTPException(status_code=404, detail="Checkpoint file is missing from the model directory.")

    started = time.perf_counter()
    try:
        detections, measured = WORKER.predict(
            checkpoint,
            image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            precision=precision,
            dataset=entry["dataset"],
        )
        annotation_started = time.perf_counter()
        annotated = png_bytes(annotate(image, detections))
        measured["annotation_ms"] = (time.perf_counter() - annotation_started) * 1000
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    measured["input_decode_ms"] = input_decode_ms
    measured["server_total_ms"] = input_decode_ms + (time.perf_counter() - started) * 1000
    return {
        "filename": image_file.filename or "image",
        "image_width": image.width,
        "image_height": image.height,
        "annotated_png": base64.b64encode(annotated).decode("ascii"),
        "detections": detections,
        "metrics": measured,
    }
