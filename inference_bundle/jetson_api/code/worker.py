"""Persistent single-thread inference worker shared by concurrent API requests.

Keeping YOLO and its CUDA predictor on this worker avoids repeated CUDA/cuDNN
initialization and serializes access to the mutable Ultralytics predictor.
"""

from __future__ import annotations

import gc
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from inference import load_model
from int8_backend import ensure_int8_onnx
from metrics import model_profile, timed_prediction


class InferenceWorker:
    def __init__(self, *, calibration_root: Path, cache_root: Path):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-inference")
        self.calibration_root = Path(calibration_root)
        self.cache_root = Path(cache_root)
        self._model = None
        self._model_path = None
        self._precision = None
        self._profiles = {}
        self._training = {}

    @staticmethod
    def _training_info(model) -> dict:
        """Extract recorded offline training duration; never infer it from FPS."""
        checkpoint = model.ckpt or {}
        history = checkpoint.get("train_results") or {}
        times = history.get("time")
        epochs = history.get("epoch")
        try:
            seconds = float(times[-1]) if times is not None else None
            if seconds is None:
                raise ValueError("missing training time")
            if not math.isfinite(seconds) or seconds < 0:
                seconds = None
        except (IndexError, TypeError, ValueError):
            seconds = None
        return {
            "training_seconds": seconds,
            "training_epochs_recorded": len(epochs) if epochs is not None and len(epochs) else None,
        }

    def predict(self, checkpoint: Path, image, *, imgsz: int, conf: float, iou: float,
                device: str, tiled: bool, tile_size: int, tile_stride: int,
                precision: str = "fp32", dataset: str | None = None):
        """Run all model/CUDA operations on the same long-lived thread."""
        return self._executor.submit(
            self._predict, checkpoint, image, imgsz=imgsz, conf=conf, iou=iou,
            device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            precision=precision, dataset=dataset,
        ).result()

    def _predict(self, checkpoint: Path, image, **kwargs):
        worker_start = time.perf_counter()
        model_load_ms = 0.0
        profile_ms = 0.0
        checkpoint = Path(checkpoint).resolve()
        precision = kwargs.pop("precision", "fp32").lower()
        dataset = kwargs.pop("dataset", None)
        training = None
        if precision == "int8":
            if dataset not in {"levir_ship", "varroa", "tinyperson"}:
                raise ValueError(f"Dataset calibration không hợp lệ: {dataset}")
            checkpoint, training = ensure_int8_onnx(
                checkpoint,
                dataset=dataset,
                image_size=int(kwargs["imgsz"]),
                calibration_root=self.calibration_root,
                cache_root=self.cache_root,
                load_model=load_model,
                training_info=self._training_info,
                profile_info=model_profile,
            )
            kwargs["device"] = "cpu"
        if precision not in {"fp32", "fp16", "int8"}:
            raise ValueError(f"Precision không hỗ trợ: {precision}")
        if precision == "fp16" and kwargs["device"] == "cpu":
            raise RuntimeError("FP16 inference trong giao diện này cần CUDA; hãy chọn GPU hoặc FP32.")
        checkpoint_str = str(checkpoint)
        if checkpoint_str != self._model_path or precision != self._precision:
            load_start = time.perf_counter()
            old_model = self._model
            self._model = None
            self._model_path = None
            self._precision = None
            self._profiles.clear()
            del old_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_model(checkpoint)
            self._model = model
            self._model_path = checkpoint_str
            self._precision = precision
            self._training = training if training is not None else self._training_info(model)
            model_load_ms = (time.perf_counter() - load_start) * 1000

        imgsz = kwargs["imgsz"]
        profile_key = (imgsz, precision)
        if profile_key not in self._profiles:
            profile_start = time.perf_counter()
            self._profiles[profile_key] = model_profile(self._model, imgsz)
            profile_ms = (time.perf_counter() - profile_start) * 1000
        kwargs["half"] = precision == "fp16"
        detections, runtime = timed_prediction(self._model, image, **kwargs)
        precision_label = (
            self._training.get("precision", "INT8/FP32 hybrid")
            if precision == "int8" else precision.upper()
        )
        return detections, {
            **self._profiles[profile_key], **self._training, **runtime,
            "precision": precision_label,
            "model_load_ms": model_load_ms,
            "profile_ms": profile_ms,
            "worker_total_ms": (time.perf_counter() - worker_start) * 1000,
        }
