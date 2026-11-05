"""Benchmark one Jetson inference artifact and emit COCO-format predictions.

Run one process per backend/precision so peak RSS and resource load stay attributable.
The official AP metrics are computed later with pycocotools on the host.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import psutil
import torch
from PIL import Image
from ultralytics import YOLO
from checkpoint_compat import patch_legacy_modules


class ResourceMonitor:
    def __init__(self, sample_interval: float = 0.2):
        self.interval = sample_interval
        self.samples: list[dict] = []
        self.gpu_samples: list[int] = []
        self.ram_samples: list[int] = []
        self.stop_event = threading.Event()
        self.process = psutil.Process()
        self.tegrastats = None
        self.thread = None

    def _read_tegrastats(self):
        if self.tegrastats is None or self.tegrastats.stdout is None:
            return
        for line in self.tegrastats.stdout:
            if self.stop_event.is_set():
                break
            gpu = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            ram = re.search(r"RAM\s+(\d+)\s*/\s*(\d+)MB", line)
            if gpu:
                self.gpu_samples.append(int(gpu.group(1)))
            if ram:
                self.ram_samples.append(int(ram.group(1)))

    def _sample(self):
        while not self.stop_event.is_set():
            vm = psutil.virtual_memory()
            self.samples.append(
                {
                    "cpu_system_percent": psutil.cpu_percent(interval=None),
                    "cpu_process_percent": self.process.cpu_percent(interval=None),
                    "process_rss_mib": self.process.memory_info().rss / 2**20,
                    "system_ram_used_mib": vm.used / 2**20,
                }
            )
            self.stop_event.wait(self.interval)

    def start(self):
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        try:
            self.tegrastats = subprocess.Popen(
                ["tegrastats", "--interval", str(int(self.interval * 1000))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._read_tegrastats, daemon=True).start()
        except (FileNotFoundError, OSError):
            self.tegrastats = None
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.tegrastats:
            self.tegrastats.terminate()
            try:
                self.tegrastats.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.tegrastats.kill()
        def mean(key):
            values = [x[key] for x in self.samples]
            return statistics.fmean(values) if values else None
        def peak(key):
            values = [x[key] for x in self.samples]
            return max(values) if values else None
        return {
            "gpu_load_avg_percent": statistics.fmean(self.gpu_samples) if self.gpu_samples else None,
            "gpu_load_peak_percent": max(self.gpu_samples) if self.gpu_samples else None,
            "cpu_system_load_avg_percent": mean("cpu_system_percent"),
            "cpu_system_load_peak_percent": peak("cpu_system_percent"),
            "cpu_process_load_avg_percent": mean("cpu_process_percent"),
            "ram_process_avg_mib": mean("process_rss_mib"),
            "ram_process_peak_mib": peak("process_rss_mib"),
            "ram_system_used_avg_mib": mean("system_ram_used_mib"),
            "ram_system_used_peak_mib": peak("system_ram_used_mib"),
            "tegrastats_samples": len(self.gpu_samples),
            "psutil_samples": len(self.samples),
        }


def synchronize(device: str):
    if device == "0" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=0, help="Limit evaluation to the first N test images (0 = all).")
    parser.add_argument("--timing-samples", type=int, default=2, help="Cycle through this many images during latency measurement.")
    parser.add_argument("--timing-only", action="store_true", help="Skip full-dataset prediction; only measure warmup/timing latency.")
    parser.add_argument("--dataset", choices=("levir_ship", "varroa", "tinyperson"), default="levir_ship")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    data = json.loads(args.test_json.read_text())
    images = data["images"]
    if args.max_images > 0:
        images = images[: args.max_images]
        selected_ids = {int(record["id"]) for record in images}
        data["annotations"] = [
            annotation for annotation in data.get("annotations", [])
            if int(annotation["image_id"]) in selected_ids
        ]
    if not images:
        raise ValueError("The COCO test manifest has no images")
    if args.timing_samples < 1:
        raise ValueError("--timing-samples must be >= 1")
    timing_records = data["images"][: args.timing_samples]
    timing_images = []
    for record in timing_records:
        image_path = args.image_dir / record["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as im:
            timing_images.append(im.convert("RGB"))

    device = "cpu" if args.backend == "ONNX Runtime CPU" else "0"
    half = args.precision.upper() == "FP16"
    model = YOLO(str(args.model), task="detect")
    patched_modules = patch_legacy_modules(model)
    if patched_modules:
        print(f"Applied inference compatibility defaults: {patched_modules}", flush=True)
    monitor = ResourceMonitor()
    monitor.start()
    for index in range(args.warmup):
        model.predict(
            source=timing_images[index % len(timing_images)], imgsz=args.imgsz,
            conf=args.conf, iou=args.iou, max_det=300, device=device,
            half=half, verbose=False,
        )
    synchronize(device)
    # Keep model-only time separate from the full predict() wall time. Ultralytics
    # records result.speed['inference'] around the backend/model call, excluding
    # its image preprocessing and detection postprocessing stages.
    inference_times = []
    end_to_end_times = []
    for index in range(args.iterations):
        sample = timing_images[index % len(timing_images)]
        start = time.perf_counter()
        result = model.predict(
            source=sample, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=300, device=device, half=half, verbose=False,
        )[0]
        synchronize(device)
        end_to_end_times.append((time.perf_counter() - start) * 1000)
        inference_ms = (result.speed or {}).get("inference")
        if inference_ms is None or not math.isfinite(float(inference_ms)) or float(inference_ms) <= 0:
            raise RuntimeError("Ultralytics did not report a valid model-only inference time in result.speed['inference']")
        inference_times.append(float(inference_ms))

    detections = []
    eval_wall = 0.0
    if not args.timing_only:
        eval_start = time.perf_counter()
        for index, record in enumerate(images, start=1):
            image_path = args.image_dir / record["file_name"]
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                result = model.predict(
                    source=image, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                    max_det=300, device=device, half=half, verbose=False,
                )[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                scores = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(float, box)
                    detections.append(
                        {
                            "image_id": int(record["id"]),
                            "category_id": int(cls) + 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )
            if index % 100 == 0:
                print(f"Evaluated {index}/{len(images)} images", flush=True)
        eval_wall = time.perf_counter() - eval_start
    synchronize(device)
    resources = monitor.stop()
    summary = {
        "backend": args.backend,
        "precision": args.precision.upper(),
        "model": str(args.model),
        "dataset": args.dataset,
        "split": "test_seed_42",
        "images": 0 if args.timing_only else len(images),
        "instances": 0 if args.timing_only else len(data.get("annotations", [])),
        "timing_sample_images": len(timing_images),
        "image_size": args.imgsz,
        "batch_size": 1,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_scope": "model inference only (result.speed['inference']); excludes preprocessing, postprocessing, and file I/O",
        "latency_median_ms": statistics.median(inference_times),
        "latency_mean_ms": statistics.fmean(inference_times),
        "latency_p95_ms": float(__import__("numpy").percentile(inference_times, 95)),
        "fps_median": 1000.0 / statistics.median(inference_times),
        "fps_mean": 1000.0 / statistics.fmean(inference_times),
        "end_to_end_latency_median_ms": statistics.median(end_to_end_times),
        "end_to_end_latency_mean_ms": statistics.fmean(end_to_end_times),
        "end_to_end_fps_median": 1000.0 / statistics.median(end_to_end_times),
        "evaluation_wall_seconds": eval_wall,
        "prediction_count": len(detections),
        "device_name": torch.cuda.get_device_name(0) if device == "0" else "CPU",
        **resources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "predictions": detections}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
