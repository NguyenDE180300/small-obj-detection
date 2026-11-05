#!/usr/bin/env python3
"""Export one packaged checkpoint to an isolated ONNX FP32/FP16 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("FP32", "FP16"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    staged_weights = args.output_dir / "best.pt"
    shutil.copy2(weights, staged_weights)

    import torch
    import onnxruntime as ort
    from ultralytics import YOLO
    from checkpoint_compat import patch_legacy_modules

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing the requested Jetson ONNX export device")
    model = YOLO(str(staged_weights), task="detect")
    patched_modules = patch_legacy_modules(model)
    if patched_modules:
        print(f"Applied inference compatibility defaults: {patched_modules}", flush=True)
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=False,
        half=args.precision == "FP16",
        device=args.device,
        simplify=False,
        opset=17,
        verbose=False,
    ))
    destination = args.output_dir / "model.onnx"
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)

    # Loading the graph is part of the export smoke test, but inference is
    # deliberately performed by the common two-image benchmark script.
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    report = {
        "weights_source": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "onnx": str(destination),
        "onnx_bytes": destination.stat().st_size,
        "precision_requested": args.precision,
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
        "inputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in session.get_inputs()],
        "outputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in session.get_outputs()],
        "image_size": args.imgsz,
        "batch_size": args.batch,
    }
    (args.output_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
