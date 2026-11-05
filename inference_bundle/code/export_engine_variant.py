"""Build a precision-specific TensorRT engine in a separate output directory.

Use this for controlled batch-1 timing experiments without overwriting the
dynamic-batch engine used by the Jetson UI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("FP32", "FP16", "INT8"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=float, default=2.0)
    parser.add_argument("--data", type=Path, help="Calibration dataset YAML (required for INT8)")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    # Share TensorRT tactic timings between all per-checkpoint builds in this
    # Jetson benchmark bundle. The cache is device/runtime-specific and stays
    # on the Jetson alongside the generated engines.
    timing_cache_path = os.environ.get("ULTRALYTICS_TRT_TIMING_CACHE")
    if not timing_cache_path:
        bundle_root = Path(__file__).resolve().parents[1]
        timing_cache_path = bundle_root / "engines/full_seed42_7mode_latency/shared_timing_cache.bin"
    os.environ["ULTRALYTICS_TRT_TIMING_CACHE"] = str(timing_cache_path)
    print(f"TensorRT shared timing cache: {timing_cache_path}", flush=True)

    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if args.precision == "INT8" and args.data is None:
        parser.error("--data is required for INT8 calibration")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    staged_weights = args.output_dir / "best.pt"
    shutil.copy2(weights, staged_weights)

    import torch
    import tensorrt as trt
    from ultralytics import YOLO
    from checkpoint_compat import patch_legacy_modules

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to build a Jetson TensorRT engine")

    half = args.precision == "FP16"
    int8 = args.precision == "INT8"
    model = YOLO(str(staged_weights), task="detect")
    patched_modules = patch_legacy_modules(model)
    if patched_modules:
        print(f"Applied inference compatibility defaults: {patched_modules}", flush=True)
    export_started = time.perf_counter()
    engine_path = Path(
        model.export(
            format="engine",
            imgsz=args.imgsz,
            batch=args.batch,
            dynamic=args.dynamic,
            half=half,
            int8=int8,
            data=str(args.data.resolve()) if args.data else None,
            device=0,
            workspace=args.workspace,
            simplify=False,
            nms=False,
            verbose=False,
        )
    )
    print(
        f"TensorRT {args.precision} export+build elapsed: {time.perf_counter() - export_started:.1f}s",
        flush=True,
    )

    raw = engine_path.read_bytes()
    header_size = int.from_bytes(raw[:4], byteorder="little")
    metadata = json.loads(raw[4 : 4 + header_size].decode("utf-8"))
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(raw[4 + header_size :])
    if engine is None:
        raise RuntimeError(f"TensorRT {trt.__version__} cannot load its freshly built engine")

    input_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    )
    profiles = [
        [[int(dim) for dim in shape] for shape in engine.get_tensor_profile_shape(input_name, i)]
        for i in range(engine.num_optimization_profiles)
    ]
    report = {
        "weights_source": str(weights),
        "weights_sha256": __import__("hashlib").sha256(weights.read_bytes()).hexdigest(),
        "engine": str(engine_path),
        "precision_requested": args.precision,
        "tensorrt": trt.__version__,
        "device": torch.cuda.get_device_name(0),
        "metadata": metadata,
        "input_name": input_name,
        "input_dtype": str(engine.get_tensor_dtype(input_name)),
        "input_profiles_min_opt_max": profiles,
    }
    (args.output_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
