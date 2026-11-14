"""Build cached static-INT8 ONNX models calibrated on local dataset images."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def _torch_pytree_compat() -> None:
    """Bridge old PyTorch pytree APIs used by some installed Transformers versions."""
    import torch.utils._pytree as pytree

    if hasattr(pytree, "register_pytree_node"):
        return
    register = pytree._register_pytree_node

    def compatible(typ, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return register(typ, flatten_fn, unflatten_fn, **kwargs)

    pytree.register_pytree_node = compatible


def ensure_int8_onnx(
    checkpoint: Path,
    *,
    dataset: str,
    image_size: int,
    calibration_root: Path,
    cache_root: Path,
    load_model,
    training_info,
    profile_info,
    calibration_images: int = 300,
) -> tuple[Path, dict]:
    """Export and static-quantize one checkpoint; execute it with ONNX Runtime CPU EP.

    The calibration set is sampled deterministically from the selected dataset's seed-42
    training COCO manifest. Cached outputs are tied to checkpoint bytes, dataset and size.
    """
    checkpoint = Path(checkpoint).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
    cache_dir = cache_root / f"{checkpoint.stem}-{digest}-{dataset}-{int(image_size)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep this name/version separate from the earlier all-output-INT8 graph.
    # That graph quantized bbox coordinates and class probabilities together;
    # the large coordinate range rounded small class scores to zero.
    quantized_path = cache_dir / "model_int8_hybrid_v2.onnx"
    metadata_path = cache_dir / "metadata_int8_hybrid_v2.json"
    if quantized_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if "parameters" in metadata and "gflops" in metadata:
            return quantized_path, metadata
    else:
        metadata = {}

    def fill_model_metadata(existing: dict) -> dict:
        source_checkpoint = cache_dir / "source.pt"
        if not source_checkpoint.is_file():
            shutil.copy2(checkpoint, source_checkpoint)
        source_model = load_model(source_checkpoint)
        if "training_seconds" not in existing:
            existing.update(training_info(source_model))
        existing.update(profile_info(source_model, int(image_size)))
        del source_model
        return existing

    if quantized_path.is_file() and metadata_path.is_file():
        metadata = fill_model_metadata(metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return quantized_path, metadata

    coco_path = calibration_root / dataset / "annotations" / "seed_42" / "train.json"
    if not coco_path.is_file():
        raise FileNotFoundError(
            f"INT8 calibration COCO manifest not found: {coco_path}. "
            "Mount processed_coco_exports or configure CALIBRATION_ROOT."
        )
    coco = json.loads(coco_path.read_text())
    image_records = list(coco.get("images", []))
    if not image_records:
        raise ValueError(f"No calibration images in {coco_path}")
    annotated_ids = {
        annotation["image_id"]
        for annotation in coco.get("annotations", [])
        if annotation.get("area", 1) > 0 and not annotation.get("iscrowd", 0)
    }
    positives = [record for record in image_records if record["id"] in annotated_ids]
    negatives = [record for record in image_records if record["id"] not in annotated_ids]
    rng = random.Random(42)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    target = min(calibration_images, len(image_records))
    # Detection calibration needs both object and background activations. Sample
    # a balanced set so a negative-heavy split does not calibrate away ship scores.
    positive_count = min((target + 1) // 2, len(positives))
    negative_count = min(target - positive_count, len(negatives))
    if positive_count + negative_count < target:
        positive_count = min(target - negative_count, len(positives))
    records = positives[:positive_count] + negatives[:negative_count]
    rng.shuffle(records)

    fp32_path = cache_dir / "model.onnx"
    metadata = {}
    if not fp32_path.is_file():
        _torch_pytree_compat()
        # Keep the source checkpoint out of the read-only model mount.
        cached_checkpoint = cache_dir / "source.pt"
        if not cached_checkpoint.is_file():
            shutil.copy2(checkpoint, cached_checkpoint)
        model = load_model(cached_checkpoint)
        metadata = training_info(model)
        metadata.update(profile_info(model, int(image_size)))
        exported = model.export(
            format="onnx", imgsz=int(image_size), batch=1, device="cpu",
            half=False, int8=False, dynamic=False, simplify=False, opset=17,
            nms=False, verbose=False,
        )
        exported = Path(exported)
        if exported.resolve() != fp32_path.resolve():
            shutil.move(str(exported), str(fp32_path))
        del model
    elif metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())

    try:
        from ultralytics.data.augment import LetterBox
        import onnx
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise RuntimeError(
            "INT8 ONNX dependencies are missing. Install onnx and onnxruntime in the inference image."
        ) from exc

    class CocoCalibrationReader(CalibrationDataReader):
        def __init__(self):
            self.index = 0
            self.letterbox = LetterBox(new_shape=(int(image_size), int(image_size)), auto=False, stride=32)

        def get_next(self):
            while self.index < len(records):
                record = records[self.index]
                self.index += 1
                path = calibration_root / dataset / record["file_name"]
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image = self.letterbox(image=image)
                # Match Ultralytics predictor preprocessing: BGR -> RGB, BCHW, float in [0, 1].
                array = image[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                return {"images": np.ascontiguousarray(array)}
            return None

        def rewind(self):
            self.index = 0

    # Keep the final detection head in FP32. Its output concatenates large
    # pixel-space coordinates with much smaller probabilities, so quantizing
    # the shared tensor can erase confidence values. Quantize the backbone and
    # neck to unsigned INT8 activations / signed INT8 weights instead.
    graph = onnx.load(str(fp32_path))
    output_names = {output.name for output in graph.graph.output}
    output_producers = [
        node for node in graph.graph.node
        if any(name in output_names for name in node.output)
    ]
    head_prefixes = {
        match.group(0)
        for node in output_producers
        if (match := re.match(r"(/model\.\d+/)", node.name))
    }
    if not head_prefixes:
        raise RuntimeError(
            "Could not identify the terminal YOLO detection head in the exported ONNX graph."
        )
    nodes_to_exclude = [
        node.name for node in graph.graph.node
        if any(node.name.startswith(prefix) for prefix in head_prefixes)
    ]
    quantize_static(
        str(fp32_path), str(quantized_path), CocoCalibrationReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    metadata.update({
        "precision": "INT8 backbone/neck + FP32 detection head (hybrid)",
        "runtime": "ONNX Runtime CPUExecutionProvider",
        "calibration_dataset": dataset,
        "calibration_split": "train, seed 42",
        "calibration_images": len(records),
        "calibration_positive_images": positive_count,
        "calibration_negative_images": negative_count,
        "image_size": int(image_size),
        "quantization": "QDQ, QUInt8 activations, QInt8 weights, MinMax, per-channel; terminal detect head excluded",
        "excluded_head_prefixes": sorted(head_prefixes),
        "excluded_head_nodes": len(nodes_to_exclude),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return quantized_path, metadata
