"""Model loading and image/tiled inference for the checkpoint browser UI."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_model(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _nms(detections: list[dict], iou: float, max_det: int) -> list[dict]:
    if not detections:
        return []
    boxes = np.asarray([d["xyxy"] for d in detections], dtype=np.float32)
    scores = np.asarray([d["confidence"] for d in detections], dtype=np.float32)
    classes = np.asarray([d["class_id"] for d in detections])
    chosen = []
    for class_id in np.unique(classes):
        order = np.where(classes == class_id)[0]
        order = order[np.argsort(scores[order])[::-1]]
        while len(order):
            current = int(order[0])
            chosen.append(current)
            if len(order) == 1:
                break
            rest = order[1:]
            x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
            area_a = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
            area_b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            overlap = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
            order = rest[overlap <= iou]
    chosen = sorted(chosen, key=lambda index: scores[index], reverse=True)[:max_det]
    return [detections[index] for index in chosen]


def predict_image(
    model, image: Image.Image, *, imgsz: int, conf: float, iou: float,
    device: str, tiled: bool, tile_size: int = 640, tile_stride: int = 512,
    max_det: int = 1000, half: bool = False, timing_breakdown: dict | None = None,
) -> list[dict]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    tiles = [(0, 0, image)] if not tiled else [
        (left, top, image.crop((left, top, min(left + tile_size, width), min(top + tile_size, height))))
        for top in tile_positions(height, tile_size, tile_stride)
        for left in tile_positions(width, tile_size, tile_stride)
    ]
    detections = []
    for left, top, tile in tiles:
        result = model.predict(
            source=tile, imgsz=imgsz, conf=conf, iou=iou,
            device=device, max_det=max_det, half=half, verbose=False,
        )[0]
        if timing_breakdown is not None:
            for source_key, target_key in (
                ("preprocess", "preprocess_ms"),
                ("inference", "model_forward_ms"),
                ("postprocess", "postprocess_ms"),
            ):
                timing_breakdown[target_key] += float((result.speed or {}).get(source_key, 0.0))
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().tolist()
            cls = int(box.cls.item())
            detections.append({
                "class_id": cls,
                "class_name": str(result.names[cls]),
                "confidence": float(box.conf.item()),
                "xyxy": [
                    max(0.0, min(width, xyxy[0] + left)),
                    max(0.0, min(height, xyxy[1] + top)),
                    max(0.0, min(width, xyxy[2] + left)),
                    max(0.0, min(height, xyxy[3] + top)),
                ],
            })
    return _nms(detections, iou, max_det) if tiled else detections


def annotate(image: Image.Image, detections: list[dict]) -> Image.Image:
    output = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(output)
    line_width = max(2, round(min(output.size) / 400))
    for detection in detections:
        box = detection["xyxy"]
        draw.rectangle(box, outline="#ff3b30", width=line_width)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(text_box, fill="#ff3b30")
        draw.text((box[0], box[1]), label, fill="white")
    return output


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
