# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lazy model exports.

Keeping these imports lazy lets a detection-only training job load ``YOLO``
without importing optional SAM stacks (and their unrelated dependencies).
"""
from __future__ import annotations

from importlib import import_module

__all__ = "NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld"

_EXPORTS = {
    "NAS": (".nas", "NAS"),
    "RTDETR": (".rtdetr", "RTDETR"),
    "SAM": (".sam", "SAM"),
    "FastSAM": (".fastsam", "FastSAM"),
    "YOLO": (".yolo", "YOLO"),
    "YOLOE": (".yolo", "YOLOE"),
    "YOLOWorld": (".yolo", "YOLOWorld"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), symbol)
