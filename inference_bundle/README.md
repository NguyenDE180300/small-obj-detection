# Inference benchmark source

This directory contains the source scripts for the seed-42 Jetson benchmark.
The benchmark's primary latency/FPS is **model inference only**, taken from
Ultralytics `result.speed["inference"]`; it excludes preprocessing,
postprocessing/NMS, file I/O, and model loading. Warmup calls are discarded.
End-to-end `model.predict()` latency is recorded separately for reference.

The scripts intentionally do not include checkpoints, datasets, exported
ONNX/TensorRT engines, or benchmark outputs. Those artifacts must be prepared
separately using the manifest and data layout expected by the runner. The
`code/ultralytics` link points to this repository's Ultralytics source tree.

Run the matrix scripts from a prepared benchmark bundle, not directly from a
fresh source checkout without the required data/checkpoint artifacts.
