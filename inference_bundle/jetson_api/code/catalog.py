"""Inventory locally downloaded inference checkpoints and their HF sources."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi


REPOS = {
    "levir_ship": "HoangTrungNguyen/Levir_ship_training",
    "varroa": "HoangTrungNguyen/Varroa_ASRM_training",
    "tinyperson": "HoangTrungNguyen/TinyPerson_training",
}


def metadata(dataset: str, remote_path: str, size: int) -> dict:
    parts = Path(remote_path).parts
    seed_index = next((i for i, part in enumerate(parts) if part.startswith("seed_")), -1)
    seed = parts[seed_index] if seed_index >= 0 else "unknown"
    case_index = seed_index - 1
    if case_index >= 0 and parts[case_index] in REPOS:
        case_index -= 1
    case = parts[case_index] if case_index >= 0 else "unknown"
    run_tag = parts[case_index - 1] if case_index >= 1 else "unknown"
    family = "YOLOv11n" if "yolov11n" in case else "YOLOv9t" if "yolov9t" in case else "YOLOv8n/custom"
    return {
        "dataset": dataset,
        "repo": REPOS[dataset],
        "remote_path": remote_path,
        "local_path": f"{dataset}/{remote_path}",
        "case": case,
        "run_tag": run_tag,
        "seed": seed,
        "family": family,
        "size_bytes": size,
    }


def remote_catalog(api: HfApi | None = None) -> list[dict]:
    api = api or HfApi()
    entries = []
    for dataset, repo in REPOS.items():
        for item in api.list_repo_tree(repo, repo_type="dataset", recursive=True):
            if item.path.endswith("/best.pt"):
                entries.append(metadata(dataset, item.path, item.size or 0))
    return sorted(entries, key=lambda item: (item["dataset"], item["remote_path"]))


def check_local(root: Path, entries: list[dict]) -> tuple[list[dict], list[dict]]:
    present, missing = [], []
    for entry in entries:
        path = root / entry["local_path"]
        if path.is_file() and path.stat().st_size == entry["size_bytes"]:
            present.append(entry)
        else:
            missing.append(entry)
    return present, missing


def save_manifest(root: Path, entries: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "manifest.json"
    destination.write_text(json.dumps({"schema": 1, "checkpoints": entries}, ensure_ascii=False, indent=2) + "\n")
    return destination


def load_manifest(root: Path) -> list[dict]:
    path = root / "manifest.json"
    if not path.is_file():
        return []
    entries = json.loads(path.read_text())["checkpoints"]
    # Never expose an arbitrary path from a modified manifest to the loader.
    valid = []
    for entry in entries:
        relative = Path(entry["local_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.name != "best.pt":
            continue
        if (root / relative).is_file():
            valid.append(entry)
    return valid
