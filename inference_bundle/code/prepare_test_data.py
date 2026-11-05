#!/usr/bin/env python3
"""Extract only seed-42 test images/annotations from the bundled COCO archives."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


DATASETS = ("levir_ship", "varroa", "tinyperson")


def extract_member(zf: zipfile.ZipFile, member: str, destination: Path) -> None:
    try:
        info = zf.getinfo(member)
    except KeyError as exc:
        raise FileNotFoundError(f"Dataset archive is missing {member}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as source, destination.open("wb") as target:
        target.write(source.read())


def prepare_one(archive: Path, output: Path, dataset: str) -> int:
    with zipfile.ZipFile(archive) as zf:
        rel_json = "annotations/seed_42/test.json"
        test_data = json.loads(zf.read(rel_json))
        extract_member(zf, rel_json, output / rel_json)
        image_names = sorted({str(record["file_name"]) for record in test_data["images"]})
        for index, name in enumerate(image_names, 1):
            extract_member(zf, name, output / name)
            if index % 1000 == 0:
                print(f"{dataset}: extracted {index}/{len(image_names)} test images", flush=True)
    return len(image_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    args = parser.parse_args()
    args.archives, args.output = args.archives.resolve(), args.output.resolve()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    summary = {}
    for dataset in selected:
        archive = args.archives / f"{dataset}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        count = prepare_one(archive, args.output / dataset, dataset)
        summary[dataset] = {"seed": 42, "test_images": count}

    metadata = args.archives / "tinyperson_eval_metadata.zip"
    if "tinyperson" in selected:
        with zipfile.ZipFile(metadata) as zf:
            for name in (
                "annotations/corner/task/tiny_set_test_sw640_sh512_all.json",
                "annotations/task/tiny_set_test_all.json",
                "tinybenchmark/test_windows.json",
            ):
                extract_member(zf, name, args.output / "tinyperson" / name)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prepared_seed42.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
