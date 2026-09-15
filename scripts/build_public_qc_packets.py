#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.tools.visual_qc import build_standalone_image_packet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render blinded QC packets from the public pilot catalog.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data/public_qc_sources/public_qc_pilot.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/visual_qc/public_abnormal_pilot",
    )
    args = parser.parse_args()

    with args.catalog.open(encoding="utf-8", newline="") as handle:
        catalog = list(csv.DictReader(handle))
    manifest: list[dict[str, object]] = []
    references: list[dict[str, str]] = []
    for item in catalog:
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        if not image_path.is_file():
            continue
        sample_id = f"{item['dataset']}::{item['sample_id']}"
        manifest.append(
            build_standalone_image_packet(
                sample_id,
                item["subject_id"],
                image_path,
                args.output_dir / "packets",
            )
        )
        references.append(
            {
                "sample_id": sample_id,
                "dataset": item["dataset"],
                "qc_label": item["qc_label"],
                "label_strength": item["label_strength"],
                "expert_score": item["expert_score"],
                "score_definition": item["score_definition"],
                "artifact_type": item["artifact_type"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    reference_path = args.output_dir / "reference_labels.jsonl"
    atomic_write_text(manifest_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    atomic_write_text(reference_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in references))
    print(json.dumps({"samples": len(manifest), "manifest": str(manifest_path), "references": str(reference_path)}, indent=2))


if __name__ == "__main__":
    main()
