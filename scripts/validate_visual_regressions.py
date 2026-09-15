#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neuro_preprocess_agent.config import PROJECT_ROOT
from neuro_preprocess_agent.io_utils import atomic_write_text


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _subject_metrics(report_path: Path, subject: str) -> dict[str, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    subject_qc = next(
        item for item in report["qc_result"]["subject_qc"] if str(item["subject"]) == subject
    )
    images = subject_qc["quantitative_metrics"]["images"]
    return {
        "anat_bold_mask_dice": float(images["anat_bold_mask_overlap"]["dice"]),
        "native_t1_mask_voxels": float(images["t1w_brain_mask"]["mask_voxels"]),
    }


def evaluate_manifest(manifest_path: Path, labels_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest_reviews: dict[str, dict[str, Any]] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            review = json.loads(line)
            latest_reviews[str(_path(review["packet_path"]).resolve())] = review

    case_results = []
    for case in manifest["cases"]:
        checks = []
        variants: dict[str, dict[str, float]] = {}
        for name in ("failing", "corrected"):
            expected = case[name]
            report_path = _path(expected["report"])
            packet_path = _path(expected["packet"])
            metrics = _subject_metrics(report_path, str(case["subject"]))
            variants[name] = metrics
            review = latest_reviews.get(str(packet_path.resolve()))
            checks.extend([
                {"name": f"{name}_report_exists", "passed": report_path.is_file()},
                {"name": f"{name}_packet_exists", "passed": packet_path.is_file()},
                {
                    "name": f"{name}_human_decision",
                    "passed": bool(review and review.get("decision") == expected["human_decision"]),
                    "actual": review.get("decision") if review else None,
                    "expected": expected["human_decision"],
                },
                {
                    "name": f"{name}_input_state",
                    "passed": bool(review and review.get("anatomical_input_state") == case["anatomical_input_state"]),
                    "actual": review.get("anatomical_input_state") if review else None,
                    "expected": case["anatomical_input_state"],
                },
            ])
            for key, value in expected.items():
                if key.startswith("min_") and key.removeprefix("min_") in metrics:
                    metric = key.removeprefix("min_")
                    checks.append({"name": f"{name}_{key}", "passed": metrics[metric] >= value, "actual": metrics[metric], "expected": value})
                if key.startswith("max_") and key.removeprefix("max_") in metrics:
                    metric = key.removeprefix("max_")
                    checks.append({"name": f"{name}_{key}", "passed": metrics[metric] <= value, "actual": metrics[metric], "expected": value})

        dice_gain = variants["corrected"]["anat_bold_mask_dice"] - variants["failing"]["anat_bold_mask_dice"]
        mask_ratio = variants["corrected"]["native_t1_mask_voxels"] / variants["failing"]["native_t1_mask_voxels"]
        checks.extend([
            {"name": "min_dice_gain", "passed": dice_gain >= case["min_dice_gain"], "actual": dice_gain, "expected": case["min_dice_gain"]},
            {"name": "min_native_t1_mask_ratio", "passed": mask_ratio >= case["min_native_t1_mask_ratio"], "actual": mask_ratio, "expected": case["min_native_t1_mask_ratio"]},
        ])
        case_results.append({
            "case_id": case["case_id"],
            "passed": all(item["passed"] for item in checks),
            "metrics": {
                "failing": variants["failing"],
                "corrected": variants["corrected"],
                "dice_gain": dice_gain,
                "native_t1_mask_ratio": mask_ratio,
            },
            "checks": checks,
        })
    return {"schema_version": "1.0", "passed": all(item["passed"] for item in case_results), "cases": case_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate human-reviewed paired visual QC regressions")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "evals/manifests/visual_qc_golden_pairs.json")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data/visual_qc/labels.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evals/reports-real/visual_regression_latest.json")
    args = parser.parse_args()
    result = evaluate_manifest(args.manifest, args.labels)
    atomic_write_text(args.output, json.dumps(result, ensure_ascii=False, indent=2))
    for case in result["cases"]:
        print(f"{'PASS' if case['passed'] else 'FAIL'} {case['case_id']}")
    print(f"report={args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
