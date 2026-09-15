#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from neuro_preprocess_agent.io_utils import atomic_write_text

CLASSES = ("pass", "review", "fail")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(rows: list[tuple[str, str]]) -> dict[str, Any]:
    confusion = {actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES}
    for actual, predicted in rows:
        confusion[actual][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in CLASSES:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in CLASSES if actual != label)
        fn = sum(confusion[label][predicted] for predicted in CLASSES if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        }

    binary_tp = sum(actual != "pass" and predicted != "pass" for actual, predicted in rows)
    binary_tn = sum(actual == "pass" and predicted == "pass" for actual, predicted in rows)
    binary_fp = sum(actual == "pass" and predicted != "pass" for actual, predicted in rows)
    binary_fn = sum(actual != "pass" and predicted == "pass" for actual, predicted in rows)
    sensitivity = binary_tp / (binary_tp + binary_fn) if binary_tp + binary_fn else 0.0
    specificity = binary_tn / (binary_tn + binary_fp) if binary_tn + binary_fp else 0.0
    return {
        "samples": len(rows),
        "accuracy": round(sum(actual == predicted for actual, predicted in rows) / len(rows), 4),
        "macro_f1": round(sum(float(per_class[label]["f1"]) for label in CLASSES) / len(CLASSES), 4),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "binary_abnormal_detection": {
            "definition": "review/fail=abnormal; pass=normal",
            "tp": binary_tp,
            "tn": binary_tn,
            "fp": binary_fp,
            "fn": binary_fn,
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "balanced_accuracy": round((sensitivity + specificity) / 2, 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score VLM decisions against public MRI QC references.")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path)
    args = parser.parse_args()

    references = {row["sample_id"]: row for row in _read_jsonl(args.references)}
    predictions = [row for path in args.predictions for row in _read_jsonl(path)]
    if args.merged_output:
        atomic_write_text(
            args.merged_output,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        )

    unavailable = [row for row in predictions if row.get("status") != "completed"]
    expert_pairs: list[tuple[str, str]] = []
    expert_by_dataset: dict[str, list[tuple[str, str]]] = {}
    weak_predictions: Counter[str] = Counter()
    missing_references: list[str] = []
    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        reference = references.get(sample_id)
        if reference is None:
            missing_references.append(sample_id)
            continue
        if prediction.get("status") != "completed":
            continue
        predicted = str(prediction["decision"])
        actual = "review" if reference["qc_label"] == "needs_review" else reference["qc_label"]
        if reference["label_strength"] == "expert":
            pair = (actual, predicted)
            expert_pairs.append(pair)
            expert_by_dataset.setdefault(reference["dataset"], []).append(pair)
        else:
            weak_predictions[predicted] += 1

    result = {
        "prediction_samples": len(predictions),
        "completed": len(predictions) - len(unavailable),
        "unavailable": len(unavailable),
        "missing_references": missing_references,
        "expert": _metrics(expert_pairs),
        "expert_by_dataset": {name: _metrics(rows) for name, rows in expert_by_dataset.items()},
        "weak_label_pool": {
            "samples": sum(weak_predictions.values()),
            "note": "ABIDE rater-disagreement cases are excluded from accuracy metrics",
            "prediction_distribution": dict(weak_predictions),
        },
    }
    atomic_write_text(args.output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
