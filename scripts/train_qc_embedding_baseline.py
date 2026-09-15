from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.tools.visual_qc import embed_manifest

ROOT = Path(__file__).resolve().parents[1]


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "samples": len(labels),
        "threshold": 0.5,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental subject-grouped DINOv2 QC baseline.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/visual_qc/public_abnormal_pilot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/visual_qc/embedding_baseline")
    parser.add_argument("--model-id", default=str(ROOT / "data/models/facebook_dinov2-small"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    references = {
        row["sample_id"]: row
        for row in map(json.loads, (args.data_dir / "reference_labels.jsonl").read_text().splitlines())
        if row["label_strength"] == "expert"
    }
    records = [
        row for row in map(json.loads, (args.data_dir / "manifest.jsonl").read_text().splitlines())
        if row["sample_id"] in references
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expert_manifest = args.output_dir / "expert_manifest.jsonl"
    atomic_write_text(expert_manifest, "".join(json.dumps(row) + "\n" for row in records))
    embedded_manifest = args.output_dir / "embedded_manifest.jsonl"
    embed_manifest(expert_manifest, embedded_manifest, args.model_id, args.device)
    embedded = list(map(json.loads, embedded_manifest.read_text().splitlines()))
    features = np.stack([np.load(row["embedding_path"], allow_pickle=False) for row in embedded])
    labels = np.array([references[row["sample_id"]]["qc_label"] != "pass" for row in embedded], dtype=int)
    datasets = np.array([references[row["sample_id"]]["dataset"] for row in embedded])
    groups = np.array([f"{dataset}:{row['subject']}" for dataset, row in zip(datasets, embedded)])
    probabilities = np.full(len(labels), np.nan)
    folds = []
    for train, test in LeaveOneGroupOut().split(features, labels, groups):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        classifier.fit(features[train], labels[train])
        probabilities[test] = classifier.predict_proba(features[test])[:, 1]
        folds.append({"held_out_subject": str(groups[test][0]), "train_samples": len(train), "test_samples": len(test)})
    cross_dataset = {}
    for dataset in np.unique(datasets):
        test = datasets == dataset
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        classifier.fit(features[~test], labels[~test])
        cross_dataset[str(dataset)] = metrics(labels[test], classifier.predict_proba(features[test])[:, 1])
    final_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    final_model.fit(features, labels)
    joblib.dump(final_model, args.output_dir / "classifier.joblib")
    result = {
        "status": "experimental_shadow_only",
        "task": "pass vs abnormal (review/fail)",
        "samples": len(labels), "subjects": len(np.unique(groups)),
        "feature_dimension": features.shape[1], "embedding_model": args.model_id,
        "subject_leave_one_out": metrics(labels, probabilities),
        "cross_dataset_held_out": cross_dataset, "folds": folds,
        "note": "Fixed C=1 and threshold=0.5; scaling fit only on training folds. Final artifact is not a deployment gate.",
    }
    predictions = [
        {"sample_id": row["sample_id"], "group": str(group), "actual_abnormal": int(label),
         "oof_abnormal_probability": float(probability)}
        for row, group, label, probability in zip(embedded, groups, labels, probabilities)
    ]
    atomic_write_text(args.output_dir / "predictions.jsonl", "".join(json.dumps(row) + "\n" for row in predictions))
    atomic_write_text(args.output_dir / "metrics.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
