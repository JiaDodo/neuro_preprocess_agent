#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data/public_qc_sources"
DEFAULT_ABIDE_MANIFEST = PROJECT_ROOT / "data/qc_labels/abide_pcp_download_manifest.csv"

FIELDS = [
    "dataset",
    "sample_id",
    "subject_id",
    "modality",
    "artifact_type",
    "qc_label",
    "label_strength",
    "expert_score",
    "score_definition",
    "image_path",
    "report_path",
    "download_status",
    "notes",
]


def _existing(path: Path) -> str:
    return "downloaded" if path.is_file() and path.stat().st_size > 0 else "metadata_only"


def _mrart_rows(source_root: Path) -> list[dict[str, str]]:
    metadata = source_root / "repos/ds004173/derivatives/scores.tsv"
    pilot = source_root / "ds004173_mrart_pilot"
    rows: list[dict[str, str]] = []
    with metadata.open(encoding="utf-8") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            sample_id = item["bids_name"]
            subject_id = sample_id.split("_", 1)[0]
            score = int(item["score"])
            image_path = pilot / subject_id / "anat" / f"{sample_id}.nii.gz"
            report_path = pilot / "derivatives/mriqc-0.16.1" / f"{sample_id}.html"
            acquisition = sample_id.split("_acq-", 1)[1].split("_", 1)[0]
            rows.append(
                {
                    "dataset": "MR-ART/ds004173",
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "modality": "T1w",
                    "artifact_type": "head_motion" if acquisition != "standard" else "clean_reference",
                    "qc_label": {1: "pass", 2: "needs_review", 3: "fail"}[score],
                    "label_strength": "expert",
                    "expert_score": str(score),
                    "score_definition": "1=good, 2=medium, 3=bad",
                    "image_path": str(image_path),
                    "report_path": str(report_path),
                    "download_status": _existing(image_path),
                    "notes": f"matched acquisition={acquisition}",
                }
            )
    return rows


def _motion_correction_rows(source_root: Path) -> list[dict[str, str]]:
    repo = source_root / "repos/ds004332"
    pilot = source_root / "ds004332_motion_pilot"
    rows: list[dict[str, str]] = []
    for score_file in sorted((repo / "derivatives/observer_scores").glob("*.tsv")):
        if score_file.name == "dwi.tsv":
            continue
        with score_file.open(encoding="utf-8") as handle:
            for item in csv.DictReader(handle, delimiter="\t"):
                score_text = item.get("neuroradiologist_score", "").strip()
                if not score_text:
                    continue
                score = int(score_text)
                subject_id = item["participant_id"]
                sequence = item["sequence"]
                suffix = sequence.rsplit("_", 1)[-1]
                image_path = pilot / subject_id / "anat" / f"{subject_id}_{sequence}.nii"
                run = "still" if "run-01" in sequence else "nod" if "run-02" in sequence else "shake"
                rows.append(
                    {
                        "dataset": "PMC motion/ds004332",
                        "sample_id": f"{subject_id}_{sequence}",
                        "subject_id": subject_id,
                        "modality": suffix,
                        "artifact_type": f"head_motion_{run}",
                        "qc_label": "pass" if score >= 4 else "needs_review" if score == 3 else "fail",
                        "label_strength": "expert",
                        "expert_score": str(score),
                        "score_definition": "1=worst, 5=best",
                        "image_path": str(image_path),
                        "report_path": "",
                        "download_status": _existing(image_path),
                        "notes": "two radiographers plus one neuroradiologist; catalog uses neuroradiologist score",
                    }
                )
    return rows


def _pilot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["label_strength"] == "expert" and row["download_status"] == "downloaded"
    ]
    abide = [row for row in rows if row["dataset"] == "ABIDE-I/PCP"]
    abide.sort(
        key=lambda row: (
            "fail" not in row["expert_score"],
            "dropout" not in row["notes"].lower(),
            row["notes"],
            row["sample_id"],
        )
    )
    site_counts: Counter[str] = Counter()
    diverse: list[dict[str, str]] = []
    for row in abide:
        site = row["notes"].split(";", 1)[0]
        if site_counts[site] < 2:
            diverse.append(row)
            site_counts[site] += 1
        if len(diverse) == 30:
            break
    if len(diverse) < 30:
        for row in abide:
            if row not in diverse:
                diverse.append(row)
            if len(diverse) == 30:
                break
    return selected + diverse


def _abide_rows(manifest: Path) -> list[dict[str, str]]:
    by_subject: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with manifest.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            by_subject[item["file_id"]][item["derivative"]] = item

    rows: list[dict[str, str]] = []
    for file_id, derivatives in sorted(by_subject.items()):
        representative = derivatives.get("func_mean") or next(iter(derivatives.values()))
        if representative["label"] != "needs_review":
            continue
        image = Path(representative["path"])
        rows.append(
            {
                "dataset": "ABIDE-I/PCP",
                "sample_id": file_id,
                "subject_id": representative["subject_id"],
                "modality": "BOLD_mean",
                "artifact_type": "coverage_or_functional_artifact",
                "qc_label": "needs_review",
                "label_strength": "weak_disagreement",
                "expert_score": f"rater2={representative['rater_2']};rater3={representative['rater_3']}",
                "score_definition": "PCP visual raters; disagreement is not a confirmed failure",
                "image_path": str(image),
                "report_path": "",
                "download_status": _existing(PROJECT_ROOT / image),
                "notes": f"site={representative['site_id']}; {representative['notes']}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified catalog of public MRI QC samples.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--abide-manifest", type=Path, default=DEFAULT_ABIDE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_SOURCE_ROOT / "public_qc_catalog.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SOURCE_ROOT / "public_qc_catalog_summary.json")
    parser.add_argument("--pilot-output", type=Path, default=DEFAULT_SOURCE_ROOT / "public_qc_pilot.csv")
    args = parser.parse_args()

    rows = _mrart_rows(args.source_root) + _motion_correction_rows(args.source_root)
    if args.abide_manifest.is_file():
        rows.extend(_abide_rows(args.abide_manifest))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pilot_rows = _pilot_rows(rows)
    with args.pilot_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(pilot_rows)

    summary = {
        "catalog": str(args.output),
        "total_samples": len(rows),
        "by_dataset": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "by_qc_label": dict(sorted(Counter(row["qc_label"] for row in rows).items())),
        "by_label_strength": dict(sorted(Counter(row["label_strength"] for row in rows).items())),
        "downloaded": sum(row["download_status"] == "downloaded" for row in rows),
        "pilot_catalog": str(args.pilot_output),
        "pilot_samples": len(pilot_rows),
        "pilot_by_dataset": dict(sorted(Counter(row["dataset"] for row in pilot_rows).items())),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
