#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from neuro_preprocess_agent.io_utils import atomic_write_text


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a multi-subject preprocessing report")
    parser.add_argument("report", type=Path)
    parser.add_argument("--timing-report", type=Path, help="Optional earlier execution report used for subject durations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    timing_report = json.loads(args.timing_report.read_text(encoding="utf-8")) if args.timing_report else report
    subject_results = timing_report.get("processed_data", {}).get("subject_results", [])
    durations = [
        float(item["duration_seconds"])
        for item in subject_results
        if item.get("duration_seconds") is not None and item.get("status") == "completed"
    ]
    status_counts: dict[str, int] = {}
    for item in subject_results:
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    qc_by_subject = {
        str(item.get("subject")): item for item in report.get("qc_result", {}).get("subject_qc", [])
    }
    rows = []
    for item in subject_results:
        subject = str(item.get("subject"))
        qc = qc_by_subject.get(subject, {})
        motion = next(
            (
                check.get("observed", {})
                for check in qc.get("checks", [])
                if str(check.get("name", "")).endswith(":motion_review_thresholds")
            ),
            {},
        )
        quantitative = qc.get("quantitative_metrics", {}).get("images", {})
        rows.append({
            "subject": subject,
            "preprocess_status": item.get("status"),
            "duration_seconds": item.get("duration_seconds"),
            "hard_qc_passed": qc.get("passed"),
            "mean_fd_mm": motion.get("mean_fd_mm"),
            "fd_outlier_fraction": motion.get("fd_outlier_fraction"),
            "median_tsnr": quantitative.get("bold_tsnr", {}).get("median"),
            "anat_bold_mask_dice": quantitative.get("anat_bold_mask_overlap", {}).get("dice"),
        })

    summary = {
        "run_id": report.get("run_id"),
        "timing_run_id": timing_report.get("run_id"),
        "status": report.get("status"),
        "subject_count": len(subject_results),
        "status_counts": status_counts,
        "successful_subjects": sum(item.get("returncode") == 0 for item in subject_results),
        "failed_subjects": sum(item.get("returncode") not in (0, None) for item in subject_results),
        "timed_completed_subjects": len(durations),
        "duration_p50_seconds": median(durations) if durations else None,
        "duration_p95_seconds": percentile(durations, 0.95),
        "qc_status": report.get("qc_result", {}).get("status"),
        "database_status": (report.get("db_result") or {}).get("status") or "not_run",
        "subjects": rows,
    }
    output = args.output or args.report.with_name(f"{report.get('run_id', 'run')}_stability.json")
    atomic_write_text(output, json.dumps(summary, ensure_ascii=False, indent=2))

    markdown = [
        f"# Stability Summary: {summary['run_id']}",
        "",
        f"- Pipeline status: `{summary['status']}`",
        f"- Subjects: {summary['successful_subjects']}/{summary['subject_count']} successful",
        f"- Status counts: `{json.dumps(status_counts, ensure_ascii=False)}`",
        f"- Duration P50/P95: {summary['duration_p50_seconds']} / {summary['duration_p95_seconds']} seconds",
        f"- QC: `{summary['qc_status']}`",
        f"- Database: `{summary['database_status']}`",
        "",
        "| Subject | Preprocess | Seconds | Hard QC | Mean FD | FD outliers | tSNR | Mask Dice |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['subject']} | {row['preprocess_status']} | {row['duration_seconds']} | "
            f"{row['hard_qc_passed']} | {row['mean_fd_mm']} | {row['fd_outlier_fraction']} | "
            f"{row['median_tsnr']} | {row['anat_bold_mask_dice']} |"
        )
    atomic_write_text(output.with_suffix(".md"), "\n".join(markdown) + "\n")
    print(f"json={output}")
    print(f"markdown={output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
