from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


PHENO_CSV = Path("data/qc_labels/Phenotypic_V1_0b_preprocessed1.csv")
OUT_DIR = Path("data/abide_pcp")
MANIFEST_PATH = Path("data/qc_labels/abide_pcp_download_manifest.csv")
PROGRESS_PATH = Path("data/qc_labels/abide_pcp_download_progress.jsonl")
PIPELINE = "cpac"
STRATEGY = "nofilt_noglobal"
DERIVATIVES = ["func_mean", "func_mask", "func_preproc"]
MAX_WORKERS = 4
RETRIES = 3
S3_PREFIX = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"


def consensus(rater_2: str, rater_3: str) -> str:
    if rater_2 == "OK" and rater_3 == "OK":
        return "pass"
    if rater_2 == "fail" and rater_3 == "fail":
        return "fail"
    return "needs_review"


def download_one(item: dict[str, str]) -> dict[str, str]:
    path = Path(item["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        item["download_status"] = "exists"
        item["size"] = str(path.stat().st_size)
        return item

    last_error = ""
    tmp_path = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            urllib.request.urlretrieve(item["url"], tmp_path)
            tmp_path.replace(path)
            item["download_status"] = "downloaded"
            item["size"] = str(path.stat().st_size)
            return item
        except Exception as exc:
            last_error = str(exc)
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(2 * attempt)

    item["download_status"] = "failed"
    item["error"] = last_error
    item["size"] = "0"
    return item


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "func_efc",
        "func_fber",
        "func_fwhm",
        "func_dvars",
        "func_outlier",
        "func_quality",
        "func_mean_fd",
        "func_num_fd",
        "func_perc_fd",
        "func_gsr",
    ]

    items: list[dict[str, str]] = []
    with PHENO_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            file_id = row.get("FILE_ID", "")
            if not file_id or file_id == "no_filename":
                continue
            if not any(row.get(col, "").strip() for col in ["qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2", "qc_anat_rater_3", "qc_func_rater_3"]):
                continue
            for derivative in DERIVATIVES:
                filename = f"{file_id}_{derivative}.nii.gz"
                rel_path = Path("Outputs") / PIPELINE / STRATEGY / derivative / filename
                url = f"{S3_PREFIX}/{rel_path.as_posix()}"
                items.append(
                    {
                        "subject_id": row.get("SUB_ID", ""),
                        "file_id": file_id,
                        "site_id": row.get("SITE_ID", ""),
                        "dx_group": row.get("DX_GROUP", ""),
                        "sex": row.get("SEX", ""),
                        "age_at_scan": row.get("AGE_AT_SCAN", ""),
                        "pipeline": PIPELINE,
                        "strategy": STRATEGY,
                        "modality": "func",
                        "label": consensus(row.get("qc_func_rater_2", ""), row.get("qc_func_rater_3", "")),
                        "rater_2": row.get("qc_func_rater_2", ""),
                        "rater_3": row.get("qc_func_rater_3", ""),
                        "notes": "; ".join(x for x in [row.get("qc_func_notes_rater_2", ""), row.get("qc_func_notes_rater_3", "")] if x),
                        "sub_in_smp": row.get("SUB_IN_SMP", ""),
                        "derivative": derivative,
                        "url": url,
                        "path": str(OUT_DIR / rel_path),
                        "size": "0",
                        "download_status": "pending",
                        "error": "",
                        "metric_json": json.dumps({k: row.get(k, "") for k in metric_cols}, ensure_ascii=False),
                    }
                )

    fieldnames = [
        "subject_id",
        "file_id",
        "site_id",
        "dx_group",
        "sex",
        "age_at_scan",
        "pipeline",
        "strategy",
        "modality",
        "label",
        "rater_2",
        "rater_3",
        "notes",
        "sub_in_smp",
        "derivative",
        "url",
        "path",
        "size",
        "download_status",
        "error",
        "metric_json",
    ]

    print(f"[{datetime.now().isoformat(timespec='seconds')}] planned files: {len(items)}", flush=True)
    completed: list[dict[str, str]] = []
    progress_file = PROGRESS_PATH.open("a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_one, item) for item in items]
        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "download_status": "failed",
                    "derivative": "unknown",
                    "file_id": "unknown",
                    "size": "0",
                    "error": str(exc),
                }
            completed.append(result)
            progress_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress_file.flush()
            print(
                f"[{idx}/{len(items)}] {result['download_status']} {result['derivative']} {result['file_id']} {result.get('size', '0')}",
                flush=True,
            )
    progress_file.close()

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(completed, key=lambda x: (x["file_id"], x["derivative"])))

    counts: dict[str, int] = {}
    for row in completed:
        counts[row["download_status"]] = counts.get(row["download_status"], 0) + 1
    print(f"[{datetime.now().isoformat(timespec='seconds')}] done {counts}", flush=True)
    print(f"manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
