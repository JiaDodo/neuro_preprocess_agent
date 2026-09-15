from __future__ import annotations

import csv
import gzip
import json
import struct
from pathlib import Path
from typing import Any

from neuro_preprocess_agent.config import project_path
from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.state import PipelineState
from neuro_preprocess_agent.tools.qc_metrics import compute_quantitative_metrics
from neuro_preprocess_agent.tools.visual_qc import run_visual_qc_shadow


def _nifti_header(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "passed": False}
    if not path.exists() or path.stat().st_size <= 348:
        return result
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as file_handle:
            header = file_handle.read(348)
        endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
        dims = struct.unpack(f"{endian}8h", header[40:56])
        ndim = dims[0]
        shape = [int(value) for value in dims[1 : 1 + max(0, ndim)]]
        result.update({
            "passed": struct.unpack(f"{endian}i", header[:4])[0] == 348 and 2 <= ndim <= 7 and all(value > 0 for value in shape),
            "shape": shape,
            "datatype": int(struct.unpack(f"{endian}h", header[70:72])[0]),
            "bitpix": int(struct.unpack(f"{endian}h", header[72:74])[0]),
        })
    except Exception as exc:  # noqa: BLE001 - malformed headers become QC evidence, not worker crashes
        result["error"] = str(exc)
    return result


def _matching_mask(image: Path, candidates: list[Path], fallback: Path) -> Path:
    image_shape = _nifti_header(image).get("shape", [])
    for candidate in candidates:
        mask_shape = _nifti_header(candidate).get("shape", [])
        if image_shape[:3] and image_shape[:3] == mask_shape[:3]:
            return candidate
    return candidates[0] if candidates else fallback


def _numeric_column(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(column, "")))
        except (TypeError, ValueError):
            continue
    return values


def run_qc(state: PipelineState) -> dict[str, Any]:
    config = state["config"]
    settings = config["qc"]
    processed = state.get("processed_data", {})
    mode = config["runtime"]["mode"]
    output_items = int(processed.get("output_items", 0))
    checks: list[dict[str, Any]] = [{
        "name": "minimum_output_items",
        "passed": output_items >= int(settings["min_output_items"]),
        "observed": output_items,
        "threshold": int(settings["min_output_items"]),
        "severity": "error",
    }]
    bids_preflight = processed.get("bids_preflight")
    if bids_preflight:
        preflight_status = bids_preflight.get("status")
        checks.append({
            "name": "bids_preflight",
            "passed": bool(bids_preflight.get("passed")),
            "status": preflight_status,
            "errors": bids_preflight.get("errors", []),
            "warnings": bids_preflight.get("warnings", []),
            "severity": "warning" if preflight_status == "not_executed" else "error",
        })

    if mode != "run":
        hard_checks_passed = all(item["passed"] for item in checks if item["severity"] == "error")
        return {
            "passed": hard_checks_passed,
            "hard_checks_passed": hard_checks_passed,
            "review_required": False,
            "approved_for_database": hard_checks_passed,
            "status": "planned",
            "engine": settings["engine"],
            "checks": checks,
            "subject_qc": [],
            "visual_qc_artifacts": [],
            "visual_model_qc": run_visual_qc_shadow(state),
            "failed_checks": [item["name"] for item in checks if not item["passed"]],
        }

    fmriprep_dir = Path(processed["fmriprep_dir"])
    execution_results = {item.get("subject"): item for item in processed.get("subject_results", [])}
    subject_modalities = processed.get("subject_modalities", {})
    subject_qc: list[dict[str, Any]] = []
    visual_artifacts: list[dict[str, Any]] = []
    quantitative_metrics: list[dict[str, Any]] = []
    quantitative_settings = {
        "enabled": True,
        "output_dir": "runs/qc",
        "max_nonfinite_fraction": 0.0,
        "min_mask_voxels": 1,
        "max_mask_edge_fraction": None,
        "min_bold_tsnr": None,
        "min_anat_bold_mask_dice": None,
        **settings.get("quantitative", {}),
    }

    for subject in processed.get("subjects", []):
        subject_dir = fmriprep_dir / f"sub-{subject}"
        anat_dir = subject_dir / "anat"
        func_dir = subject_dir / "func"
        figures_dir = subject_dir / "figures"
        report = fmriprep_dir / f"sub-{subject}.html"
        modalities = subject_modalities.get(subject) or {"anat": True, "func": True}
        required = {"subject_html_report": report}
        image_outputs: list[str] = []
        spatial_pairs: list[tuple[str, str, str]] = []
        if modalities.get("anat"):
            preproc_t1w = next(iter(sorted(anat_dir.glob("*desc-preproc_T1w.nii.gz"))), anat_dir / "missing-preproc-T1w.nii.gz")
            t1w_mask = _matching_mask(
                preproc_t1w,
                sorted(anat_dir.glob("*desc-brain_mask.nii.gz")),
                anat_dir / "missing-T1w-mask.nii.gz",
            )
            required.update({
                "preproc_t1w": preproc_t1w,
                "t1w_brain_mask": t1w_mask,
            })
            image_outputs.extend(("preproc_t1w", "t1w_brain_mask"))
            spatial_pairs.append(("anat", "preproc_t1w", "t1w_brain_mask"))
        if modalities.get("func"):
            preproc_bold = next(iter(sorted(func_dir.glob("*desc-preproc_bold.nii.gz"))), func_dir / "missing-preproc-bold.nii.gz")
            bold_mask = _matching_mask(
                preproc_bold,
                sorted(func_dir.glob("*desc-brain_mask.nii.gz")),
                func_dir / "missing-bold-mask.nii.gz",
            )
            required.update({
                "preproc_bold": preproc_bold,
                "bold_brain_mask": bold_mask,
                "confounds_timeseries": next(iter(sorted(func_dir.glob("*desc-confounds_timeseries.tsv"))), func_dir / "missing-confounds.tsv"),
            })
            image_outputs.extend(("preproc_bold", "bold_brain_mask"))
            spatial_pairs.append(("func", "preproc_bold", "bold_brain_mask"))
        subject_checks: list[dict[str, Any]] = []
        for name, path in required.items():
            exists = path.exists()
            subject_checks.append({
                "name": f"{subject}:{name}", "passed": exists and path.stat().st_size > 0,
                "path": str(path), "size": path.stat().st_size if exists else 0, "severity": "error",
            })

        image_headers: dict[str, dict[str, Any]] = {}
        for name in image_outputs:
            image_check = _nifti_header(required[name])
            image_headers[name] = image_check
            image_check.update({"name": f"{subject}:nifti_header:{name}", "severity": "error"})
            subject_checks.append(image_check)
        for modality, image_name, mask_name in spatial_pairs:
            image_shape = image_headers.get(image_name, {}).get("shape", [])
            mask_shape = image_headers.get(mask_name, {}).get("shape", [])
            subject_checks.append({
                "name": f"{subject}:spatial_shape_match:{modality}",
                "passed": bool(image_shape[:3]) and image_shape[:3] == mask_shape[:3],
                "image": str(required[image_name]),
                "mask": str(required[mask_name]),
                "image_shape": image_shape,
                "mask_shape": mask_shape,
                "severity": "error",
            })

        quantitative_images = {name: required[name] for name in image_outputs}
        standard_t1w_mask = next(iter(sorted(anat_dir.glob("*space-*_desc-brain_mask.nii.gz"))), None)
        if standard_t1w_mask is not None:
            quantitative_images["t1w_standard_brain_mask"] = standard_t1w_mask
        quantitative = compute_quantitative_metrics(
            subject,
            quantitative_images,
            quantitative_settings,
        )
        subject_checks.extend(quantitative["checks"])
        quantitative_metrics.append(quantitative)

        figure_files = sorted(figures_dir.glob("*.svg")) + sorted(figures_dir.glob("*.html"))
        subject_checks.append({
            "name": f"{subject}:visual_report_artifacts",
            "passed": report.exists() and report.stat().st_size > 0 and len(figure_files) >= int(settings["min_visual_artifacts"]),
            "html_report": str(report), "figure_count": len(figure_files),
            "threshold": int(settings["min_visual_artifacts"]), "severity": "error",
        })
        visual_artifacts.append({"subject": subject, "html_report": str(report), "figures": [str(path) for path in figure_files]})

        confounds = required.get("confounds_timeseries")
        if confounds and confounds.exists():
            with confounds.open(encoding="utf-8") as file_handle:
                rows = list(csv.DictReader(file_handle, delimiter="\t"))
            metrics: dict[str, Any] = {"rows": len(rows)}
            for column in ("framewise_displacement", "std_dvars", "dvars"):
                values = _numeric_column(rows, column)
                if values:
                    metrics[column] = {"max": max(values), "mean": sum(values) / len(values)}
            subject_checks.append({"name": f"{subject}:confounds_summary", "passed": bool(rows), "metrics": metrics, "severity": "warning"})
            fd_values = _numeric_column(rows, "framewise_displacement")
            std_dvars_values = _numeric_column(rows, "std_dvars")
            fd_threshold = float(settings.get("fd_threshold_mm", 0.5))
            mean_fd = sum(fd_values) / len(fd_values) if fd_values else None
            fd_outlier_fraction = (
                sum(value > fd_threshold for value in fd_values) / len(fd_values) if fd_values else None
            )
            mean_std_dvars = sum(std_dvars_values) / len(std_dvars_values) if std_dvars_values else None
            threshold_results = {
                "mean_fd": settings.get("max_mean_fd_mm") is None
                or (mean_fd is not None and mean_fd <= float(settings["max_mean_fd_mm"])),
                "fd_outlier_fraction": settings.get("max_fd_outlier_fraction") is None
                or (
                    fd_outlier_fraction is not None
                    and fd_outlier_fraction <= float(settings["max_fd_outlier_fraction"])
                ),
                "mean_std_dvars": settings.get("max_mean_std_dvars") is None
                or (mean_std_dvars is not None and mean_std_dvars <= float(settings["max_mean_std_dvars"])),
            }
            subject_checks.append({
                "name": f"{subject}:motion_review_thresholds",
                "passed": all(threshold_results.values()),
                "observed": {
                    "mean_fd_mm": mean_fd,
                    "fd_outlier_fraction": fd_outlier_fraction,
                    "mean_std_dvars": mean_std_dvars,
                },
                "thresholds": {
                    "fd_threshold_mm": fd_threshold,
                    "max_mean_fd_mm": settings.get("max_mean_fd_mm"),
                    "max_fd_outlier_fraction": settings.get("max_fd_outlier_fraction"),
                    "max_mean_std_dvars": settings.get("max_mean_std_dvars"),
                },
                "threshold_results": threshold_results,
                "severity": "warning",
            })

        execution = execution_results.get(subject, {})
        log_path = Path(execution.get("log_file", ""))
        log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
        reuse_evidence = Path(execution.get("reuse_evidence", ""))
        reused_with_evidence = execution.get("status") == "reused" and reuse_evidence.is_file()
        subject_checks.append({
            "name": f"{subject}:fmriprep_log_success",
            "passed": execution.get("returncode") == 0
            and ("fMRIPrep finished successfully!" in log_text or reused_with_evidence),
            "returncode": execution.get("returncode"),
            "status": execution.get("status"),
            "log_file": str(log_path),
            "reuse_evidence": str(reuse_evidence) if reused_with_evidence else None,
            "severity": "error",
        })
        checks.extend(subject_checks)
        subject_qc.append({
            "subject": subject,
            "passed": all(item["passed"] for item in subject_checks if item["severity"] == "error"),
            "checks": subject_checks,
            "quantitative_metrics": quantitative,
        })

    hard_checks_passed = all(item["passed"] for item in checks if item["severity"] == "error")
    review_required = hard_checks_passed and any(
        not item["passed"] for item in checks if item["severity"] == "warning"
    )
    approved_for_database = hard_checks_passed and (
        not review_required or not bool(settings.get("block_database_on_review", True))
    )
    result = {
        "passed": approved_for_database,
        "hard_checks_passed": hard_checks_passed,
        "review_required": review_required,
        "approved_for_database": approved_for_database,
        "status": "failed" if not hard_checks_passed else ("review_required" if review_required else "completed"),
        "engine": settings["engine"],
        "checks": checks,
        "subject_qc": subject_qc,
        "visual_qc_artifacts": visual_artifacts,
        "quantitative_metrics": quantitative_metrics,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
    }
    if state.get("run_id") and config.get("project_root"):
        metrics_dir = project_path(quantitative_settings["output_dir"], config["project_root"]) / state["run_id"]
        metrics_path = metrics_dir / "dataset_qc_metrics.json"
        atomic_write_text(
            metrics_path,
            json.dumps({"run_id": state["run_id"], "subjects": quantitative_metrics}, ensure_ascii=False, indent=2),
        )
        result["quantitative_metrics_path"] = str(metrics_path)
    result["visual_model_qc"] = run_visual_qc_shadow(state)
    return result
