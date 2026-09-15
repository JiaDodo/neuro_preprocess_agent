from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from neuro_preprocess_agent.config import project_path
from neuro_preprocess_agent.db import check_database
from neuro_preprocess_agent.state import PipelineState


def _check(name: str, passed: bool, severity: str, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "severity": severity, "detail": detail}


def run_preflight(state: PipelineState) -> dict[str, Any]:
    config = state["config"]
    settings = config["preflight"]
    mode = config["runtime"]["mode"]
    preprocess = config["preprocess"]
    checks: list[dict[str, Any]] = []

    if not settings["enabled"]:
        return {"status": "disabled", "passed": True, "checks": [], "hard_failures": [], "warnings": []}

    source = config["source"]
    if source["type"] == "local_path":
        source_path = project_path(source["path"], config.get("project_root"))
        checks.append(_check("source_readable", source_path.exists() and os.access(source_path, os.R_OK), "error", str(source_path)))

    output_paths = [
        project_path(config["storage"]["raw_dir"], config.get("project_root")),
        project_path(preprocess["derivatives_dir"], config.get("project_root")),
        project_path(config["storage"]["report_dir"], config.get("project_root")),
    ]
    for path in output_paths:
        existing_parent = next((parent for parent in (path, *path.parents) if parent.exists()), None)
        writable = existing_parent is not None and os.access(existing_parent, os.W_OK)
        checks.append(_check(f"output_writable:{path}", writable, "error", str(existing_parent or path)))

    disk_anchor = next((parent for parent in (output_paths[1], *output_paths[1].parents) if parent.exists()), Path.cwd())
    free_disk_gb = shutil.disk_usage(disk_anchor).free / (1024**3)
    checks.append(_check(
        "free_disk_space",
        mode != "run" or free_disk_gb >= float(settings["minimum_free_disk_gb"]),
        "error",
        {"free_gb": round(free_disk_gb, 2), "minimum_gb": settings["minimum_free_disk_gb"], "path": str(disk_anchor)},
    ))

    run_fmriprep = bool(preprocess["run_fmriprep"])
    if mode == "run" and run_fmriprep and settings["check_docker"]:
        docker = shutil.which("docker")
        daemon = subprocess.run(
            [docker, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        ) if docker else None
        checks.append(_check("docker_daemon", bool(daemon and daemon.returncode == 0), "error", docker or "not found"))
        image = preprocess["fmriprep_image"]
        inspected = subprocess.run(
            [docker, "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            check=False,
        ) if docker and daemon and daemon.returncode == 0 else None
        checks.append(_check(
            "fmriprep_image",
            bool(inspected and inspected.returncode == 0),
            "error",
            {"reference": image, "image_id": inspected.stdout.strip() if inspected else None},
        ))
        license_path = project_path(preprocess["freesurfer_license"], config.get("project_root"))
        checks.append(_check(
            "freesurfer_license",
            license_path.is_file() and license_path.stat().st_size > 0 and os.access(license_path, os.R_OK),
            "error",
            str(license_path),
        ))

    explicit_dicom = preprocess["input_format"] == "dicom" or preprocess.get("dicom_jobs")
    if mode == "run" and explicit_dicom:
        dcm2bids = project_path(preprocess["dcm2bids"], config.get("project_root"))
        checks.append(_check("dcm2bids_executable", dcm2bids.is_file() and os.access(dcm2bids, os.X_OK), "error", str(dcm2bids)))
        dcm2niix = shutil.which("dcm2niix") or str(dcm2bids.with_name("dcm2niix"))
        checks.append(_check("dcm2niix_executable", Path(dcm2niix).is_file() and os.access(dcm2niix, os.X_OK), "error", dcm2niix))

    options = preprocess["fmriprep_options"]
    cpu_count = os.cpu_count() or 1
    available_memory_mb: int | None = None
    try:
        memory_lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        available_kb = next(
            int(line.split()[1]) for line in memory_lines if line.startswith("MemAvailable:")
        )
        available_memory_mb = available_kb // 1024
    except (FileNotFoundError, StopIteration, ValueError):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_memory_mb = int(pages * page_size / (1024**2))
        except (OSError, ValueError):
            pass
    requested_cpu = (options.get("nprocs") or 1) * int(preprocess["max_workers"])
    requested_memory = (options.get("memory_mb") or 0) * int(preprocess["max_workers"])
    if mode == "run" and settings["check_resource_budget"]:
        checks.append(_check(
            "cpu_budget",
            requested_cpu <= cpu_count,
            "warning",
            {"requested": requested_cpu, "available": cpu_count},
        ))
        if requested_memory and available_memory_mb is not None:
            checks.append(_check(
                "memory_budget",
                requested_memory <= available_memory_mb,
                "warning",
                {"requested_mb": requested_memory, "available_mb": available_memory_mb},
            ))

    if mode == "run" and settings["check_database"] and config["database"]["enabled"]:
        database = check_database(config)
        checks.append(_check("database_connection", database.get("status") == "ready", "error", database))

    hard_failures = [item["name"] for item in checks if item["severity"] == "error" and not item["passed"]]
    warnings = [item["name"] for item in checks if item["severity"] == "warning" and not item["passed"]]
    return {
        "status": "passed" if not hard_failures else "failed",
        "passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "resources": {
            "cpu_count": cpu_count,
            "available_memory_mb": available_memory_mb,
            "free_disk_gb": round(free_disk_gb, 2),
            "max_workers": preprocess["max_workers"],
            "requested_cpu": requested_cpu,
            "requested_memory_mb": requested_memory,
        },
    }
