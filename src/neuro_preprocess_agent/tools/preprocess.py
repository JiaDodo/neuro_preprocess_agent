from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from neuro_preprocess_agent.config import project_path
from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.state import PipelineState
from neuro_preprocess_agent.tools.source import count_files


def _is_bids_dir(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_description.json").exists() and any(path.glob("sub-*"))


def _detect_format(path: Path) -> str:
    if _is_bids_dir(path):
        return "bids"
    if path.is_dir() and (any(path.rglob("*.dcm")) or any(path.rglob("DICOMDIR"))):
        return "dicom"
    if path.is_dir() and (any(path.rglob("*.nii")) or any(path.rglob("*.nii.gz"))):
        return "nifti"
    if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return "nifti"
    return "unknown"


def _nifti_target(
    source: Path,
    *,
    subject: str,
    session: str | None,
    datatype: str,
    suffix: str,
    task: str | None = None,
    run: int | None = None,
    acquisition: str | None = None,
) -> Path:
    extension = ".nii.gz" if source.name.endswith(".nii.gz") else ".nii"
    entities = [f"sub-{subject}"]
    if session:
        entities.append(f"ses-{session}")
    if task:
        entities.append(f"task-{task}")
    if acquisition:
        entities.append(f"acq-{acquisition}")
    if run:
        entities.append(f"run-{run}")
    directory = Path(f"sub-{subject}")
    if session:
        directory /= f"ses-{session}"
    return directory / datatype / ("_".join([*entities, suffix]) + extension)


def _nifti_sidecar(path: Path) -> Path:
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return path.with_name(f"{stem}.json")


def inspect_dicom_headers(dicom_dirs: list[Path], max_samples: int = 512) -> dict[str, Any]:
    import pydicom

    candidates: list[Path] = []
    for directory in dicom_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"DICOM input does not exist: {directory}")
        candidates.extend(path for path in directory.rglob("*") if path.is_file())
    if not candidates:
        raise FileNotFoundError("No files were found in the configured DICOM directories")

    candidates = sorted(set(candidates))
    selected = candidates
    if len(candidates) > max_samples:
        selected = [candidates[round(index * (len(candidates) - 1) / (max_samples - 1))] for index in range(max_samples)]
        first_by_parent = {path.parent: path for path in candidates}
        selected = sorted(set(selected) | set(first_by_parent.values()))

    studies: set[tuple[str, str]] = set()
    series: dict[str, dict[str, Any]] = {}
    readable = 0
    for path in selected:
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                specific_tags=[
                    "PatientID", "StudyInstanceUID", "StudyDate", "SeriesInstanceUID",
                    "SeriesNumber", "SeriesDescription", "ProtocolName", "Modality",
                ],
            )
        except Exception:  # noqa: BLE001,S112 - non-DICOM files are expected in source trees
            continue
        patient_id = str(getattr(dataset, "PatientID", ""))
        study_uid = str(getattr(dataset, "StudyInstanceUID", ""))
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if not patient_id or not study_uid:
            continue
        readable += 1
        studies.add((hashlib.sha256(patient_id.encode()).hexdigest()[:12], study_uid))
        if series_uid and series_uid not in series:
            series[series_uid] = {
                "study_date": str(getattr(dataset, "StudyDate", "")),
                "series_number": str(getattr(dataset, "SeriesNumber", "")),
                "series_description": str(getattr(dataset, "SeriesDescription", "")),
                "protocol_name": str(getattr(dataset, "ProtocolName", "")),
                "modality": str(getattr(dataset, "Modality", "")),
            }
    if not readable:
        raise ValueError("No readable DICOM headers were found")
    return {
        "sampled_files": len(selected),
        "readable_headers": readable,
        "study_count": len(studies),
        "series": sorted(series.values(), key=lambda item: (item["series_number"], item["series_description"])),
        "passed": len(studies) == 1,
    }


def _managed_fmriprep_args(settings: dict[str, Any]) -> list[str]:
    options = settings.get("fmriprep_options", {})
    args = ["--output-layout", options.get("output_layout", "bids")]
    if options.get("output_spaces"):
        args.extend(["--output-spaces", *options["output_spaces"]])
    for key, flag in (("nprocs", "--nprocs"), ("omp_nthreads", "--omp-nthreads"), ("memory_mb", "--mem")):
        if options.get(key) is not None:
            args.extend([flag, str(options[key])])
    if options.get("low_mem"):
        args.append("--low-mem")
    if options.get("anat_only"):
        args.append("--anat-only")
    if options.get("level", "full") != "full":
        args.extend(["--level", options["level"]])
    if options.get("ignore"):
        args.extend(["--ignore", *options["ignore"]])
    if options.get("force"):
        args.extend(["--force", *options["force"]])
    if options.get("subject_anatomical_reference", "first-lex") != "first-lex":
        args.extend(["--subject-anatomical-reference", options["subject_anatomical_reference"]])
    args.extend(["--skull-strip-t1w", options.get("skull_strip_t1w", "auto")])
    if options.get("fs_no_reconall", True):
        args.append("--fs-no-reconall")
    if options.get("skip_bids_validation"):
        args.append("--skip-bids-validation")
    if options.get("stop_on_first_crash", True):
        args.append("--stop-on-first-crash")

    args.extend(settings.get("fmriprep_args", []))
    return args


def _job_fingerprint(subject: str, bids_dir: str, image: str, fmriprep_args: list[str]) -> str:
    return hashlib.sha256(json.dumps({
        "subject": subject,
        "bids_dir": bids_dir,
        "fmriprep_image": image,
        "fmriprep_args": fmriprep_args,
    }, sort_keys=True).encode()).hexdigest()


def inspect_anatomical_inputs(state: PipelineState) -> dict[str, Any]:
    context = state["preprocess_context"]
    settings = state["config"]["preprocess"]
    configured_action = settings["fmriprep_options"].get("skull_strip_t1w", "auto")
    subjects = context.get("subjects", [])
    if not settings.get("input_qc_enabled", True):
        return {
            "status": "disabled",
            "review_required": False,
            "configured_action": configured_action,
            "subject_results": [],
            "recommended_actions": {subject: configured_action for subject in subjects},
        }
    if context.get("mode") != "run" or not context.get("run_fmriprep", True):
        return {
            "status": "not_executed",
            "reason": "Input QC requires run mode with fMRIPrep enabled",
            "review_required": False,
            "configured_action": configured_action,
            "subject_results": [],
            "recommended_actions": {subject: configured_action for subject in subjects},
        }

    import nibabel as nib
    import numpy as np

    bids_dir = Path(context["bids_dir"])
    threshold = float(settings.get("pre_skull_stripped_nonzero_fraction", 0.3))
    subject_results: list[dict[str, Any]] = []
    recommended_actions: dict[str, str] = {}
    for subject in subjects:
        has_anatomical_input = context.get("subject_modalities", {}).get(subject, {}).get("anat", True)
        files = sorted((bids_dir / f"sub-{subject}").rglob("*_T1w.nii"))
        files.extend(sorted((bids_dir / f"sub-{subject}").rglob("*_T1w.nii.gz")))
        images: list[dict[str, Any]] = []
        for path in files:
            try:
                image = nib.load(str(path))
                data = np.asanyarray(image.dataobj)
                finite = np.isfinite(data)
                nonzero_fraction = float(np.count_nonzero(data[finite]) / data.size) if data.size else 0.0
                nonfinite_fraction = float(1.0 - np.count_nonzero(finite) / data.size) if data.size else 1.0
                voxel_size = [round(float(value), 4) for value in image.header.get_zooms()[:3]]
                issues = []
                if len(image.shape) != 3:
                    issues.append(f"expected a 3D T1w image, found shape {image.shape}")
                if len(voxel_size) != 3 or any(not np.isfinite(value) or value <= 0 for value in voxel_size):
                    issues.append(f"invalid voxel size: {voxel_size}")
                if nonfinite_fraction > 0:
                    issues.append(f"non-finite voxel fraction is {nonfinite_fraction:.6f}")
                images.append({
                    "path": str(path),
                    "shape": list(image.shape),
                    "voxel_size_mm": voxel_size,
                    "nonzero_fraction": round(nonzero_fraction, 6),
                    "nonfinite_fraction": round(nonfinite_fraction, 6),
                    "suspected_pre_skull_stripped": nonzero_fraction < threshold,
                    "issues": issues,
                })
            except Exception as exc:  # noqa: BLE001 - inspection failure must be reviewable
                images.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

        flags = [item["suspected_pre_skull_stripped"] for item in images if "suspected_pre_skull_stripped" in item]
        invalid_images = any(item.get("error") or item.get("issues") for item in images)
        if not has_anatomical_input:
            classification = "not_applicable_no_t1w"
            recommendation = "auto"
            needs_review = False
        elif not images or len(flags) != len(images) or invalid_images:
            classification = "missing_unreadable_or_invalid_t1w"
            recommendation = "auto"
            needs_review = True
        elif configured_action != "auto":
            classification = "configured_override"
            recommendation = configured_action
            needs_review = False
        elif all(flags):
            classification = "suspected_pre_skull_stripped"
            recommendation = "skip"
            needs_review = True
        elif any(flags):
            classification = "mixed_or_uncertain"
            recommendation = "auto"
            needs_review = True
        else:
            classification = "skull_present_or_defaced"
            recommendation = "force"
            needs_review = False
        recommended_actions[subject] = recommendation
        subject_results.append({
            "subject": subject,
            "classification": classification,
            "recommended_action": recommendation,
            "review_required": needs_review,
            "images": images,
        })

    review_required = bool(
        settings.get("input_qc_require_review", True)
        and any(item["review_required"] for item in subject_results)
    )
    return {
        "status": "review_required" if review_required else "completed",
        "method": "T1w nonzero-volume screening",
        "limitations": "This heuristic flags likely pre-skull-stripped inputs; it is not a definitive anatomical classifier.",
        "threshold": threshold,
        "configured_action": configured_action,
        "review_required": review_required,
        "subject_results": subject_results,
        "recommended_actions": recommended_actions,
    }


def apply_skull_strip_actions(
    jobs: list[dict[str, Any]],
    actions: dict[str, str],
) -> list[dict[str, Any]]:
    updated_jobs: list[dict[str, Any]] = []
    for original in jobs:
        job = {**original, "command": list(original["command"]), "fmriprep_args": list(original["fmriprep_args"])}
        action = actions.get(job["subject"], job.get("skull_strip_t1w", "auto"))
        if action not in {"auto", "skip", "force"}:
            raise ValueError(f"Unsupported skull stripping action for sub-{job['subject']}: {action}")
        argument_index = job["fmriprep_args"].index("--skull-strip-t1w")
        job["fmriprep_args"][argument_index + 1] = action
        command_index = job["command"].index("--skull-strip-t1w")
        job["command"][command_index + 1] = action
        job["skull_strip_t1w"] = action
        job["configuration_fingerprint"] = _job_fingerprint(
            job["subject"], job["bids_dir"], job["fmriprep_image"], job["fmriprep_args"]
        )
        updated_jobs.append(job)
    return updated_jobs


def inspect_bids_dataset(bids_dir: Path, selected_subjects: list[str] | None = None) -> dict[str, Any]:
    from bids import BIDSLayout

    errors: list[str] = []
    warnings: list[str] = []
    modalities: dict[str, dict[str, Any]] = {}
    description_path = bids_dir / "dataset_description.json"
    try:
        description = json.loads(description_path.read_text(encoding="utf-8"))
        if not str(description.get("Name", "")).strip():
            warnings.append("dataset_description.json has an empty Name")
    except FileNotFoundError:
        errors.append("dataset_description.json is missing")
    except json.JSONDecodeError as exc:
        errors.append(f"dataset_description.json is invalid JSON: {exc}")

    empty_files = [str(path.relative_to(bids_dir)) for path in bids_dir.rglob("*") if path.is_file() and path.stat().st_size == 0]
    if empty_files:
        errors.append(f"empty files are not allowed: {', '.join(empty_files[:10])}")

    try:
        layout = BIDSLayout(bids_dir, validate=False, derivatives=False)
        available_subjects = sorted(layout.get_subjects())
        requested = list(dict.fromkeys(selected_subjects or []))
        missing_subjects = sorted(set(requested) - set(available_subjects))
        if missing_subjects:
            errors.append("requested BIDS subjects were not found: " + ", ".join(missing_subjects))
        subjects = [subject for subject in requested if subject in available_subjects] if requested else available_subjects
    except Exception as exc:  # noqa: BLE001 - preserve PyBIDS indexing diagnostics
        errors.append(f"PyBIDS could not index the dataset: {type(exc).__name__}: {exc}")
        subjects = []
        layout = None
    if not subjects:
        errors.append("no BIDS subjects were found")

    for subject in subjects:
        files = layout.get(subject=subject, return_type="filename") if layout is not None else []
        t1w_files = sorted(path for path in files if path.endswith(("_T1w.nii", "_T1w.nii.gz")))
        bold_files = sorted(path for path in files if path.endswith(("_bold.nii", "_bold.nii.gz")))
        subject_info = {
            "anat": bool(t1w_files),
            "func": bool(bold_files),
            "t1w_count": len(t1w_files),
            "bold_count": len(bold_files),
        }
        modalities[subject] = subject_info
        if not t1w_files and not bold_files:
            errors.append(f"sub-{subject} has no T1w or BOLD image supported by fMRIPrep")
        for bold_file in bold_files:
            try:
                metadata = layout.get_metadata(bold_file)
            except Exception as exc:  # noqa: BLE001 - preserve per-file metadata diagnostics
                errors.append(f"cannot read inherited metadata for {Path(bold_file).name}: {exc}")
                continue
            repetition_time = metadata.get("RepetitionTime")
            if not isinstance(repetition_time, (int, float)) or repetition_time <= 0:
                errors.append(f"{Path(bold_file).name} has no positive RepetitionTime in BIDS metadata")

    return {
        "status": "completed",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "subjects": subjects,
        "subject_modalities": modalities,
        "subject_sessions": {
            subject: sorted(layout.get_sessions(subject=subject)) if layout is not None else []
            for subject in subjects
        },
    }


def validate_bids_with_container(bids_dir: Path, image: str) -> dict[str, Any]:
    command = [
        "docker", "run", "--rm", "--entrypoint", "bids-validator",
        "-v", f"{bids_dir.resolve()}:/data:ro", image, "/data", "--json",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        return {
            "status": "failed", "passed": False,
            "errors": [f"BIDS Validator did not return JSON: {detail}"],
            "warnings": [], "command": command,
        }

    def issue_messages(level: str) -> list[str]:
        messages: list[str] = []
        for group in payload.get("issues", {}).get(level, []):
            files = group.get("files") or [{}]
            for item in files:
                path = item.get("file", {}).get("relativePath") or item.get("file", {}).get("name") or "dataset"
                evidence = item.get("evidence") or group.get("reason") or group.get("key") or "unknown issue"
                messages.append(f"{path}: {evidence}")
        return messages

    errors = issue_messages("errors")
    warnings = issue_messages("warnings")
    return {
        "status": "completed", "passed": completed.returncode == 0 and not errors,
        "errors": errors, "warnings": warnings, "command": command,
        "summary": payload.get("summary", {}),
    }


def prepare_preprocessing(state: PipelineState) -> dict[str, Any]:
    config = state["config"]
    mode = config["runtime"]["mode"]
    settings = config["preprocess"]
    root = config.get("project_root")
    raw_location = Path(state.get("raw_data", {}).get("location", config["storage"]["raw_dir"])).expanduser()
    bids_dir = project_path(settings["bids_dir"], root)
    derivatives_dir = project_path(settings["derivatives_dir"], root)
    log_dir = project_path(settings["log_dir"], root)
    work_dir = project_path(settings["work_dir"], root)
    lock_dir = project_path(settings["lock_dir"], root)
    input_format = settings.get("input_format", "auto")
    nifti_type_counts = {"anat": 0, "func": 0}
    nifti_metadata_errors: list[str] = []

    if input_format == "auto":
        input_format = _detect_format(raw_location)
        if input_format == "unknown" and mode in {"mock", "dry_run"}:
            input_format = settings.get("assume_format", "bids")
    if input_format in {"dicom", "nifti"} and settings.get("isolate_converted_bids", True):
        bids_dir = bids_dir / state["run_id"]

    if input_format == "bids":
        bids_dir = raw_location
        bids_step: dict[str, Any] = {"status": "already_bids", "bids_dir": str(bids_dir)}
    elif input_format == "dicom":
        dcm2bids = project_path(settings["dcm2bids"], root)
        configured_jobs = settings.get("dicom_jobs", [])
        jobs_to_convert = configured_jobs or [{
            "participant_label": settings.get("participant_label", "001"),
            "session_label": None,
            "dicom_dirs": [str(raw_location)],
            "dcm2bids_config": settings.get("dcm2bids_config"),
        }]
        conversion_jobs: list[dict[str, Any]] = []
        for item in jobs_to_convert:
            dicom_dirs = []
            for value in item["dicom_dirs"]:
                candidate = Path(value).expanduser()
                dicom_dirs.append(candidate if candidate.is_absolute() else raw_location / candidate)
            config_value = item.get("dcm2bids_config") or settings["dcm2bids_config"]
            dcm2bids_config = project_path(config_value, root)
            command = [str(dcm2bids), "-d", *[str(path) for path in dicom_dirs], "-p", item["participant_label"]]
            if item.get("session_label"):
                command.extend(["-s", item["session_label"]])
            command.extend(["-c", str(dcm2bids_config), "-o", str(bids_dir)])
            header_check = None
            if mode == "run" and settings.get("dicom_header_validation", True):
                header_check = inspect_dicom_headers(dicom_dirs)
                if not header_check["passed"]:
                    raise ValueError(
                        "A DICOM conversion job contains multiple patients or studies. "
                        "Split it into explicit preprocess.dicom_jobs by participant/session."
                    )
            conversion_jobs.append({
                "participant_label": item["participant_label"],
                "session_label": item.get("session_label"),
                "dicom_dirs": [str(path) for path in dicom_dirs],
                "config": str(dcm2bids_config),
                "command": command,
                "header_check": header_check,
            })
        if mode == "run":
            if not dcm2bids.exists():
                raise FileNotFoundError(f"dcm2bids executable not found: {dcm2bids}")
            bids_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["PATH"] = os.pathsep.join([str(dcm2bids.parent), environment.get("PATH", "")])
            for job in conversion_jobs:
                if not Path(job["config"]).exists():
                    raise FileNotFoundError(f"dcm2bids config not found: {job['config']}")
                subprocess.run(job["command"], check=True, env=environment)
        bids_step = {
            "status": "converted" if mode == "run" else "planned",
            "input_format": "dicom",
            "jobs": conversion_jobs,
            "bids_dir": str(bids_dir),
        }
    elif input_format == "nifti":
        source_files = [raw_location] if raw_location.is_file() else sorted(raw_location.rglob("*.nii")) + sorted(raw_location.rglob("*.nii.gz"))
        if not source_files and mode == "run":
            raise FileNotFoundError(f"No NIfTI files found in: {raw_location}")
        planned_files: list[str] = []
        manifest = settings.get("nifti_manifest", [])
        if manifest:
            items = []
            for item in manifest:
                source_file = Path(item["path"]).expanduser()
                if not source_file.is_absolute():
                    source_file = raw_location / source_file
                items.append({**item, "source_file": source_file})
        else:
            has_subject_or_session_entities = any(
                re.search(r"(?:^|[_/])(sub|ses)-[A-Za-z0-9]+", str(path.relative_to(raw_location if raw_location.is_dir() else raw_location.parent)))
                for path in source_files
            )
            if has_subject_or_session_entities:
                raise ValueError(
                    "Non-BIDS NIfTI input contains subject/session entities. "
                    "Provide preprocess.nifti_manifest instead of relying on filename inference."
                )
            items = []
            counts = {"anat": 0, "func": 0}
            for source_file in source_files:
                datatype = "func" if any(token in source_file.name.lower() for token in ("bold", "fmri", "rest", "func")) else "anat"
                counts[datatype] += 1
                items.append({
                    "source_file": source_file,
                    "participant_label": settings["nifti_subject_label"],
                    "session_label": settings.get("nifti_session_label"),
                    "datatype": datatype,
                    "suffix": "bold" if datatype == "func" else "T1w",
                    "task": settings["nifti_task_name"] if datatype == "func" else None,
                    "run": counts[datatype] if counts[datatype] > 1 else None,
                    "acquisition": None,
                    "metadata": {},
                })
        for item in items:
            source_file = item["source_file"]
            if mode == "run" and not source_file.is_file():
                raise FileNotFoundError(f"NIfTI manifest source does not exist: {source_file}")
            category = item["datatype"]
            nifti_type_counts[category] += 1
            relative_target = _nifti_target(
                source_file,
                subject=item["participant_label"],
                session=item.get("session_label"),
                datatype=category,
                suffix=item["suffix"],
                task=item.get("task"),
                run=item.get("run"),
                acquisition=item.get("acquisition"),
            )
            target = bids_dir / relative_target
            planned_files.append(str(target))
            source_sidecar = _nifti_sidecar(source_file)
            metadata: dict[str, Any] = {}
            if source_sidecar.exists():
                metadata.update(json.loads(source_sidecar.read_text(encoding="utf-8")))
            metadata.update(settings.get("nifti_metadata", {}))
            metadata.update(item.get("metadata", {}))
            if category == "func":
                metadata.setdefault("TaskName", item["task"])
                repetition_time = metadata.get("RepetitionTime")
                if not isinstance(repetition_time, (int, float)) or repetition_time <= 0:
                    nifti_metadata_errors.append(
                        f"Functional NIfTI requires positive RepetitionTime metadata: {source_file}"
                    )
            if mode == "run":
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target)
                if metadata:
                    _nifti_sidecar(target).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if mode == "run":
            (bids_dir / "dataset_description.json").write_text(
                json.dumps({"Name": settings["bids_dataset_name"], "BIDSVersion": "1.9.0", "DatasetType": "raw"}, indent=2),
                encoding="utf-8",
            )
        bids_step = {"status": "organized" if mode == "run" else "planned", "input_format": "nifti", "bids_dir": str(bids_dir), "files": planned_files}
    else:
        raise ValueError(f"Cannot infer BIDS, DICOM, or NIfTI input from: {raw_location}")

    repairs: list[dict[str, str]] = []
    if mode == "run":
        if input_format == "dicom" and (bids_dir / "tmp_dcm2bids").exists():
            bidsignore = bids_dir / ".bidsignore"
            ignored = bidsignore.read_text(encoding="utf-8").splitlines() if bidsignore.exists() else []
            if "tmp_dcm2bids/" not in ignored:
                bidsignore.write_text("\n".join([*ignored, "tmp_dcm2bids/"]) + "\n", encoding="utf-8")
                repairs.append({"file": str(bidsignore), "repair": "ignored dcm2bids temporary conversion files"})
        description = bids_dir / "dataset_description.json"
        if not description.exists() and input_format == "dicom":
            description.write_text(
                json.dumps(
                    {"Name": settings["bids_dataset_name"], "BIDSVersion": "1.9.0", "DatasetType": "raw"},
                    indent=2,
                ),
                encoding="utf-8",
            )
            repairs.append({"file": str(description), "repair": "created required BIDS dataset description"})
        readme = bids_dir / "README"
        if not readme.exists() or readme.stat().st_size == 0:
            readme.write_text("Dataset prepared by neuro-preprocess-agent.\n", encoding="utf-8")
            repairs.append({"file": str(readme), "repair": "filled missing or empty BIDS README"})

    subjects = sorted(path.name.removeprefix("sub-") for path in bids_dir.glob("sub-*") if path.is_dir())
    if not subjects and mode != "run":
        subjects = [str(item) for item in (settings.get("subjects") or ["001"])]
    if not subjects and mode == "run":
        raise RuntimeError(f"No BIDS subjects found in: {bids_dir}")

    if _is_bids_dir(bids_dir):
        bids_preflight = inspect_bids_dataset(bids_dir, settings.get("subjects"))
        subjects = bids_preflight["subjects"] or subjects
        if (
            mode == "run"
            and settings.get("validate_bids", True)
            and settings.get("run_fmriprep", True)
            and not settings.get("fmriprep_options", {}).get("skip_bids_validation", False)
            and bids_preflight["passed"]
        ):
            official_validation = validate_bids_with_container(bids_dir, settings["fmriprep_image"])
            bids_preflight["official_validator"] = official_validation
            bids_preflight["passed"] = official_validation["passed"]
            bids_preflight["errors"].extend(official_validation["errors"])
            bids_preflight["warnings"].extend(official_validation["warnings"])
    else:
        assumed = set(settings.get("assume_modalities", ["anat", "func"]))
        if input_format == "nifti" and any(nifti_type_counts.values()):
            assumed = {name for name, count in nifti_type_counts.items() if count > 0}
        preflight_executed = input_format == "nifti" and any(nifti_type_counts.values())
        bids_preflight = {
            "status": "planned" if preflight_executed else ("mocked" if mode == "mock" else "not_executed"),
            "passed": mode == "mock" or (preflight_executed and not nifti_metadata_errors),
            "errors": nifti_metadata_errors,
            "warnings": [] if mode == "mock" or preflight_executed else ["BIDS output does not exist yet; validation was not executed"],
            "subjects": subjects,
            "subject_modalities": {
                subject: {
                    "anat": "anat" in assumed,
                    "func": "func" in assumed,
                    "t1w_count": nifti_type_counts["anat"],
                    "bold_count": nifti_type_counts["func"],
                }
                for subject in subjects
            },
        }
    if mode == "run" and settings.get("validate_bids", True) and not bids_preflight["passed"]:
        details = bids_preflight["errors"] or bids_preflight["warnings"] or [f"status={bids_preflight['status']}"]
        raise ValueError("BIDS preflight failed: " + "; ".join(details))

    derivatives_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir = log_dir / state["run_id"]
    run_work_dir = work_dir / state["run_id"]
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_work_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)
    license_file = project_path(settings["freesurfer_license"], root)
    image = settings["fmriprep_image"]
    jobs: list[dict[str, Any]] = []
    for subject in subjects if settings.get("run_fmriprep", True) else []:
        subject_work_dir = run_work_dir / f"sub-{subject}"
        subject_work_dir.mkdir(parents=True, exist_ok=True)
        fmriprep_args = _managed_fmriprep_args(settings)
        command = [
            "docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{license_file.resolve()}:/opt/freesurfer/license.txt:ro",
            "-v", f"{bids_dir.resolve()}:/data:ro",
            "-v", f"{derivatives_dir.resolve()}:/out",
            "-v", f"{subject_work_dir.resolve()}:/work",
            image, "/data", "/out", "participant", "--participant-label", subject,
            "--work-dir", "/work",
            *fmriprep_args,
        ]
        resolved_bids_dir = str(bids_dir.resolve())
        jobs.append({
            "run_id": state["run_id"],
            "subject": subject,
            "mode": mode,
            "command": command,
            "bids_dir": resolved_bids_dir,
            "fmriprep_image": image,
            "fmriprep_args": fmriprep_args,
            "skull_strip_t1w": settings["fmriprep_options"].get("skull_strip_t1w", "auto"),
            "log_file": str(run_log_dir / f"sub-{subject}.log"),
            "lock_file": str(lock_dir / f"sub-{subject}.lock"),
            "fmriprep_dir": str(derivatives_dir),
            "reuse_completed": bool(settings.get("reuse_completed", True)),
            "configuration_fingerprint": _job_fingerprint(subject, resolved_bids_dir, image, fmriprep_args),
        })

    if mode == "run" and settings.get("run_fmriprep", True):
        if shutil.which("docker") is None:
            raise RuntimeError("docker not found in PATH")
        if not license_file.exists():
            raise FileNotFoundError(f"FreeSurfer license not found: {license_file}")

    return {
        "preprocess_context": {
            "mode": mode,
            "input_format": input_format,
            "bids_step": bids_step,
            "bids_repairs": repairs,
            "bids_preflight": bids_preflight,
            "bids_dir": str(bids_dir),
            "fmriprep_dir": str(derivatives_dir),
            "log_dir": str(run_log_dir),
            "work_dir": str(run_work_dir),
            "subjects": subjects,
            "subject_modalities": bids_preflight["subject_modalities"],
            "subject_sessions": bids_preflight.get("subject_sessions", {subject: [] for subject in subjects}),
            "max_workers": int(settings["max_workers"]),
            "run_fmriprep": bool(settings.get("run_fmriprep", True)),
        },
        "preprocess_jobs": jobs,
    }


def run_subject_preprocessing(state: PipelineState) -> dict[str, Any]:
    job = state["preprocess_job"]
    result = {
        "run_id": job["run_id"],
        "subject": job["subject"],
        "log_file": job["log_file"],
    }
    if job["mode"] != "run":
        result.update({"status": "planned", "command": job["command"]})
        return {"preprocess_subject_results": [result]}

    log_path = Path(job["log_file"])
    lock_path = Path(job["lock_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(json.dumps({"run_id": job["run_id"], "subject": job["subject"], "started_at": started_at}))
        lock_file.flush()

        report = Path(job["fmriprep_dir"]) / f"sub-{job['subject']}.html"
        subject_dir = Path(job["fmriprep_dir"]) / f"sub-{job['subject']}"
        marker_path = Path(job["fmriprep_dir"]) / ".neuro_preprocess_agent" / f"sub-{job['subject']}.complete.json"
        marker_matches = False
        if marker_path.is_file():
            try:
                marker_matches = json.loads(marker_path.read_text(encoding="utf-8")).get(
                    "configuration_fingerprint"
                ) == job.get("configuration_fingerprint")
            except json.JSONDecodeError:
                marker_matches = False
        historical_logs = [log_path]
        if log_path.parent.parent.is_dir():
            historical_logs.extend(sorted(log_path.parent.parent.glob(f"*/sub-{job['subject']}.log")))
        legacy_success_log = next((
            path for path in historical_logs
            if path.is_file() and "fMRIPrep finished successfully!" in path.read_text(encoding="utf-8", errors="ignore")
        ), None)
        reusable_outputs = report.is_file() and any(subject_dir.rglob("*.nii.gz"))
        if job.get("reuse_completed") and reusable_outputs and (marker_matches or legacy_success_log):
            if not marker_matches:
                atomic_write_text(marker_path, json.dumps({
                    "subject": job["subject"],
                    "configuration_fingerprint": job.get("configuration_fingerprint"),
                    "adopted_from_log": str(legacy_success_log),
                    "completed_at": started_at,
                }, indent=2))
            result.update({
                "returncode": 0,
                "status": "reused",
                "reuse_evidence": str(marker_path if marker_path.is_file() else legacy_success_log),
                "started_at": started_at,
                "finished_at": started_at,
                "duration_seconds": 0.0,
            })
            return {"preprocess_subject_results": [result]}

        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                job["command"], stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False
            )
    result.update({
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds": round(time.monotonic() - started, 3),
    })
    if completed.returncode == 0 and report.is_file() and any(subject_dir.rglob("*.nii.gz")):
        atomic_write_text(marker_path, json.dumps({
            "subject": job["subject"],
            "configuration_fingerprint": job.get("configuration_fingerprint"),
            "run_id": job["run_id"],
            "completed_at": result["finished_at"],
            "log_file": str(log_path),
        }, indent=2))
    return {"preprocess_subject_results": [result]}


def finalize_preprocessing(state: PipelineState) -> dict[str, Any]:
    context = state["preprocess_context"]
    run_results = [
        item for item in state.get("preprocess_subject_results", [])
        if item.get("run_id") == state["run_id"]
    ]
    latest_by_subject = {item["subject"]: item for item in run_results}
    results = []
    for subject in context["subjects"]:
        results.append(latest_by_subject.get(subject, {
            "run_id": state["run_id"],
            "subject": subject,
            "status": "failed",
            "returncode": None,
            "error": "LangGraph subject worker did not return a result",
        }))

    status = "planned"
    if context["mode"] == "run" and not context.get("run_fmriprep", True):
        status = "bids_ready"
    elif context["mode"] == "run":
        status = "completed" if results and all(item.get("returncode") == 0 for item in results) else "failed"
    return {
        "status": status,
        "input_format": context["input_format"],
        "bids_step": context["bids_step"],
        "bids_repairs": context["bids_repairs"],
        "bids_preflight": context["bids_preflight"],
        "bids_dir": context["bids_dir"],
        "fmriprep_dir": context["fmriprep_dir"],
        "log_dir": context["log_dir"],
        "work_dir": context["work_dir"],
        "subjects": context["subjects"],
        "subject_modalities": context["subject_modalities"],
        "subject_sessions": context.get("subject_sessions", {}),
        "subject_results": results,
        "output_items": count_files(Path(context["fmriprep_dir"])) if context["mode"] == "run" else len(results),
        "max_workers": context["max_workers"],
    }
