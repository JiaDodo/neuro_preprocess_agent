from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuro_preprocess_agent.config import project_path
from neuro_preprocess_agent.io_utils import advisory_lock, atomic_write_text

LABELS = (
    "brain_mask_error",
    "normalization_error",
    "coregistration_error",
    "coverage_error",
    "ghosting_dropout",
    "segmentation_error",
    "severe_motion",
)

ANATOMICAL_INPUT_STATES = (
    "unknown",
    "full_head",
    "defaced",
    "pre_skull_stripped",
)


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "brain_mask_error",
        "normalization_error",
        "coregistration_error",
        "coverage_error",
        "ghosting_dropout",
        "segmentation_error",
        "severe_motion",
    ]
    abnormal: bool
    severity: Literal["low", "medium", "high"]
    evidence: str = Field(min_length=1, max_length=500)


class VisualAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["pass", "review", "fail"]
    confidence: float = Field(ge=0, le=1)
    observations: list[VisualObservation] = Field(default_factory=list, max_length=len(LABELS) * 2)
    summary: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def merge_duplicate_categories(self) -> VisualAssessment:
        severity_rank = {"low": 0, "medium": 1, "high": 2}
        merged: dict[str, VisualObservation] = {}
        for observation in self.observations:
            current = merged.get(observation.category)
            if current is None or (
                observation.abnormal,
                severity_rank[observation.severity],
            ) > (
                current.abnormal,
                severity_rank[current.severity],
            ):
                merged[observation.category] = observation
        self.observations = list(merged.values())
        return self


def _visual_imports():
    try:
        import nibabel as nib
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Visual QC requires nibabel, numpy and Pillow. Install the project with .[visual-qc]."
        ) from exc
    return nib, np, Image, ImageDraw, ImageFont


def _render_reportlet(path: Path, title: str):
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError("Rendering fMRIPrep SVG reportlets requires CairoSVG. Install .[visual-qc].") from exc
    _, _, Image, ImageDraw, _ = _visual_imports()
    rendered = Image.open(io.BytesIO(cairosvg.svg2png(url=str(path), output_width=672))).convert("RGB")
    rendered.thumbnail((672, 300), Image.Resampling.LANCZOS)
    row = Image.new("RGB", (672, 330), "white")
    ImageDraw.Draw(row).text((8, 7), title, fill="black")
    row.paste(rendered, ((672 - rendered.width) // 2, 30))
    return row


def _load_image(path: Path):
    nib, np, *_ = _visual_imports()
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) == 4:
        data = np.zeros(image.shape[:3], dtype=np.float32)
        for volume_index in range(image.shape[3]):
            data += np.asarray(image.dataobj[..., volume_index], dtype=np.float32)
        data /= max(1, image.shape[3])
    else:
        data = np.asarray(image.dataobj, dtype=np.float32)
    data[~np.isfinite(data)] = 0
    return data


def _image_metadata(path: Path, data) -> dict[str, Any]:
    nib, np, *_ = _visual_imports()
    image = nib.load(str(path))
    zooms = image.header.get_zooms()
    finite = np.isfinite(data)
    return {
        "path": str(path),
        "shape": [int(value) for value in image.shape],
        "voxel_size_mm": [round(float(value), 4) for value in zooms[:3]],
        "volumes": int(image.shape[3]) if len(image.shape) == 4 else 1,
        "nonzero_fraction": round(float(np.count_nonzero(finite & (data != 0)) / data.size), 6),
    }


def _normalize(data):
    _, np, *_ = _visual_imports()
    foreground = data[np.isfinite(data) & (data != 0)]
    if foreground.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(foreground, (1, 99))
    if high <= low:
        high = low + 1
    return (np.clip((data - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _slice_tile(volume, mask, axis: int, index: int, tile_size: int = 224):
    _, np, Image, *_ = _visual_imports()
    image_slice = np.take(volume, index, axis=axis)
    image_slice = np.rot90(image_slice)
    rgb = np.repeat(_normalize(image_slice)[..., None], 3, axis=2)
    if mask is not None and mask.shape == volume.shape:
        mask_slice = np.rot90(np.take(mask, index, axis=axis) > 0)
        edge = mask_slice & ~(
            np.roll(mask_slice, 1, 0)
            & np.roll(mask_slice, -1, 0)
            & np.roll(mask_slice, 1, 1)
            & np.roll(mask_slice, -1, 1)
        )
        rgb[edge] = (255, 70, 70)
    return Image.fromarray(rgb).resize((tile_size, tile_size), Image.Resampling.BILINEAR)


def _volume_row(volume, mask=None, title: str = ""):
    _, np, Image, ImageDraw, _ = _visual_imports()
    row = Image.new("RGB", (224 * 3, 254), "white")
    draw = ImageDraw.Draw(row)
    draw.text((8, 7), title, fill="black")
    for column, axis in enumerate((0, 1, 2)):
        if mask is not None and mask.shape == volume.shape and np.any(mask > 0):
            index = int(np.median(np.where(mask > 0)[axis]))
        else:
            index = volume.shape[axis] // 2
        row.paste(_slice_tile(volume, mask, axis, index), (column * 224, 30))
    return row


def _motion_row(confounds: Path | None, injected_values: list[float] | None = None):
    _, np, Image, ImageDraw, _ = _visual_imports()
    width, height = 224 * 3, 180
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "Motion: framewise displacement", fill="black")
    draw.line((42, 35, 42, height - 25), fill=(80, 80, 80), width=1)
    draw.line((42, height - 25, width - 15, height - 25), fill=(80, 80, 80), width=1)
    values: list[float] = list(injected_values or [])
    if not values and confounds and confounds.is_file():
        with confounds.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                try:
                    values.append(float(row.get("framewise_displacement", "")))
                except (TypeError, ValueError):
                    continue
    if values:
        upper = max(0.5, float(np.percentile(values, 99)))
        points = []
        for index, value in enumerate(values):
            x = 42 + index * (width - 60) / max(1, len(values) - 1)
            y = height - 25 - min(value, upper) * (height - 65) / upper
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=(38, 93, 160), width=2)
        threshold_y = height - 25 - min(0.5, upper) * (height - 65) / upper
        draw.line((42, threshold_y, width - 15, threshold_y), fill=(210, 55, 55), width=1)
        draw.text((47, 38), f"n={len(values)}, max={max(values):.3f} mm", fill="black")
    else:
        draw.text((55, 90), "No framewise-displacement values", fill=(120, 120, 120))
    return canvas


def _placeholder(title: str):
    _, _, Image, ImageDraw, _ = _visual_imports()
    row = Image.new("RGB", (224 * 3, 254), (245, 245, 245))
    ImageDraw.Draw(row).text((20, 110), title, fill=(110, 110, 110))
    return row


def _compose_packet(
    subject: str,
    anat_data,
    anat_mask_data,
    bold_data,
    bold_mask_data,
    confounds: Path | None,
    motion_values: list[float] | None = None,
    reportlet_rows: list[Any] | None = None,
):
    _, _, Image, ImageDraw, _ = _visual_imports()
    rows = [
        _volume_row(anat_data, anat_mask_data, "Anatomical image with brain-mask boundary")
        if anat_data is not None else _placeholder("Anatomical output unavailable"),
        _volume_row(bold_data, bold_mask_data, "Functional reference with brain-mask boundary")
        if bold_data is not None else _placeholder("Functional output unavailable"),
        _motion_row(confounds, motion_values),
    ]
    rows.extend(reportlet_rows or [])
    packet = Image.new("RGB", (224 * 3, 30 + sum(row.height for row in rows)), "white")
    ImageDraw.Draw(packet).text((8, 7), f"Visual QC packet: sub-{subject}", fill="black")
    y = 30
    for row in rows:
        packet.paste(row, (0, y))
        y += row.height
    return packet


def build_standalone_image_packet(
    sample_id: str,
    subject_id: str,
    image_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render one public source image without exposing its reference QC label."""
    _, _, Image, ImageDraw, _ = _visual_imports()
    data = _load_image(image_path)
    panel = _volume_row(data, title="Source image")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sample_id.replace("/", "_")
    panel_path = output_dir / f"{safe_id}_source.png"
    packet_path = output_dir / f"{safe_id}_qc_packet.png"
    panel.save(panel_path, format="PNG", optimize=True)
    packet = Image.new("RGB", (panel.width, panel.height + 30), "white")
    ImageDraw.Draw(packet).text((8, 7), f"Visual QC sample: {sample_id}", fill="black")
    packet.paste(panel, (0, 30))
    packet.save(packet_path, format="PNG", optimize=True)
    return {
        "schema_version": "1.1",
        "sample_id": sample_id,
        "subject": subject_id.removeprefix("sub-"),
        "packet_path": str(packet_path),
        "raw_panel_paths": [str(panel_path)],
        "panel_paths": [],
        "reportlet_panel_paths": [],
        "sources": {"raw_anat_image": str(image_path)},
        "image_metadata": {"source": _image_metadata(image_path, data)},
        "labels": {label: None for label in LABELS},
        "severity": None,
        "review": {"status": "unreviewed", "annotator": None, "note": None, "reviewed_at": None},
    }


def build_subject_packet(
    subject: str,
    fmriprep_dir: Path,
    output_dir: Path,
    bids_dir: Path | None = None,
) -> dict[str, Any]:
    subject_dir = fmriprep_dir / f"sub-{subject}"
    anat_dir = subject_dir / "anat"
    func_dir = subject_dir / "func"
    anat_image = next(iter(sorted(anat_dir.glob("*desc-preproc_T1w.nii.gz"))), None)
    anat_mask = next(iter(sorted(anat_dir.glob("*desc-brain_mask.nii.gz"))), None)
    bold_image = next(iter(sorted(func_dir.glob("*boldref.nii.gz"))), None)
    if bold_image is None:
        bold_image = next(iter(sorted(func_dir.glob("*desc-preproc_bold.nii.gz"))), None)
    bold_mask = next(iter(sorted(func_dir.glob("*desc-brain_mask.nii.gz"))), None)
    confounds = next(iter(sorted(func_dir.glob("*desc-confounds_timeseries.tsv"))), None)
    raw_subject_dir = bids_dir / f"sub-{subject}" if bids_dir else None
    raw_anat = next(iter(sorted(raw_subject_dir.glob("**/*_T1w.nii*"))), None) if raw_subject_dir else None
    raw_bold = next(iter(sorted(raw_subject_dir.glob("**/*_bold.nii*"))), None) if raw_subject_dir else None

    sources: dict[str, str | None] = {
        "raw_anat_image": str(raw_anat) if raw_anat else None,
        "raw_bold_image": str(raw_bold) if raw_bold else None,
        "anat_image": str(anat_image) if anat_image else None,
        "anat_mask": str(anat_mask) if anat_mask else None,
        "bold_image": str(bold_image) if bold_image else None,
        "bold_mask": str(bold_mask) if bold_mask else None,
        "confounds": str(confounds) if confounds else None,
    }
    anat_data = _load_image(anat_image) if anat_image else None
    anat_mask_data = _load_image(anat_mask) if anat_mask else None
    bold_data = _load_image(bold_image) if bold_image else None
    bold_mask_data = _load_image(bold_mask) if bold_mask else None
    raw_anat_data = _load_image(raw_anat) if raw_anat else None
    raw_bold_data = _load_image(raw_bold) if raw_bold else None
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = output_dir / f"sub-{subject}_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: list[str] = []
    raw_panel_paths: list[str] = []
    reportlet_panel_paths: list[str] = []
    raw_panels = (
        ("raw_anatomical", _volume_row(raw_anat_data, title="Raw T1w input")
         if raw_anat_data is not None else None),
        ("raw_functional", _volume_row(raw_bold_data, title="Raw BOLD temporal mean")
         if raw_bold_data is not None else None),
    )
    for panel_name, row in raw_panels:
        if row is None:
            continue
        row_path = panel_dir / f"{panel_name}.png"
        row.save(row_path, format="PNG", optimize=True)
        raw_panel_paths.append(str(row_path))
    base_panels = (
        ("anatomical_mask", _volume_row(anat_data, anat_mask_data, "Anatomical image with brain-mask boundary")
         if anat_data is not None else _placeholder("Anatomical output unavailable")),
        ("functional_mask", _volume_row(bold_data, bold_mask_data, "Functional reference with brain-mask boundary")
         if bold_data is not None else _placeholder("Functional output unavailable")),
        ("motion", _motion_row(confounds)),
    )
    for panel_name, row in base_panels:
        row_path = panel_dir / f"{panel_name}.png"
        row.save(row_path, format="PNG", optimize=True)
        panel_paths.append(str(row_path))
    reportlet_specs = (
        ("segmentation", "*dseg.svg", "Tissue segmentation"),
        ("normalization", "*space-*_T1w.svg", "T1w to standard-space normalization"),
        ("coregistration", "*desc-coreg_bold.svg", "BOLD to T1w coregistration"),
        ("functional_rois", "*desc-rois_bold.svg", "Functional coverage and masks"),
        ("carpetplot", "*desc-carpetplot_bold.svg", "Functional carpet plot"),
    )
    reportlet_rows = []
    for panel_name, pattern, title in reportlet_specs:
        reportlet = next(iter(sorted((subject_dir / "figures").glob(pattern))), None)
        if reportlet:
            row = _render_reportlet(reportlet, title)
            row_path = panel_dir / f"{panel_name}.png"
            row.save(row_path, format="PNG", optimize=True)
            reportlet_rows.append(row)
            panel_paths.append(str(row_path))
            reportlet_panel_paths.append(str(row_path))
    packet = _compose_packet(
        subject,
        anat_data,
        anat_mask_data,
        bold_data,
        bold_mask_data,
        confounds,
        reportlet_rows=reportlet_rows,
    )
    packet_path = output_dir / f"sub-{subject}_qc_packet.png"
    packet.save(packet_path, format="PNG", optimize=True)
    reportlets = sorted((subject_dir / "figures").glob("*.svg"))
    image_metadata = {}
    for name, path, data in (
        ("raw_t1w", raw_anat, raw_anat_data),
        ("raw_bold", raw_bold, raw_bold_data),
        ("preprocessed_t1w", anat_image, anat_data),
        ("preprocessed_bold_reference", bold_image, bold_data),
    ):
        if path is not None and data is not None:
            image_metadata[name] = _image_metadata(path, data)
    return {
        "schema_version": "1.1",
        "subject": subject,
        "packet_path": str(packet_path),
        "raw_panel_paths": raw_panel_paths,
        "panel_paths": panel_paths,
        "reportlet_panel_paths": reportlet_panel_paths,
        "sources": sources,
        "image_metadata": image_metadata,
        "fmriprep_report": str(fmriprep_dir / f"sub-{subject}.html"),
        "reportlets": [str(path) for path in reportlets],
        "labels": {label: None for label in LABELS},
        "severity": None,
        "review": {"status": "unreviewed", "annotator": None, "note": None, "reviewed_at": None},
    }


def generate_synthetic_dataset(manifest_path: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    _, np, *_ = _visual_imports()
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for record in records:
        sources = record["sources"]
        anat = _load_image(Path(sources["anat_image"])) if sources.get("anat_image") else None
        anat_mask = _load_image(Path(sources["anat_mask"])) if sources.get("anat_mask") else None
        bold = _load_image(Path(sources["bold_image"])) if sources.get("bold_image") else None
        bold_mask = _load_image(Path(sources["bold_mask"])) if sources.get("bold_mask") else None
        variants: list[tuple[str, Any, Any, Any, Any, list[float] | None]] = []

        if anat is not None and anat_mask is not None:
            shift = max(2, anat_mask.shape[0] // 12)
            variants.append((
                "brain_mask_error", anat, np.roll(anat_mask, shift, axis=0), bold, bold_mask, None,
            ))
        image_for_coverage = anat if anat is not None else bold
        if image_for_coverage is not None:
            damaged = image_for_coverage.copy()
            damaged[..., int(damaged.shape[2] * 0.8):] = 0
            variants.append((
                "coverage_error",
                damaged if anat is not None else anat,
                anat_mask,
                damaged if anat is None else bold,
                bold_mask,
                None,
            ))
        if bold is not None:
            shift = max(2, bold.shape[1] // 10)
            ghosted = bold + 0.35 * np.roll(bold, shift, axis=1)
            dropout = ghosted.copy()
            dropout[:, : max(1, dropout.shape[1] // 6), :] *= 0.1
            variants.append(("ghosting_dropout", anat, anat_mask, dropout, bold_mask, None))
            variants.append((
                "coregistration_error",
                anat,
                anat_mask,
                np.roll(bold, max(2, bold.shape[0] // 10), axis=0),
                bold_mask,
                None,
            ))
        motion = np.abs(rng.normal(0.15, 0.08, 180))
        motion[rng.choice(len(motion), size=18, replace=False)] += rng.uniform(0.8, 2.0, 18)
        variants.append(("severe_motion", anat, anat_mask, bold, bold_mask, motion.tolist()))

        for label, variant_anat, variant_anat_mask, variant_bold, variant_bold_mask, motion_values in variants:
            subject = str(record["subject"])
            synthetic_id = f"sub-{subject}_{label}"
            output_path = output_dir / f"{synthetic_id}_qc_packet.png"
            if label in {"brain_mask_error", "coverage_error"} and variant_anat is not None:
                panel = _volume_row(variant_anat, variant_anat_mask, f"Synthetic {label}: sub-{subject}")
                panel_type = "anatomical_mask"
            elif label == "severe_motion":
                panel = _motion_row(None, motion_values)
                panel_type = "motion"
            else:
                panel = _volume_row(variant_bold, variant_bold_mask, f"Synthetic {label}: sub-{subject}")
                panel_type = "functional_mask"
            panel.save(output_path, format="PNG", optimize=True)
            generated.append({
                "schema_version": "1.0",
                "sample_id": synthetic_id,
                "subject": subject,
                "packet_path": str(output_path),
                "source_packet": record["packet_path"],
                "synthetic": True,
                "synthetic_transform": label,
                "panel_type": panel_type,
                "simulation_scope": "component_proxy",
                "labels": {name: name == label for name in LABELS},
                "severity": "fail",
                "review": {"status": "synthetic", "annotator": "generator", "note": None},
            })
    output_manifest = output_dir / "synthetic_manifest.jsonl"
    atomic_write_text(output_manifest, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in generated))
    return {"manifest_path": str(output_manifest), "samples": len(generated), "seed": seed}


def record_visual_review(
    labels_path: Path,
    sample: dict[str, Any],
    decision: str,
    labels: list[str],
    annotator: str,
    note: str,
    severity: str | None = None,
    exclude_subject: bool | None = None,
    anatomical_input_state: str = "unknown",
) -> dict[str, Any]:
    if decision not in {"pass", "uncertain", "fail"}:
        raise ValueError("decision must be pass, uncertain or fail")
    if not annotator.strip():
        raise ValueError("annotator cannot be empty")
    if decision != "pass" and not note.strip():
        raise ValueError("uncertain and fail reviews require a note")
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError("Unknown visual QC labels: " + ", ".join(unknown))
    if decision == "pass" and labels:
        raise ValueError("pass reviews cannot contain abnormal labels")
    if decision in {"uncertain", "fail"} and not labels:
        raise ValueError("uncertain and fail reviews require at least one abnormal label")
    if severity not in {None, "low", "medium", "high"}:
        raise ValueError("severity must be low, medium, high or null")
    if anatomical_input_state not in ANATOMICAL_INPUT_STATES:
        raise ValueError(
            "anatomical_input_state must be one of: " + ", ".join(ANATOMICAL_INPUT_STATES)
        )
    if labels and severity is None:
        severity = "high" if decision == "fail" else "medium"
    if not labels:
        severity = None
    if exclude_subject is None:
        exclude_subject = decision == "fail"
    if decision == "pass" and exclude_subject:
        raise ValueError("pass reviews cannot exclude a subject")
    record = {
        "schema_version": "1.2",
        "sample_id": sample.get("sample_id") or sample.get("subject"),
        "subject": sample.get("subject"),
        "packet_path": sample.get("packet_path"),
        "decision": decision,
        "labels": {name: name in labels for name in LABELS},
        "severity": severity,
        "exclude_subject": bool(exclude_subject),
        "anatomical_input_state": anatomical_input_state,
        "annotator": annotator.strip(),
        "note": note.strip(),
        "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "synthetic": bool(sample.get("synthetic")),
    }
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with advisory_lock(labels_path.with_suffix(labels_path.suffix + ".lock")):
        existing = labels_path.read_text(encoding="utf-8") if labels_path.exists() else ""
        atomic_write_text(labels_path, existing + json.dumps(record, ensure_ascii=False) + "\n")
    return record


@lru_cache(maxsize=2)
def _load_dinov2(model_id: str, device: str, local_files_only: bool):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("DINOv2 requires torch and transformers. Install .[visual-model].") from exc
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only).to(device).eval()
    return torch, processor, model


def _dinov2_embedding(image_paths: list[Path], settings: dict[str, Any]):
    _, np, Image, *_ = _visual_imports()
    device = settings.get("device", "cuda:0")
    torch, processor, model = _load_dinov2(
        settings["model_id"], device, not bool(settings.get("allow_model_download")),
    )
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = {key: value.to(device) for key, value in processor(images=images, return_tensors="pt").items()}
    with torch.inference_mode():
        embedding = model(**inputs).last_hidden_state[:, 0].float().cpu().numpy().mean(axis=0)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm else embedding


@lru_cache(maxsize=2)
def _load_qwen_vl(
    model_id: str,
    device: str,
    local_files_only: bool,
    min_pixels: int,
    max_pixels: int,
):
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Qwen2.5-VL requires torch and transformers. Install .[visual-model].") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable but qc.visual.device={device}")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_fast=False,
    )
    return torch, processor, model


def _parse_visual_assessment(text: str) -> VisualAssessment:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```json").removeprefix("```")
        candidate = candidate.rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM response does not contain a JSON object") from None
        payload = json.loads(candidate[start : end + 1])
    return VisualAssessment.model_validate(payload)


def _packet_context(record: dict[str, Any]) -> str:
    panel_names = [
        Path(path).stem
        for path in (record.get("panel_paths") or record.get("raw_panel_paths", []))
    ]
    context = ["Panel order: " + ", ".join(panel_names)]
    confounds_value = record.get("sources", {}).get("confounds")
    confounds = Path(confounds_value) if confounds_value else None
    values: list[float] = []
    if confounds and confounds.is_file():
        with confounds.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                try:
                    values.append(float(row.get("framewise_displacement", "")))
                except (TypeError, ValueError):
                    continue
    if values:
        threshold = 0.5
        context.append(
            "Framewise displacement: "
            f"mean={sum(values) / len(values):.3f} mm, max={max(values):.3f} mm, "
            f"fraction_above_{threshold:.1f}mm={sum(value > threshold for value in values) / len(values):.3f}."
        )
    return " ".join(context)


def _qwen_vl_assessment(
    image_paths: list[Path],
    settings: dict[str, Any],
    context: str = "",
) -> dict[str, Any]:
    _, _, Image, *_ = _visual_imports()
    selected_paths = image_paths[: int(settings.get("max_panels", 8))]
    if not selected_paths:
        raise ValueError("No visual QC panels were provided")
    torch, processor, model = _load_qwen_vl(
        settings["model_id"],
        settings.get("device", "cuda:0"),
        not bool(settings.get("allow_model_download")),
        int(settings.get("min_pixels", 200704)),
        int(settings.get("max_pixels", 401408)),
    )
    images = [Image.open(path).convert("RGB") for path in selected_paths]
    if settings.get("review_target") == "source_image":
        role = (
            "You are a conservative technical quality-control reviewer for source MRI images. "
            "The panels are orthogonal source-image views without masks, contours, segmentations, or registration "
            "overlays. Assess only visible acquisition defects: severe_motion, ghosting_dropout, and coverage_error. "
            "Do not report brain_mask_error, normalization_error, coregistration_error, or segmentation_error because "
            "those operations are not shown. Facial defacing or missing facial features are expected and are not QC "
            "failures. Distinguish genuine ringing, ghosting, duplicated edges, and motion blur from normal anatomy. "
        )
        allowed = "Allowed finding categories are: coverage_error, ghosting_dropout, severe_motion. "
        prompt_version = "source-mri-visual-qc-v1"
    else:
        role = (
            "You are a conservative technical quality-control reviewer for fMRIPrep outputs. "
            "Inspect the labeled panels in order and identify only visible preprocessing failures. "
            "Red contours on anatomical and functional slices are expected brain-mask overlays; the presence of "
            "a red contour alone is not an error. Judge whether the contour follows the visible brain boundary. "
            "For severe_motion, use the supplied numeric framewise-displacement summary as authoritative and do "
            "not infer severe motion merely because the plot contains peaks. "
        )
        allowed = (
            "Allowed finding categories are: brain_mask_error, normalization_error, coregistration_error, "
            "coverage_error, ghosting_dropout, segmentation_error, severe_motion. "
        )
        prompt_version = "fmriprep-visual-qc-v2"
    prompt = (
        role
        + "Do not make a clinical diagnosis. If evidence is ambiguous, choose review rather than pass or fail. "
        + allowed
        + "Return exactly one JSON object with this schema and no markdown: "
        '{"decision":"pass|review|fail","confidence":0.0,'
        '"observations":[{"category":"allowed_category","abnormal":true,'
        '"severity":"low|medium|high","evidence":"visible evidence and panel name"}],'
        '"summary":"brief technical conclusion"}. '
        "Include only categories for which the panels or numeric context provide useful evidence. Set abnormal=false "
        "when the evidence supports a normal result. Set abnormal=true only for an actual or suspected defect. "
        "Use decision=pass when every observation has abnormal=false or observations is empty; use review for an "
        "ambiguous defect and fail only for a clear severe technical defect. Do not invent overlays or evidence "
        "that is absent from the supplied image. "
        f"Case context: {context or 'No additional numeric context.'}"
    )
    content: list[dict[str, str]] = []
    for index, path in enumerate(selected_paths, start=1):
        content.extend((
            {"type": "text", "text": f"Panel {index}: {path.stem}"},
            {"type": "image"},
        ))
    content.append({"type": "text", "text": prompt})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[rendered], images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(settings.get("device", "cuda:0")) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(settings.get("max_new_tokens", 384)),
            do_sample=False,
        )
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs["input_ids"], generated)]
    raw_response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    try:
        assessment = _parse_visual_assessment(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{exc}; raw_response={raw_response[:2000]}") from exc
    parsed = assessment.model_dump()
    abnormal = [item for item in parsed["observations"] if item["abnormal"]]
    decision = "fail" if any(item["severity"] == "high" for item in abnormal) else ("review" if abnormal else "pass")
    findings = [
        {
            "category": item["category"],
            "severity": item["severity"],
            "evidence": item["evidence"],
        }
        for item in abnormal
    ]
    return {
        **parsed,
        "decision": decision,
        "model_decision": parsed["decision"],
        "decision_adjusted": decision != parsed["decision"],
        "findings": findings,
        "raw_response": raw_response,
        "panels_scored": [str(path) for path in selected_paths],
        "prompt_version": prompt_version,
    }


def embed_manifest(
    manifest_path: Path,
    output_path: Path,
    model_id: str = "facebook/dinov2-small",
    device: str = "cuda:0",
    allow_model_download: bool = False,
) -> dict[str, Any]:
    _, np, *_ = _visual_imports()
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    embedding_dir = output_path.parent / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "model_id": model_id,
        "device": device,
        "allow_model_download": allow_model_download,
    }
    for index, record in enumerate(records):
        image_paths = [Path(path) for path in record.get("panel_paths", [])]
        if not image_paths:
            image_paths = [Path(path) for path in record.get("raw_panel_paths", [])]
        if not image_paths:
            image_paths = [Path(record["packet_path"])]
        embedding = _dinov2_embedding(image_paths, settings)
        sample_id = str(record.get("sample_id") or record.get("subject") or index)
        safe_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
        embedding_path = embedding_dir / f"{safe_id}_dinov2.npy"
        np.save(embedding_path, embedding)
        record.update({
            "embedding_path": str(embedding_path),
            "embedding_model": model_id,
            "panels_scored": len(image_paths),
        })
    atomic_write_text(output_path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
    return {"manifest_path": str(output_path), "samples": len(records), "model_id": model_id, "device": device}


def assess_manifest_with_vlm(
    manifest_path: Path,
    output_path: Path,
    model_id: str,
    device: str = "cuda:0",
    allow_model_download: bool = False,
    max_new_tokens: int = 384,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
    max_panels: int = 8,
) -> dict[str, Any]:
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    settings = {
        "model_id": model_id,
        "device": device,
        "allow_model_download": allow_model_download,
        "max_new_tokens": max_new_tokens,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_panels": max_panels,
    }
    results: list[dict[str, Any]] = []
    for record in records:
        image_paths = [Path(path) for path in record.get("panel_paths", [])]
        review_target = "fmriprep_output"
        if not image_paths and record.get("raw_panel_paths"):
            image_paths = [Path(path) for path in record["raw_panel_paths"]]
            review_target = "source_image"
        if not image_paths:
            image_paths = [Path(record["packet_path"])]
        result: dict[str, Any] = {
            "sample_id": record.get("sample_id") or record.get("subject"),
            "subject": record.get("subject"),
            "model_id": model_id,
            "mode": "shadow",
            "affects_gate": False,
        }
        try:
            record_settings = {**settings, "review_target": review_target}
            result.update({
                "status": "completed",
                **_qwen_vl_assessment(image_paths, record_settings, _packet_context(record)),
            })
        except Exception as exc:  # noqa: BLE001 - one malformed sample must not erase the batch
            result.update({"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
        results.append(result)
        atomic_write_text(output_path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results))
    return {
        "output_path": str(output_path),
        "samples": len(results),
        "completed": sum(item["status"] == "completed" for item in results),
        "unavailable": sum(item["status"] != "completed" for item in results),
        "model_id": model_id,
        "device": device,
        "affects_gate": False,
    }


def score_vlm_predictions(
    manifest_path: Path,
    predictions_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    samples = {
        str(item.get("sample_id") or item.get("subject")): item
        for item in (
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    predictions = {
        str(item.get("sample_id") or item.get("subject")): item
        for item in (
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    tp = fp = fn = exact = completed = labeled_samples = 0
    cases: list[dict[str, Any]] = []
    for sample_id, sample in samples.items():
        labels = sample.get("labels", {})
        if not any(isinstance(value, bool) for value in labels.values()):
            continue
        labeled_samples += 1
        truth = {label for label, value in labels.items() if value is True}
        prediction = predictions.get(sample_id, {})
        predicted = {item["category"] for item in prediction.get("findings", [])}
        available = prediction.get("status") == "completed"
        if available:
            completed += 1
            tp += len(truth & predicted)
            fp += len(predicted - truth)
            fn += len(truth - predicted)
            exact += truth == predicted
        cases.append({
            "sample_id": sample_id,
            "available": available,
            "truth": sorted(truth),
            "predicted": sorted(predicted),
            "exact_match": available and truth == predicted,
        })
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "schema_version": "1.0",
        "manifest_path": str(manifest_path),
        "predictions_path": str(predictions_path),
        "samples": len(samples),
        "labeled_samples": labeled_samples,
        "completed": completed,
        "coverage": completed / labeled_samples if labeled_samples else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_match_rate": exact / completed if completed else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "cases": cases,
    }
    if output_path:
        atomic_write_text(output_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run_visual_qc_shadow(state: dict[str, Any]) -> dict[str, Any]:
    settings = dict(state["config"]["qc"].get("visual", {}))
    if not settings.get("enabled"):
        return {"status": "disabled", "mode": "shadow", "subjects": []}
    if state["config"]["runtime"]["mode"] != "run":
        return {"status": "planned", "mode": "shadow", "subjects": []}

    root = state["config"].get("project_root")
    model_candidate = project_path(settings.get("model_id", ""), root)
    if model_candidate.is_dir():
        settings["model_id"] = str(model_candidate)
    packet_root = project_path(settings["packet_dir"], root) / state.get("run_id", "standalone")
    fmriprep_dir = Path(state["processed_data"]["fmriprep_dir"])
    results: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    try:
        for subject in state["processed_data"].get("subjects", []):
            bids_value = state["processed_data"].get("bids_dir")
            bids_dir = Path(bids_value) if bids_value else None
            record = build_subject_packet(subject, fmriprep_dir, packet_root, bids_dir)
            manifest.append(record)
            prediction: dict[str, Any] = {
                "subject": subject,
                "packet_path": record["packet_path"],
                "model_id": settings.get("model_id"),
                "decision": "unscored",
            }
            if settings.get("model_backend") == "dinov2_embedding":
                image_paths = [Path(path) for path in record.get("panel_paths", [])]
                if not image_paths:
                    image_paths = [Path(record["packet_path"])]
                embedding = _dinov2_embedding(image_paths, settings)
                embedding_path = packet_root / f"sub-{subject}_dinov2.npy"
                _, np, *_ = _visual_imports()
                np.save(embedding_path, embedding)
                prediction["embedding_path"] = str(embedding_path)
                prediction["panels_scored"] = len(image_paths)
                record["embedding_path"] = str(embedding_path)
                record["embedding_model"] = settings["model_id"]
                reference_value = settings.get("reference_path")
                reference_path = project_path(reference_value, root) if reference_value else None
                if reference_path and reference_path.is_file():
                    reference = np.load(reference_path, allow_pickle=False)
                    reference_model = str(reference["model_id"]) if "model_id" in reference else None
                    if reference_model and reference_model != settings["model_id"]:
                        raise ValueError(
                            f"Reference model mismatch: {reference_model} != {settings['model_id']}"
                        )
                    centroid = reference["centroid"]
                    score = float(1 - np.dot(embedding, centroid) / (np.linalg.norm(centroid) or 1))
                    threshold = settings.get("anomaly_threshold")
                    if threshold is None and "threshold" in reference:
                        threshold = float(reference["threshold"])
                    prediction.update({
                        "anomaly_score": score,
                        "threshold": threshold,
                        "decision": "suspected_anomaly" if threshold is not None and score > threshold else "pass_candidate",
                    })
                else:
                    prediction["status"] = "needs_reference"
            elif settings.get("model_backend") == "qwen2_5_vl":
                image_paths = [Path(path) for path in record.get("panel_paths", [])]
                if not image_paths:
                    image_paths = [Path(record["packet_path"])]
                try:
                    assessment = _qwen_vl_assessment(image_paths, settings, _packet_context(record))
                    prediction.update({"status": "completed", **assessment})
                    record["vlm_assessment"] = assessment
                except Exception as exc:  # noqa: BLE001 - preserve the rest of the shadow batch
                    prediction.update({
                        "status": "unavailable",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            else:
                prediction["status"] = "packet_only"
            results.append(prediction)
    except Exception as exc:  # noqa: BLE001 - shadow inference must never break deterministic QC
        return {
            "status": "unavailable",
            "mode": "shadow",
            "error": f"{type(exc).__name__}: {exc}",
            "subjects": results,
        }

    manifest_path = packet_root / "manifest.jsonl"
    atomic_write_text(manifest_path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest))
    return {
        "status": "completed",
        "mode": "shadow",
        "model_backend": settings.get("model_backend"),
        "model_id": settings.get("model_id"),
        "manifest_path": str(manifest_path),
        "subjects": results,
        "affects_gate": False,
    }


def fit_normal_reference(
    manifest_path: Path,
    labels_path: Path,
    output_path: Path,
    percentile: float = 99.0,
) -> dict[str, Any]:
    _, np, *_ = _visual_imports()
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reviews = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest_review = {str(item.get("sample_id")): item for item in reviews}
    selected = [
        item
        for item in records
        if item.get("embedding_path")
        and latest_review.get(str(item.get("sample_id") or item.get("subject")), {}).get("decision") == "pass"
        and not item.get("synthetic")
    ]
    paths = [Path(item["embedding_path"]) for item in selected]
    if len(paths) < 2:
        raise ValueError("At least two human-reviewed normal embeddings are required")
    model_ids = {str(item.get("embedding_model")) for item in selected}
    if len(model_ids) != 1 or "None" in model_ids:
        raise ValueError("Normal reference embeddings must use one recorded model")
    model_id = model_ids.pop()
    embeddings = np.stack([np.load(path, allow_pickle=False) for path in paths])
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1
    scores = 1 - embeddings @ centroid
    threshold = float(np.percentile(scores, percentile))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        centroid=centroid,
        threshold=threshold,
        sample_count=len(paths),
        percentile=percentile,
        model_id=model_id,
    )
    return {
        "reference_path": str(output_path),
        "sample_count": len(paths),
        "threshold": threshold,
        "labels_path": str(labels_path),
        "model_id": model_id,
    }
