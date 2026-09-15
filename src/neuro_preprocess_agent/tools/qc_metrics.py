from __future__ import annotations

from pathlib import Path
from typing import Any


def compute_quantitative_metrics(
    subject: str,
    images: dict[str, Path],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not settings.get("enabled", True):
        return {"status": "disabled", "subject": subject, "images": {}, "checks": []}
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        return {"status": "unavailable", "subject": subject, "error": str(exc), "images": {}, "checks": []}

    metrics: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    arrays: dict[str, Any] = {}
    affines: dict[str, Any] = {}
    loaded_images: dict[str, Any] = {}

    for name, path in images.items():
        if not path.is_file():
            continue
        try:
            image = nib.load(path)
            loaded_images[name] = image
            data = image.get_fdata(dtype=np.float32)
            arrays[name] = data
            affines[name] = image.affine
            finite = np.isfinite(data)
            finite_count = int(finite.sum())
            total_count = int(data.size)
            finite_values = data[finite]
            stride = max(1, finite_values.size // 1_000_000)
            sample = finite_values[::stride]
            item = {
                "path": str(path),
                "shape": list(data.shape),
                "zooms": [float(value) for value in image.header.get_zooms()[: data.ndim]],
                "nonfinite_fraction": 1.0 - (finite_count / total_count if total_count else 0.0),
                "zero_fraction": float(np.count_nonzero(data == 0) / total_count) if total_count else 1.0,
                "intensity_percentiles": (
                    {str(level): float(value) for level, value in zip((1, 50, 99), np.percentile(sample, (1, 50, 99)))}
                    if sample.size else {}
                ),
            }
            metrics[name] = item
            checks.append({
                "name": f"{subject}:nonfinite_fraction:{name}",
                "passed": item["nonfinite_fraction"] <= float(settings["max_nonfinite_fraction"]),
                "observed": item["nonfinite_fraction"],
                "threshold": settings["max_nonfinite_fraction"],
                "severity": "error",
            })
        except Exception as exc:  # noqa: BLE001 - malformed image details are returned to QC
            metrics[name] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    for name in ("t1w_brain_mask", "t1w_standard_brain_mask", "bold_brain_mask"):
        data = arrays.get(name)
        if data is None:
            continue
        mask = np.isfinite(data) & (data > 0)
        voxel_count = int(mask.sum())
        spatial = mask[..., 0] if mask.ndim == 4 else mask
        edge = np.zeros_like(spatial, dtype=bool)
        edge[[0, -1], :, :] = True
        edge[:, [0, -1], :] = True
        edge[:, :, [0, -1]] = True
        edge_fraction = float((spatial & edge).sum() / voxel_count) if voxel_count else 1.0
        zooms = metrics[name].get("zooms", [1, 1, 1])
        metrics[name].update({
            "mask_voxels": voxel_count,
            "mask_volume_mm3": float(voxel_count * zooms[0] * zooms[1] * zooms[2]),
            "mask_fraction": float(voxel_count / spatial.size),
            "mask_edge_fraction": edge_fraction,
        })
        checks.append({
            "name": f"{subject}:minimum_mask_voxels:{name}",
            "passed": voxel_count >= int(settings["min_mask_voxels"]),
            "observed": voxel_count,
            "threshold": settings["min_mask_voxels"],
            "severity": "error",
        })
        if settings.get("max_mask_edge_fraction") is not None:
            checks.append({
                "name": f"{subject}:mask_edge_fraction:{name}",
                "passed": edge_fraction <= float(settings["max_mask_edge_fraction"]),
                "observed": edge_fraction,
                "threshold": settings["max_mask_edge_fraction"],
                "severity": "warning",
            })

    bold = arrays.get("preproc_bold")
    bold_mask = arrays.get("bold_brain_mask")
    if bold is not None and bold.ndim == 4 and bold_mask is not None and bold.shape[:3] == bold_mask.shape[:3]:
        mask = np.isfinite(bold_mask) & (bold_mask > 0)
        finite_bold = np.where(np.isfinite(bold), bold, np.nan)
        temporal_mean = np.nanmean(finite_bold, axis=3)
        temporal_std = np.nanstd(finite_bold, axis=3)
        valid = mask & np.isfinite(temporal_mean) & np.isfinite(temporal_std) & (temporal_std > 0)
        tsnr_values = temporal_mean[valid] / temporal_std[valid]
        tsnr = float(np.median(tsnr_values)) if tsnr_values.size else None
        metrics["bold_tsnr"] = {"median": tsnr, "valid_voxels": int(valid.sum())}
        if settings.get("min_bold_tsnr") is not None:
            checks.append({
                "name": f"{subject}:bold_tsnr",
                "passed": tsnr is not None and tsnr >= float(settings["min_bold_tsnr"]),
                "observed": tsnr,
                "threshold": settings["min_bold_tsnr"],
                "severity": "warning",
            })

    anat_mask_name = "t1w_standard_brain_mask" if "t1w_standard_brain_mask" in arrays else "t1w_brain_mask"
    anat_mask = arrays.get(anat_mask_name)
    if anat_mask is not None and bold_mask is not None:
        same_grid = anat_mask.shape[:3] == bold_mask.shape[:3] and np.allclose(
            affines[anat_mask_name], affines["bold_brain_mask"], atol=1e-3
        )
        comparable_bold_mask = bold_mask
        comparison = "same_grid"
        if not same_grid and anat_mask_name == "t1w_standard_brain_mask":
            try:
                from nibabel.processing import resample_from_to

                comparable_bold_mask = resample_from_to(
                    loaded_images["bold_brain_mask"],
                    (anat_mask.shape[:3], affines[anat_mask_name]),
                    order=0,
                ).get_fdata(dtype=np.float32)
                same_grid = True
                comparison = "bold_resampled_to_standard_t1w"
            except (ImportError, ValueError):
                pass
        if same_grid:
            anat_binary = np.isfinite(anat_mask) & (anat_mask > 0)
            bold_binary = np.isfinite(comparable_bold_mask) & (comparable_bold_mask > 0)
            denominator = int(anat_binary.sum() + bold_binary.sum())
            dice = float(2 * (anat_binary & bold_binary).sum() / denominator) if denominator else 0.0
            metrics["anat_bold_mask_overlap"] = {
                "status": "computed",
                "dice": dice,
                "comparison": comparison,
                "anatomical_mask": anat_mask_name,
            }
            if settings.get("min_anat_bold_mask_dice") is not None:
                checks.append({
                    "name": f"{subject}:anat_bold_mask_dice",
                    "passed": dice >= float(settings["min_anat_bold_mask_dice"]),
                    "observed": dice,
                    "threshold": settings["min_anat_bold_mask_dice"],
                    "severity": "warning",
                })
        else:
            metrics["anat_bold_mask_overlap"] = {
                "status": "not_comparable",
                "reason": "masks are not on the same voxel grid",
            }

    return {"status": "completed", "subject": subject, "images": metrics, "checks": checks}
