from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RuntimeMode = Literal["mock", "dry_run", "run"]
SourceType = Literal["openneuro", "url_file", "local_path"]
PermissionMode = Literal["allow", "ask", "deny"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(StrictModel):
    mode: RuntimeMode = "mock"


class PreflightConfig(StrictModel):
    enabled: bool = True
    minimum_free_disk_gb: float = Field(default=20.0, ge=0)
    check_database: bool = True
    check_docker: bool = True
    check_resource_budget: bool = True
    require_warning_approval: bool = True


class SupervisorConfig(StrictModel):
    mode: Literal["rule", "llm"] = "rule"
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str | None = None
    temperature: float = 0
    fallback_to_rule: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)


class SourceConfig(StrictModel):
    name: str = "OpenNeuro dataset"
    type: SourceType = "openneuro"
    dataset_id: str | None = None
    url: str | None = None
    path: str | None = None
    uri: str | None = None
    tag: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    max_concurrency: int = Field(default=5, ge=1, le=32)
    reuse_existing: bool = True

    @model_validator(mode="after")
    def validate_reference(self) -> SourceConfig:
        required = {
            "openneuro": self.dataset_id,
            "url_file": self.url,
            "local_path": self.path,
        }
        if not required[self.type]:
            field = {"openneuro": "dataset_id", "url_file": "url", "local_path": "path"}[self.type]
            raise ValueError(f"source.{field} is required for source.type={self.type}")
        return self


class StorageConfig(StrictModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    report_dir: str = "runs/reports"
    copy_local_data: bool = True


class DicomJobConfig(StrictModel):
    participant_label: str = Field(pattern=r"^[A-Za-z0-9]+$")
    session_label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]+$")
    dicom_dirs: list[str] = Field(min_length=1)
    dcm2bids_config: str | None = None


class NiftiItemConfig(StrictModel):
    path: str
    participant_label: str = Field(pattern=r"^[A-Za-z0-9]+$")
    session_label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]+$")
    datatype: Literal["anat", "func"]
    suffix: Literal["T1w", "bold"]
    task: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]+$")
    run: int | None = Field(default=None, ge=1)
    acquisition: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]+$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bids_entities(self) -> NiftiItemConfig:
        if self.datatype == "func" and (self.suffix != "bold" or not self.task):
            raise ValueError("functional NIfTI items require suffix=bold and a task")
        if self.datatype == "anat" and self.suffix != "T1w":
            raise ValueError("anatomical NIfTI items currently support suffix=T1w only")
        return self


class FMRIPrepOptions(StrictModel):
    output_layout: Literal["bids", "legacy"] = "bids"
    output_spaces: list[str] = Field(default_factory=list)
    nprocs: int | None = Field(default=None, ge=1)
    omp_nthreads: int | None = Field(default=None, ge=1)
    memory_mb: int | None = Field(default=None, ge=1024)
    low_mem: bool = False
    anat_only: bool = False
    level: Literal["minimal", "resampling", "full"] = "full"
    ignore: list[Literal["fieldmaps", "slicetiming", "sbref", "t2w", "flair", "fmap-jacobian"]] = Field(default_factory=list)
    force: list[Literal["bbr", "no-bbr", "syn-sdc", "fmap-jacobian"]] = Field(default_factory=list)
    subject_anatomical_reference: Literal["first-lex", "unbiased", "sessionwise"] = "first-lex"
    skull_strip_t1w: Literal["auto", "skip", "force"] = "auto"
    fs_no_reconall: bool = True
    skip_bids_validation: bool = False
    stop_on_first_crash: bool = True


class PreprocessConfig(StrictModel):
    input_format: Literal["auto", "bids", "dicom", "nifti"] = "auto"
    assume_format: Literal["bids", "dicom", "nifti"] = "bids"
    bids_dir: str = "data/bids"
    bids_dataset_name: str = "Dataset prepared by neuro-preprocess-agent"
    derivatives_dir: str = "data/derivatives/fmriprep"
    log_dir: str = "runs/logs/fmriprep"
    work_dir: str = "data/work/fmriprep"
    lock_dir: str = "runs/locks/fmriprep"
    dcm2bids: str = "dcm2bids"
    dcm2bids_config: str = "configs/dcm2bids.example.json"
    freesurfer_license: str = "data/private/license.txt"
    fmriprep_image: str = "nipreps/fmriprep:latest"
    run_fmriprep: bool = True
    max_workers: int = Field(default=2, ge=1, le=32)
    participant_label: str = Field(default="001", pattern=r"^[A-Za-z0-9]+$")
    subjects: list[str] = Field(default_factory=list)
    dicom_jobs: list[DicomJobConfig] = Field(default_factory=list)
    dicom_header_validation: bool = True
    reuse_completed: bool = True
    validate_bids: bool = True
    isolate_converted_bids: bool = True
    input_qc_enabled: bool = True
    input_qc_require_review: bool = True
    pre_skull_stripped_nonzero_fraction: float = Field(default=0.3, gt=0, lt=1)
    assume_modalities: list[Literal["anat", "func"]] = Field(default_factory=lambda: ["anat", "func"])
    nifti_subject_label: str = Field(default="001", pattern=r"^[A-Za-z0-9]+$")
    nifti_session_label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]+$")
    nifti_task_name: str = Field(default="rest", pattern=r"^[A-Za-z0-9]+$")
    nifti_metadata: dict[str, Any] = Field(default_factory=dict)
    nifti_manifest: list[NiftiItemConfig] = Field(default_factory=list)
    fmriprep_options: FMRIPrepOptions = Field(default_factory=FMRIPrepOptions)
    fmriprep_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fmriprep_extra_args(self) -> PreprocessConfig:
        protected = {
            "--participant-label", "--participant_label", "--session-label", "--work-dir", "-w",
            "--fs-license-file", "--output-layout", "--output-spaces", "--nprocs", "--nthreads",
            "--omp-nthreads", "--mem", "--mem-mb", "--anat-only", "--level", "--ignore",
            "--force", "--subject-anatomical-reference", "--skull-strip-t1w",
            "--skip-bids-validation", "--skip_bids_validation", "--stop-on-first-crash",
        }
        legacy_defaults = {"--fs-no-reconall", "--output-layout", "bids"}
        if len(self.fmriprep_args) == 3 and set(self.fmriprep_args) == legacy_defaults:
            self.fmriprep_args = []
            return self
        unexpected = [item for item in self.fmriprep_args if item.startswith("-") and item in protected]
        if unexpected:
            raise ValueError(
                "fmriprep_args may not override managed options: " + ", ".join(sorted(set(unexpected)))
            )
        return self


class VisualQCConfig(StrictModel):
    enabled: bool = False
    mode: Literal["shadow"] = "shadow"
    packet_dir: str = "data/visual_qc/packets"
    labels_path: str = "data/visual_qc/labels.jsonl"
    model_backend: Literal["none", "dinov2_embedding", "qwen2_5_vl"] = "none"
    model_id: str = "facebook/dinov2-small"
    device: str = "cuda:0"
    allow_model_download: bool = False
    reference_path: str | None = None
    anomaly_threshold: float | None = Field(default=None, ge=0)
    max_new_tokens: int = Field(default=384, ge=64, le=2048)
    min_pixels: int = Field(default=200704, ge=3136)
    max_pixels: int = Field(default=401408, ge=3136)
    max_panels: int = Field(default=8, ge=1, le=12)

    @model_validator(mode="after")
    def validate_visual_resolution(self) -> VisualQCConfig:
        if self.max_pixels < self.min_pixels:
            raise ValueError("qc.visual.max_pixels must be greater than or equal to min_pixels")
        return self


class QuantitativeQCConfig(StrictModel):
    enabled: bool = True
    output_dir: str = "runs/qc"
    max_nonfinite_fraction: float = Field(default=0.0, ge=0, le=1)
    min_mask_voxels: int = Field(default=1, ge=1)
    max_mask_edge_fraction: float | None = Field(default=None, ge=0, le=1)
    min_bold_tsnr: float | None = Field(default=None, ge=0)
    min_anat_bold_mask_dice: float | None = Field(default=None, ge=0, le=1)


class QCConfig(StrictModel):
    engine: str = "rule_file_image_qc"
    min_output_items: int = Field(default=1, ge=0)
    min_visual_artifacts: int = Field(default=3, ge=0)
    fd_threshold_mm: float = Field(default=0.5, ge=0)
    max_mean_fd_mm: float | None = Field(default=0.5, ge=0)
    max_fd_outlier_fraction: float | None = Field(default=0.2, ge=0, le=1)
    max_mean_std_dvars: float | None = Field(default=1.5, ge=0)
    block_database_on_review: bool = True
    human_review_enabled: bool = True
    require_review_note: bool = True
    quantitative: QuantitativeQCConfig = Field(default_factory=QuantitativeQCConfig)
    visual: VisualQCConfig = Field(default_factory=VisualQCConfig)


class DatabaseConfig(StrictModel):
    enabled: bool = True
    backend: Literal["jsonl", "mysql"] = "jsonl"
    jsonl_path: str = "runs/db_records.jsonl"
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "neuro_agent"
    password: str | None = None
    password_env: str = "NEURO_AGENT_MYSQL_PASSWORD"
    database: str = "neuro_preprocess"
    charset: str = "utf8mb4"
    connect_timeout: int = Field(default=5, ge=1)
    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_interval_seconds: float = Field(default=1.0, ge=0, le=60)


class PermissionConfig(StrictModel):
    preflight: PermissionMode = "allow"
    source: PermissionMode = "allow"
    fetch: PermissionMode = "ask"
    preprocess: PermissionMode = "ask"
    qc: PermissionMode = "allow"
    db: PermissionMode = "ask"
    report: PermissionMode = "allow"


class AppConfig(StrictModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    source: SourceConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    mock: dict[str, Any] = Field(default_factory=lambda: {"raw_items": 3})


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path, project_root: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(project_root or PROJECT_ROOT) / path


def find_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable
    environment_candidate = Path(sys.executable).parent / name
    return str(environment_candidate) if environment_candidate.is_file() else None


def extract_absolute_path(text: str) -> str | None:
    quoted = re.search(r"[\"'](/[^\"']+)[\"']", text)
    if quoted:
        return quoted.group(1)
    plain = re.search(r"(/[^\s，。；,;]+)", text)
    return plain.group(1) if plain else None


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        cwd_candidate = Path.cwd() / config_path
        config_path = cwd_candidate if cwd_candidate.exists() else PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(raw).model_dump(mode="json")
    preprocess = config["preprocess"]
    if preprocess["dcm2bids"] == "dcm2bids":
        preprocess["dcm2bids"] = find_executable("dcm2bids") or "dcm2bids"
    if preprocess["freesurfer_license"] == "data/private/license.txt" and os.environ.get("FS_LICENSE"):
        preprocess["freesurfer_license"] = os.environ["FS_LICENSE"]
    config["project_root"] = str(PROJECT_ROOT)
    config["config_path"] = str(config_path.resolve())
    return config
