# Changelog

## Unreleased

- Added pre-fMRIPrep anatomical input QC with per-subject skull-stripping decisions, conservative HITL review, command fingerprint updates, and real ten-subject validation.
- Restricted CLI `--yes` to initial plan approval so preflight, input-QC, and output-QC reviews still require explicit human decisions.
- Added interactive CLI heartbeats with elapsed time and active fMRIPrep log paths for long-running graph nodes.
- Added the first human-reviewed failing/corrected visual Golden Case and a repeatable metric-and-label regression validator.
- Validated the input-QC decision on a real pre-skull-stripped OpenNeuro subject; the corrected run raised anatomical-functional mask Dice from 0.693498 to 0.921624 and passed MySQL gating.
- Separated anatomical input state (`full_head`, `defaced`, `pre_skull_stripped`, or `unknown`) from visual QC failure labels in both the UI and audit ledger.
- Added paired raw T1w/raw BOLD and processed QC views, acquisition metadata, motion metrics, and batch-relative brain-mask warnings to the annotation workspace.
- Fixed managed fMRIPrep arguments so `skull_strip_t1w=auto` is passed explicitly instead of silently falling back to fMRIPrep's `force` default.
- Added a blind-first Streamlit QC annotation workspace with packet navigation, quantitative evidence, optional VLM comparison, multi-label severity/exclusion decisions, and append-only review history.
- Hardened human review records with cross-field validation and anonymized synthetic cases so filenames and sample identifiers cannot leak target labels.
- Added a local Qwen2.5-VL-3B shadow reviewer with multi-panel input, Pydantic output validation, per-subject failure isolation, and deterministic decision reconciliation.
- Added offline VLM scoring for labeled manifests and recorded the first real/synthetic zero-shot baseline without granting the model database-gating authority.
- Downloaded and checksum-verified the official model weights, then validated the complete LangGraph QC/HITL/report path on ten OpenNeuro subjects using one RTX 3090.
- Rebuilt the project virtual environment around CUDA-enabled PyTorch 2.4.0+cu124.
- Added a reproducible GPU venv rebuild script that reuses the validated local CUDA binaries.
- Isolated project dependencies from unrelated Conda packages and documented the Python 3.10 server-extra boundary.
- Constrained Transformers to 4.x for compatibility with the validated PyTorch 2.4 CUDA stack.
- Validated a three-subject OpenNeuro batch with bounded two-worker fMRIPrep fan-out, QC, visual embeddings, and MySQL persistence.
- Run fMRIPrep containers with the host UID/GID so new mounted outputs remain user-owned.
- Added a deterministic evaluation harness with isolated LangGraph checkpoints and per-case workspaces.
- Added worker-boundary fault injection, HITL scenario playback, path assertions, safety metrics, and JSON/Markdown reports.
- Added an offline CI quality gate covering normal flow, QC/database blocking, plan revision, review rejection, and retry recovery.
- Added a preflight graph node for source access, output permissions, disk, Docker image, license, DICOM tools, resource budget, and database checks.
- Made exhausted database retries converge to an auditable failure report and verified real MySQL upsert/QC blocking behavior.
- Added quantitative NIfTI, mask, tSNR, and standard-space anatomical-functional overlap metrics.
- Added a real-data Golden smoke suite, aggregate content fingerprints, and explicit BIDS subject selection.
- Added durable fMRIPrep completion markers and adoption of previously validated successful outputs.

## 0.7.0 - 2026-09-01

- Added standardized visual QC packets from NIfTI outputs and fMRIPrep reportlets.
- Added traceable component-level synthetic failures and an append-only human label ledger.
- Added optional DINOv2 embedding extraction and reviewed-normal reference scoring in shadow mode.
- Persisted visual model evidence without allowing it to bypass deterministic QC or human review.
- Loaded the local MySQL credential file automatically in CLI commands.
- Added visual QC dataset commands, documentation, configuration, and tests.

## 0.6.0 - 2026-09-01

- Added real MySQL 8.4 local service management and run/subject-level schemas.
- Added explicit DICOM participant/session jobs, sampled header validation, and dcm2niix PATH handling.
- Added multi-subject and multi-session NIfTI manifests instead of filename-only organization.
- Added a second LangGraph interrupt for auditable post-QC human approval or rejection.
- Added structured fMRIPrep options, protected command fields, and interface compatibility checks.
- Fixed OpenNeuro invocation for current `openneuro-py` and added selective download controls.
- Validated a 14,986-file DICOM study into T1w/BOLD/DWI BIDS output.

## 0.5.0 - 2026-09-01

- Added QC spatial shape checks and configurable FD/DVARS review thresholds.
- Added `review_required` QC status and database gating.
- Added atomic report writes and lock-protected atomic JSONL updates.
- Prevented LangGraph thread reuse and invalid resume after completion.
- Isolated converted DICOM/NIfTI BIDS directories by run ID.
- Added strict Pydantic configuration validation for unknown fields.
- Expanded regression coverage from 15 to 21 tests.

## 0.4.0 - 2026-08-31

- Added PyBIDS preflight and modality-aware QC.
- Added bounded subject batching with LangGraph `Send`.
- Added run-isolated fMRIPrep work/log directories and subject locks.
- Added completed-subject reuse for graph retries.
