from __future__ import annotations

import json
from typing import Any

from neuro_preprocess_agent.config import project_path
from neuro_preprocess_agent.db import record_run
from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.state import PipelineState
from neuro_preprocess_agent.tools.preflight import run_preflight
from neuro_preprocess_agent.tools.preprocess import prepare_preprocessing
from neuro_preprocess_agent.tools.qc import run_qc
from neuro_preprocess_agent.tools.registry import WorkerRegistry, WorkerSpec
from neuro_preprocess_agent.tools.source import fetch_data, resolve_source


def write_database(state: PipelineState) -> dict[str, Any]:
    qc_result = state.get("qc_result", {})
    if not qc_result.get("approved_for_database", qc_result.get("passed", False)):
        return {"status": "blocked_by_qc", "review_required": bool(qc_result.get("review_required"))}
    if state["config"]["runtime"]["mode"] != "run":
        return {
            "status": "planned",
            "backend": state["config"]["database"]["backend"],
            "message": "Non-run mode: database write skipped",
        }
    return record_run(state["config"], dict(state))


def generate_report(state: PipelineState) -> dict[str, Any]:
    report_dir = project_path(state["config"]["storage"]["report_dir"], state["config"].get("project_root"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{state['run_id']}_summary.json"
    if state.get("errors"):
        status = "failed"
    elif state.get("plan_approved") is False:
        status = "cancelled"
    elif state.get("qc_result", {}).get("manual_review", {}).get("decision") == "reject":
        status = "qc_rejected"
    elif state.get("qc_result", {}).get("review_required") and not state.get("qc_result", {}).get("approved_for_database"):
        status = "review_required"
    elif state.get("qc_result") and not state["qc_result"].get("passed"):
        status = "qc_failed"
    elif state["config"]["runtime"]["mode"] == "dry_run":
        status = "planned"
    elif state["config"]["runtime"]["mode"] == "mock":
        status = "mocked"
    else:
        status = "completed"
    report = {
        "schema_version": "1.0",
        "run_id": state.get("run_id"),
        "status": status,
        "request": state.get("request"),
        "preflight_result": state.get("preflight_result"),
        "source": state.get("source"),
        "raw_data": state.get("raw_data"),
        "input_qc_result": state.get("input_qc_result"),
        "input_qc_review": state.get("input_qc_review"),
        "processed_data": state.get("processed_data"),
        "qc_result": state.get("qc_result"),
        "qc_review": state.get("qc_review"),
        "db_result": state.get("db_result"),
        "execution_plan": state.get("execution_plan"),
        "completed_steps": state.get("completed_steps", []),
        "skipped_steps": state.get("skipped_steps", []),
        "supervisor_decisions": state.get("supervisor_decisions", []),
        "errors": state.get("errors", []),
        "report_path": str(report_path),
    }
    atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report


def build_worker_registry() -> WorkerRegistry:
    registry = WorkerRegistry()
    registry.register(WorkerSpec("preflight", "Validate data access, services, tools, disk, and resource budgets", "preflight_result", "read", run_preflight))
    registry.register(WorkerSpec("source", "Validate and normalize a local, URL, or OpenNeuro source", "source", "read", resolve_source))
    registry.register(WorkerSpec("fetch", "Download public data or stage local input", "raw_data", "network_write", fetch_data))
    registry.register(WorkerSpec("preprocess", "Convert input to BIDS and prepare fMRIPrep subject jobs", "preprocess_context", "compute_write", prepare_preprocessing))
    registry.register(WorkerSpec("qc", "Check outputs, headers, motion, logs, and optional visual-QC shadow evidence", "qc_result", "read", run_qc))
    registry.register(WorkerSpec("db", "Persist a QC-passed run to JSONL or MySQL", "db_result", "database_write", write_database))
    registry.register(WorkerSpec("report", "Write the final structured run report", "report", "write", generate_report))
    return registry
