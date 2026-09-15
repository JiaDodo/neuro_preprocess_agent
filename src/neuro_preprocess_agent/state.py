from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

PipelineStage = Literal[
    "load_config",
    "supervisor",
    "human_approval",
    "preflight",
    "preflight_review",
    "source",
    "fetch",
    "input_qc",
    "input_qc_review",
    "preprocess",
    "qc",
    "qc_review",
    "db",
    "report",
    "end",
    "failed",
]


class PipelineState(TypedDict, total=False):
    request: str
    config_path: str
    config: dict[str, Any]
    stage: PipelineStage
    next_node: str
    run_id: str
    execution_plan: list[str]
    plan_approved: bool
    human_feedback: Annotated[list[dict[str, Any]], operator.add]
    completed_steps: Annotated[list[str], operator.add]
    skipped_steps: Annotated[list[str], operator.add]
    supervisor_decisions: Annotated[list[dict[str, Any]], operator.add]
    preflight_result: dict[str, Any]
    preflight_reviewed: bool
    source: dict[str, Any]
    raw_data: dict[str, Any]
    input_qc_result: dict[str, Any]
    input_qc_review: dict[str, Any]
    preprocess_context: dict[str, Any]
    preprocess_jobs: list[dict[str, Any]]
    preprocess_job: dict[str, Any]
    preprocess_subject_results: Annotated[list[dict[str, Any]], operator.add]
    processed_data: dict[str, Any]
    qc_result: dict[str, Any]
    qc_review: dict[str, Any]
    db_result: dict[str, Any]
    report: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
