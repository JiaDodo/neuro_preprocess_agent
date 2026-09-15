from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuro_preprocess_agent.graph import WORKERS, build_graph
from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.runtime import AgentRuntime, interrupt_message


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssertionSpec(EvalModel):
    path: str
    operator: Literal[
        "eq",
        "ne",
        "contains",
        "not_contains",
        "exists",
        "not_exists",
        "length_eq",
        "ge",
        "le",
    ]
    value: Any = None


class FaultSpec(EvalModel):
    type: Literal["worker_result", "transient_worker_error"]
    worker: str
    result: dict[str, Any] | None = None
    failures: int = Field(default=1, ge=1, le=10)
    message: str = "Injected evaluation failure"

    @model_validator(mode="after")
    def validate_fault(self) -> FaultSpec:
        if self.type == "worker_result" and self.result is None:
            raise ValueError("worker_result fault requires result")
        return self


class EvalCase(EvalModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str
    tags: list[str] = Field(default_factory=list)
    request: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    responses: list[Any] = Field(default_factory=lambda: [True])
    fault: FaultSpec | None = None
    assertions: list[AssertionSpec] = Field(min_length=1)
    must_block_database: bool = False


class EvalSuite(EvalModel):
    schema_version: str = "1.0"
    name: str
    description: str = ""
    base_config: str
    cases: list[EvalCase] = Field(min_length=1)


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _replace_placeholders(value: Any, case_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace("__CASE_ROOT__", str(case_root))
    if isinstance(value, list):
        return [_replace_placeholders(item, case_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_placeholders(item, case_root) for key, item in value.items()
        }
    return value


def _materialize_config(base: dict[str, Any], case: EvalCase, case_root: Path) -> Path:
    config = _replace_placeholders(_deep_merge(base, case.config_overrides), case_root)
    config.setdefault("storage", {})
    config["storage"].update(
        {
            "raw_dir": str(case_root / "data/raw"),
            "processed_dir": str(case_root / "data/processed"),
            "report_dir": str(case_root / "runs/reports"),
        }
    )
    config.setdefault("preprocess", {})
    config["preprocess"].update(
        {
            "bids_dir": str(case_root / "data/bids"),
            "derivatives_dir": str(case_root / "data/derivatives/fmriprep"),
            "log_dir": str(case_root / "runs/logs/fmriprep"),
            "work_dir": str(case_root / "data/work/fmriprep"),
            "lock_dir": str(case_root / "runs/locks/fmriprep"),
        }
    )
    config.setdefault("database", {})
    config["database"].update(
        {"backend": "jsonl", "jsonl_path": str(case_root / "runs/db_records.jsonl")}
    )
    visual = config.setdefault("qc", {}).setdefault("visual", {})
    visual["packet_dir"] = str(case_root / "data/visual_qc/packets")
    visual["labels_path"] = str(case_root / "data/visual_qc/labels.jsonl")

    config_path = case_root / "config.json"
    atomic_write_text(config_path, json.dumps(config, ensure_ascii=False, indent=2))
    return config_path


@contextmanager
def _inject_fault(fault: FaultSpec | None) -> Iterator[dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "type": None,
        "worker": None,
        "attempts": 0,
        "injected_failures": 0,
    }
    if fault is None:
        yield diagnostics
        return

    original = WORKERS.get(fault.worker)
    diagnostics.update({"type": fault.type, "worker": fault.worker})

    if fault.type == "worker_result":

        def handler(_state: dict[str, Any]) -> dict[str, Any]:
            diagnostics["attempts"] += 1
            return deepcopy(fault.result or {})
    else:

        def handler(state: dict[str, Any]) -> dict[str, Any]:
            diagnostics["attempts"] += 1
            if diagnostics["injected_failures"] < fault.failures:
                diagnostics["injected_failures"] += 1
                raise subprocess.CalledProcessError(
                    75, ["eval-fault", fault.worker], stderr=fault.message
                )
            return original.handler(state)  # type: ignore[arg-type]

    WORKERS._workers[fault.worker] = replace(original, handler=handler)
    try:
        yield diagnostics
    finally:
        WORKERS._workers[fault.worker] = original


_MISSING = object()


def _lookup(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _evaluate_assertion(
    document: dict[str, Any], spec: AssertionSpec
) -> dict[str, Any]:
    actual = _lookup(document, spec.path)
    if spec.operator == "exists":
        passed = actual is not _MISSING
    elif spec.operator == "not_exists":
        passed = actual is _MISSING
    elif actual is _MISSING:
        passed = False
    elif spec.operator == "eq":
        passed = actual == spec.value
    elif spec.operator == "ne":
        passed = actual != spec.value
    elif spec.operator == "contains":
        passed = spec.value in actual
    elif spec.operator == "not_contains":
        passed = spec.value not in actual
    elif spec.operator == "length_eq":
        passed = len(actual) == spec.value
    elif spec.operator == "ge":
        passed = actual >= spec.value
    elif spec.operator == "le":
        passed = actual <= spec.value
    else:  # pragma: no cover - Pydantic rejects unknown operators
        passed = False
    return {
        "path": spec.path,
        "operator": spec.operator,
        "expected": spec.value,
        "actual": None if actual is _MISSING else actual,
        "passed": passed,
    }


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "stage",
        "execution_plan",
        "completed_steps",
        "skipped_steps",
        "preflight_result",
        "source",
        "raw_data",
        "processed_data",
        "qc_result",
        "qc_review",
        "db_result",
        "report",
        "errors",
    )
    return {key: state[key] for key in keys if key in state}


def run_case(
    case: EvalCase, base_config: dict[str, Any], workspace_root: Path
) -> dict[str, Any]:
    started = time.monotonic()
    case_root = workspace_root / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_config(base_config, case, case_root)
    request = _replace_placeholders(case.request, case_root)
    thread_id = f"eval-{case.case_id}-{uuid4().hex[:8]}"
    state: dict[str, Any] = {}
    execution_error: str | None = None

    with _inject_fault(case.fault) as fault_diagnostics:
        try:
            runtime = AgentRuntime(build_graph(checkpointer=InMemorySaver()))
            state = runtime.start(request, config_path, thread_id)
            response_index = 0
            while interrupt_message(state) is not None:
                if response_index >= len(case.responses):
                    raise RuntimeError(
                        "Evaluation case ran out of responses while the graph was interrupted"
                    )
                state = runtime.resume(case.responses[response_index], thread_id)
                response_index += 1
        except Exception as exc:  # noqa: BLE001 - evaluation reports arbitrary graph failures as data
            execution_error = f"{type(exc).__name__}: {exc}"

    document = {
        **state,
        "_evaluation": {"fault": fault_diagnostics, "execution_error": execution_error},
    }
    assertion_results = [
        _evaluate_assertion(document, assertion) for assertion in case.assertions
    ]
    qc_result = state.get("qc_result", {})
    db_result = state.get("db_result")
    database_reached = (
        db_result is not None and db_result.get("status") != "blocked_by_qc"
    )
    false_pass = case.must_block_database and (
        bool(qc_result.get("passed") or qc_result.get("approved_for_database"))
        or database_reached
    )
    passed = (
        execution_error is None
        and all(item["passed"] for item in assertion_results)
        and not false_pass
    )
    return {
        "case_id": case.case_id,
        "description": case.description,
        "tags": case.tags,
        "passed": passed,
        "duration_seconds": round(time.monotonic() - started, 3),
        "must_block_database": case.must_block_database,
        "database_reached": database_reached,
        "false_pass": false_pass,
        "execution_error": execution_error,
        "fault": fault_diagnostics,
        "assertions": assertion_results,
        "state": _state_summary(state),
        "workspace": str(case_root),
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    assertion_results = [
        assertion for result in results for assertion in result["assertions"]
    ]
    safety_results = [result for result in results if result["must_block_database"]]
    recovery_results = [result for result in results if "recovery" in result["tags"]]
    durations = [float(result["duration_seconds"]) for result in results]
    return {
        "total_cases": len(results),
        "passed_cases": sum(result["passed"] for result in results),
        "case_pass_rate": sum(result["passed"] for result in results) / len(results),
        "total_assertions": len(assertion_results),
        "passed_assertions": sum(item["passed"] for item in assertion_results),
        "assertion_pass_rate": sum(item["passed"] for item in assertion_results)
        / len(assertion_results),
        "safety_cases": len(safety_results),
        "false_pass_count": sum(result["false_pass"] for result in safety_results),
        "hard_failure_block_rate": (
            sum(not result["false_pass"] for result in safety_results)
            / len(safety_results)
            if safety_results
            else None
        ),
        "recovery_cases": len(recovery_results),
        "recovery_success_rate": (
            sum(result["passed"] for result in recovery_results) / len(recovery_results)
            if recovery_results
            else None
        ),
        "duration_mean_seconds": round(mean(durations), 3),
        "duration_median_seconds": round(median(durations), 3),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"# Evaluation Report: {report['suite']['name']}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Quality gate: **{'PASS' if report['quality_gate']['passed'] else 'FAIL'}**",
        f"- Cases: {metrics['passed_cases']}/{metrics['total_cases']}",
        f"- Assertions: {metrics['passed_assertions']}/{metrics['total_assertions']}",
        f"- False passes: {metrics['false_pass_count']}",
        f"- Mean duration: {metrics['duration_mean_seconds']} s",
        "",
        "| Case | Result | Duration (s) | DB reached | False pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        lines.append(
            f"| `{result['case_id']}` | {'PASS' if result['passed'] else 'FAIL'} | "
            f"{result['duration_seconds']} | {result['database_reached']} | {result['false_pass']} |"
        )
    failures = [
        (result["case_id"], assertion)
        for result in report["results"]
        for assertion in result["assertions"]
        if not assertion["passed"]
    ]
    if failures:
        lines.extend(["", "## Failed Assertions", ""])
        lines.extend(
            f"- `{case_id}`: `{item['path']} {item['operator']} {item['expected']!r}`; actual={item['actual']!r}"
            for case_id, item in failures
        )
    return "\n".join(lines) + "\n"


def run_suite(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    case_ids: set[str] | None = None,
    tags: set[str] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    suite_file = Path(suite_path).resolve()
    suite = EvalSuite.model_validate_json(suite_file.read_text(encoding="utf-8"))
    base_config_path = (suite_file.parent / suite.base_config).resolve()
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    selected = [
        case
        for case in suite.cases
        if (not case_ids or case.case_id in case_ids)
        and (not tags or tags.intersection(case.tags))
    ]
    if not selected:
        raise ValueError("No evaluation cases matched the requested filters")

    now = datetime.now().astimezone()
    generated_at = now.isoformat(timespec="seconds")
    run_stamp = now.strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir).resolve()
    workspace_root = output_root / "workspaces" / run_stamp
    results = [run_case(case, base_config, workspace_root) for case in selected]
    metrics = _metrics(results)
    report = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "path": str(suite_file),
            "selected_cases": len(selected),
        },
        "quality_gate": {
            "passed": metrics["passed_cases"] == metrics["total_cases"]
            and metrics["false_pass_count"] == 0,
            "requirements": ["all selected cases pass", "false_pass_count equals zero"],
        },
        "metrics": metrics,
        "results": results,
    }
    json_path = output_root / f"evaluation_{run_stamp}.json"
    markdown_path = output_root / f"evaluation_{run_stamp}.md"
    atomic_write_text(
        json_path, json.dumps(report, ensure_ascii=False, indent=2, default=str)
    )
    atomic_write_text(markdown_path, _markdown_report(report))
    atomic_write_text(
        output_root / "latest.json",
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
    )
    atomic_write_text(output_root / "latest.md", _markdown_report(report))
    return report, json_path, markdown_path
