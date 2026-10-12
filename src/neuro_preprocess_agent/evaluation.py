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


class AgentExpectation(EvalModel):
    expected_plan: list[str] | None = None
    required_steps: list[str] = Field(default_factory=list)
    forbidden_steps: list[str] = Field(default_factory=list)
    expected_interrupts: int | None = Field(default=None, ge=0)
    max_supervisor_decisions: int | None = Field(default=None, ge=1)


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
    agent_expectation: AgentExpectation | None = None


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
        "supervisor_decisions",
    )
    return {key: state[key] for key in keys if key in state}


def _agent_trace(state: dict[str, Any], interrupt_count: int) -> dict[str, Any]:
    worker_steps = {"preflight", "source", "fetch", "preprocess", "qc", "db", "report"}
    executed_steps: list[str] = []
    event_nodes: list[str] = []
    for event in state.get("events", []):
        node = str(event.get("node", ""))
        if not node:
            continue
        event_nodes.append(node)
        if node in worker_steps and node not in executed_steps:
            executed_steps.append(node)
    decisions = state.get("supervisor_decisions", [])
    config = state.get("config", {})
    preprocess = config.get("preprocess", {})
    return {
        "final_plan": list(state.get("execution_plan", [])),
        "executed_steps": executed_steps,
        "event_nodes": event_nodes,
        "interrupt_count": interrupt_count,
        "supervisor_decision_count": len(decisions),
        "supervisor_modes": [item.get("mode") for item in decisions],
        "llm_decision_count": sum(item.get("mode") == "llm" for item in decisions),
        "effective_parameters": {
            "runtime_mode": config.get("runtime", {}).get("mode"),
            "source_type": config.get("source", {}).get("type"),
            "max_workers": preprocess.get("max_workers"),
            "fmriprep_options": preprocess.get("fmriprep_options", {}),
        },
    }


def _evaluate_agent_expectation(
    trace: dict[str, Any], expectation: AgentExpectation | None
) -> dict[str, Any] | None:
    if expectation is None:
        return None
    checks: list[dict[str, Any]] = []

    if expectation.expected_plan is not None:
        checks.append({
            "name": "exact_plan_match",
            "expected": expectation.expected_plan,
            "actual": trace["final_plan"],
            "passed": trace["final_plan"] == expectation.expected_plan,
        })

    executed = set(trace["executed_steps"])
    missing_required = [step for step in expectation.required_steps if step not in executed]
    checks.append({
        "name": "required_steps",
        "expected": expectation.required_steps,
        "actual": trace["executed_steps"],
        "missing": missing_required,
        "passed": not missing_required,
    })
    required_index = 0
    for step in trace["executed_steps"]:
        if required_index < len(expectation.required_steps) and step == expectation.required_steps[required_index]:
            required_index += 1
    step_order_match = required_index == len(expectation.required_steps)
    checks.append({
        "name": "required_step_order",
        "expected": expectation.required_steps,
        "actual": trace["executed_steps"],
        "passed": step_order_match,
    })
    forbidden_observed = [step for step in expectation.forbidden_steps if step in executed]
    checks.append({
        "name": "forbidden_steps",
        "expected": expectation.forbidden_steps,
        "actual": forbidden_observed,
        "passed": not forbidden_observed,
    })

    if expectation.expected_interrupts is not None:
        checks.append({
            "name": "interrupt_count",
            "expected": expectation.expected_interrupts,
            "actual": trace["interrupt_count"],
            "passed": trace["interrupt_count"] == expectation.expected_interrupts,
        })
    if expectation.max_supervisor_decisions is not None:
        checks.append({
            "name": "supervisor_decision_budget",
            "expected": expectation.max_supervisor_decisions,
            "actual": trace["supervisor_decision_count"],
            "passed": trace["supervisor_decision_count"] <= expectation.max_supervisor_decisions,
        })

    required_count = len(expectation.required_steps)
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "plan_evaluated": expectation.expected_plan is not None,
        "plan_exact_match": (
            trace["final_plan"] == expectation.expected_plan
            if expectation.expected_plan is not None
            else None
        ),
        "required_step_recall": (
            (required_count - len(missing_required)) / required_count
            if required_count
            else None
        ),
        "step_order_match": step_order_match,
        "forbidden_action_count": len(forbidden_observed),
        "hitl_evaluated": expectation.expected_interrupts is not None,
        "hitl_match": (
            trace["interrupt_count"] == expectation.expected_interrupts
            if expectation.expected_interrupts is not None
            else None
        ),
    }


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
    interrupt_count = 0

    with _inject_fault(case.fault) as fault_diagnostics:
        try:
            runtime = AgentRuntime(build_graph(checkpointer=InMemorySaver()))
            state = runtime.start(request, config_path, thread_id)
            response_index = 0
            while interrupt_message(state) is not None:
                interrupt_count += 1
                if response_index >= len(case.responses):
                    raise RuntimeError(
                        "Evaluation case ran out of responses while the graph was interrupted"
                    )
                state = runtime.resume(case.responses[response_index], thread_id)
                response_index += 1
        except Exception as exc:  # noqa: BLE001 - evaluation reports arbitrary graph failures as data
            execution_error = f"{type(exc).__name__}: {exc}"

    agent_trace = _agent_trace(state, interrupt_count)
    agent_evaluation = _evaluate_agent_expectation(agent_trace, case.agent_expectation)
    document = {
        **state,
        "_evaluation": {
            "fault": fault_diagnostics,
            "execution_error": execution_error,
            "agent_trace": agent_trace,
        },
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
        and (agent_evaluation is None or agent_evaluation["passed"])
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
        "agent_evaluation": agent_evaluation,
        "agent_trace": agent_trace,
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
    agent_results = [result["agent_evaluation"] for result in results if result["agent_evaluation"] is not None]
    plan_results = [item for item in agent_results if item["plan_evaluated"]]
    hitl_results = [item for item in agent_results if item["hitl_evaluated"]]
    required_recalls = [item["required_step_recall"] for item in agent_results if item["required_step_recall"] is not None]
    ordered_results = [item for item in agent_results if item["required_step_recall"] is not None]
    decision_counts = [result["agent_trace"]["supervisor_decision_count"] for result in results]
    executed_counts = [len(result["agent_trace"]["executed_steps"]) for result in results]
    llm_decision_counts = [result["agent_trace"]["llm_decision_count"] for result in results]
    tags = sorted({tag for result in results for tag in result["tags"]})
    tag_pass_rates = {
        tag: sum(result["passed"] for result in results if tag in result["tags"])
        / sum(tag in result["tags"] for result in results)
        for tag in tags
    }
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
        "agent_cases": len(agent_results),
        "agent_passed_cases": sum(item["passed"] for item in agent_results),
        "agent_pass_rate": (
            sum(item["passed"] for item in agent_results) / len(agent_results)
            if agent_results
            else None
        ),
        "plan_cases": len(plan_results),
        "plan_exact_match_rate": (
            sum(item["plan_exact_match"] for item in plan_results) / len(plan_results)
            if plan_results
            else None
        ),
        "required_step_recall": mean(required_recalls) if required_recalls else None,
        "step_order_match_rate": (
            sum(item["step_order_match"] for item in ordered_results) / len(ordered_results)
            if ordered_results
            else None
        ),
        "forbidden_action_count": sum(item["forbidden_action_count"] for item in agent_results),
        "hitl_cases": len(hitl_results),
        "hitl_match_rate": (
            sum(item["hitl_match"] for item in hitl_results) / len(hitl_results)
            if hitl_results
            else None
        ),
        "supervisor_decisions_mean": round(mean(decision_counts), 3),
        "executed_steps_mean": round(mean(executed_counts), 3),
        "llm_decision_cases": sum(count > 0 for count in llm_decision_counts),
        "llm_decision_count": sum(llm_decision_counts),
        "tag_pass_rates": tag_pass_rates,
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
        f"- Agent behavior cases: {metrics['agent_passed_cases']}/{metrics['agent_cases']}",
        f"- Exact plan match: {metrics['plan_exact_match_rate']}",
        f"- Required-step recall: {metrics['required_step_recall']}",
        f"- Step-order match: {metrics['step_order_match_rate']}",
        f"- Forbidden actions: {metrics['forbidden_action_count']}",
        f"- HITL match rate: {metrics['hitl_match_rate']}",
        f"- LLM decision coverage: {metrics['llm_decision_cases']}/{metrics['total_cases']} cases",
        f"- Mean duration: {metrics['duration_mean_seconds']} s",
        "",
        "| Case | Result | Agent | Decisions | Duration (s) | DB reached | False pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        lines.append(
            f"| `{result['case_id']}` | {'PASS' if result['passed'] else 'FAIL'} | "
            f"{'PASS' if result['agent_evaluation'] and result['agent_evaluation']['passed'] else ('N/A' if result['agent_evaluation'] is None else 'FAIL')} | "
            f"{result['agent_trace']['supervisor_decision_count']} | {result['duration_seconds']} | "
            f"{result['database_reached']} | {result['false_pass']} |"
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
    agent_failures = [
        (result["case_id"], check)
        for result in report["results"]
        if result["agent_evaluation"] is not None
        for check in result["agent_evaluation"]["checks"]
        if not check["passed"]
    ]
    if agent_failures:
        lines.extend(["", "## Failed Agent Checks", ""])
        lines.extend(
            f"- `{case_id}`: `{item['name']}`; expected={item['expected']!r}, actual={item['actual']!r}"
            for case_id, item in agent_failures
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
        "schema_version": "1.1",
        "generated_at": generated_at,
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "path": str(suite_file),
            "selected_cases": len(selected),
        },
        "quality_gate": {
            "passed": metrics["passed_cases"] == metrics["total_cases"]
            and metrics["false_pass_count"] == 0
            and metrics["forbidden_action_count"] == 0,
            "requirements": [
                "all selected cases pass",
                "false_pass_count equals zero",
                "forbidden_action_count equals zero",
            ],
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
