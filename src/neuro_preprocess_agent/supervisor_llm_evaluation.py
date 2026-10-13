from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from neuro_preprocess_agent.agents.supervisor import Supervisor
from neuro_preprocess_agent.config import load_config
from neuro_preprocess_agent.graph import _apply_request_overrides
from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.supervisor_eval_dataset import SupervisorGoldenCase
from neuro_preprocess_agent.tools.workers import build_worker_registry

Planner = Callable[[dict[str, Any]], dict[str, Any]]


def load_golden_cases(path: str | Path) -> list[SupervisorGoldenCase]:
    cases: list[SupervisorGoldenCase] = []
    identifiers: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = SupervisorGoldenCase.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid golden JSONL line {line_number}: {exc}") from exc
            if case.review_status != "approved":
                raise ValueError(f"golden case {case.case_id} is not approved")
            if case.case_id in identifiers:
                raise ValueError(f"duplicate golden case_id: {case.case_id}")
            identifiers.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("golden dataset contains no cases")
    return cases


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _token_usage(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        ),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _score(
    case: SupervisorGoldenCase,
    invocation: dict[str, Any],
    *,
    repeat: int,
    latency_seconds: float,
) -> dict[str, Any]:
    decision = invocation["raw_decision"]
    raw_plan = list(decision.get("execution_plan", []))
    constrained_plan = list(invocation.get("constrained_plan", []))
    predicted_parameters = decision.get("fmriprep_options") or {}
    required_missing = [step for step in case.required_steps if step not in raw_plan]
    forbidden_observed = [step for step in case.forbidden_steps if step in raw_plan]
    parameter_keys = sorted(
        set(case.expected_parameters).union(predicted_parameters)
    )
    parameter_checks = {
        key: {
            "expected": case.expected_parameters.get(key),
            "actual": predicted_parameters.get(key),
            "passed": case.expected_parameters.get(key) == predicted_parameters.get(key),
        }
        for key in parameter_keys
    }
    usage = _token_usage(invocation.get("usage", {}))
    checks = {
        "raw_plan_exact": raw_plan == case.expected_plan,
        "constrained_plan_exact": constrained_plan == case.expected_plan,
        "required_steps": not required_missing,
        "forbidden_steps": not forbidden_observed,
        "next_node": decision.get("next_node") == case.expected_next_node,
        "parameters_exact": predicted_parameters == case.expected_parameters,
        "escalation": case.expected_escalation == "plan_approval",
    }
    return {
        "case_id": case.case_id,
        "split": case.split,
        "category": case.category,
        "repeat": repeat,
        "passed": all(checks.values()),
        "schema_valid": True,
        "checks": checks,
        "expected": {
            "plan": case.expected_plan,
            "required_steps": case.required_steps,
            "next_node": case.expected_next_node,
            "parameters": case.expected_parameters,
            "escalation": case.expected_escalation,
        },
        "actual": {
            "raw_decision": decision,
            "constrained_plan": constrained_plan,
            "constrained_next_node": invocation.get("constrained_next_node"),
            "escalation": "plan_approval",
        },
        "required_missing": required_missing,
        "forbidden_observed": forbidden_observed,
        "parameter_checks": parameter_checks,
        "hallucinated_parameters": sorted(
            set(predicted_parameters) - set(case.expected_parameters)
        ),
        "latency_seconds": round(latency_seconds, 3),
        "usage": usage,
        "response": invocation.get("response", {}),
        "error": None,
    }


def _failed_result(
    case: SupervisorGoldenCase,
    *,
    repeat: int,
    latency_seconds: float,
    error: Exception,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "category": case.category,
        "repeat": repeat,
        "passed": False,
        "schema_valid": False,
        "checks": {},
        "expected": {
            "plan": case.expected_plan,
            "required_steps": case.required_steps,
            "next_node": case.expected_next_node,
            "parameters": case.expected_parameters,
            "escalation": case.expected_escalation,
        },
        "actual": None,
        "required_missing": case.required_steps,
        "forbidden_observed": [],
        "parameter_checks": {},
        "hallucinated_parameters": [],
        "latency_seconds": round(latency_seconds, 3),
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "response": {},
        "error": f"{type(error).__name__}: {error}",
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [result for result in results if result["schema_valid"]]
    latencies = [float(result["latency_seconds"]) for result in results]
    parameter_checks = [
        check
        for result in valid
        for check in result["parameter_checks"].values()
    ]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_case[result["case_id"]].append(result)
    stable_cases = 0
    for repetitions in by_case.values():
        outputs = {
            json.dumps(
                {
                    "execution_plan": item["actual"]["raw_decision"].get(
                        "execution_plan"
                    ),
                    "next_node": item["actual"]["raw_decision"].get("next_node"),
                    "skip_steps": item["actual"]["raw_decision"].get("skip_steps", []),
                    "fmriprep_options": item["actual"]["raw_decision"].get(
                        "fmriprep_options", {}
                    ),
                    "constrained_plan": item["actual"]["constrained_plan"],
                    "constrained_next_node": item["actual"][
                        "constrained_next_node"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for item in repetitions
            if item["schema_valid"]
        }
        if len(outputs) == 1 and all(item["schema_valid"] for item in repetitions):
            stable_cases += 1
    total = len(results)
    return {
        "runs": total,
        "passed_runs": sum(result["passed"] for result in results),
        "run_pass_rate": sum(result["passed"] for result in results) / total,
        "schema_valid_rate": len(valid) / total,
        "raw_plan_exact_match_rate": (
            sum(result["checks"]["raw_plan_exact"] for result in valid) / len(valid)
            if valid
            else 0.0
        ),
        "constrained_plan_exact_match_rate": (
            sum(result["checks"]["constrained_plan_exact"] for result in valid)
            / len(valid)
            if valid
            else 0.0
        ),
        "required_step_recall": (
            mean(
                (
                    len(result["expected"]["required_steps"])
                    - len(result["required_missing"])
                )
                / len(result["expected"]["required_steps"])
                for result in valid
            )
            if valid
            else 0.0
        ),
        "forbidden_action_count": sum(
            len(result["forbidden_observed"]) for result in results
        ),
        "parameter_case_exact_rate": (
            sum(result["checks"]["parameters_exact"] for result in valid) / len(valid)
            if valid
            else 0.0
        ),
        "escalation_match_rate": (
            sum(result["checks"]["escalation"] for result in valid) / len(valid)
            if valid
            else 0.0
        ),
        "parameter_field_accuracy": (
            sum(check["passed"] for check in parameter_checks) / len(parameter_checks)
            if parameter_checks
            else None
        ),
        "hallucinated_parameter_count": sum(
            len(result["hallucinated_parameters"]) for result in results
        ),
        "decision_stability_rate": stable_cases / len(by_case),
        "latency_p50_seconds": round(median(latencies), 3),
        "latency_p95_seconds": round(_percentile(latencies, 0.95) or 0.0, 3),
        "input_tokens": sum(result["usage"]["input_tokens"] for result in results),
        "output_tokens": sum(result["usage"]["output_tokens"] for result in results),
        "total_tokens": sum(result["usage"]["total_tokens"] for result in results),
        "errors": sum(result["error"] is not None for result in results),
    }


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# LLM Supervisor Evaluation",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Model: `{report['model']['provider']}/{report['model']['name']}`",
        f"- Cases: {report['dataset']['cases']}; repeats: {report['dataset']['repeats']}",
        f"- Run pass rate: {metrics['run_pass_rate']:.3f}",
        f"- Schema valid rate: {metrics['schema_valid_rate']:.3f}",
        f"- Raw plan exact match: {metrics['raw_plan_exact_match_rate']:.3f}",
        f"- Constrained plan exact match: {metrics['constrained_plan_exact_match_rate']:.3f}",
        f"- Parameter case exact: {metrics['parameter_case_exact_rate']:.3f}",
        f"- Escalation match: {metrics['escalation_match_rate']:.3f}",
        f"- Forbidden actions: {metrics['forbidden_action_count']}",
        f"- Decision stability: {metrics['decision_stability_rate']:.3f}",
        f"- Latency P50/P95: {metrics['latency_p50_seconds']}/{metrics['latency_p95_seconds']} s",
        f"- Total tokens: {metrics['total_tokens']}",
        "",
        "| Case | Split | Repeat | Result | Raw plan | Parameters | Latency (s) | Error |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in report["results"]:
        checks = result["checks"]
        lines.append(
            f"| `{result['case_id']}` | {result['split']} | {result['repeat']} | "
            f"{'PASS' if result['passed'] else 'FAIL'} | "
            f"{checks.get('raw_plan_exact', False)} | "
            f"{checks.get('parameters_exact', False)} | "
            f"{result['latency_seconds']} | {result['error'] or ''} |"
        )
    return "\n".join(lines) + "\n"


def run_llm_evaluation(
    dataset_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    repeats: int = 3,
    split: str = "all",
    case_ids: set[str] | None = None,
    max_cases: int | None = None,
    planner: Planner | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    cases = load_golden_cases(dataset_path)
    selected = [
        case
        for case in cases
        if (split == "all" or case.split == split)
        and (not case_ids or case.case_id in case_ids)
    ]
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("no golden cases matched the requested filters")

    base_config = load_config(config_path)
    base_config["supervisor"]["mode"] = "llm"
    base_config["supervisor"]["fallback_to_rule"] = False
    supervisor = Supervisor(build_worker_registry())
    invoke = planner or supervisor.invoke_llm_for_evaluation
    results: list[dict[str, Any]] = []
    for case in selected:
        for repeat in range(1, repeats + 1):
            config = json.loads(json.dumps(base_config))
            config = _apply_request_overrides(config, case.request)
            state = {
                "request": case.request,
                "config": config,
                "stage": case.stage,
            }
            started = time.monotonic()
            try:
                invocation = invoke(state)
                results.append(
                    _score(
                        case,
                        invocation,
                        repeat=repeat,
                        latency_seconds=time.monotonic() - started,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - model failures are evaluation data
                results.append(
                    _failed_result(
                        case,
                        repeat=repeat,
                        latency_seconds=time.monotonic() - started,
                        error=exc,
                    )
                )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    metrics = _metrics(results)
    split_metrics = {
        name: _metrics([result for result in results if result["split"] == name])
        for name in ("dev", "test")
        if any(result["split"] == name for result in results)
    }
    report = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "model": {
            "provider": base_config["supervisor"].get("provider"),
            "name": base_config["supervisor"].get("model"),
            "temperature": base_config["supervisor"].get("temperature"),
        },
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "cases": len(selected),
            "repeats": repeats,
            "split": split,
        },
        "metrics": metrics,
        "split_metrics": split_metrics,
        "results": results,
    }
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir)
    json_path = output_root / f"llm_supervisor_{stamp}.json"
    markdown_path = output_root / f"llm_supervisor_{stamp}.md"
    rendered_json = json.dumps(report, ensure_ascii=False, indent=2)
    rendered_markdown = _markdown(report)
    atomic_write_text(json_path, rendered_json)
    atomic_write_text(markdown_path, rendered_markdown)
    atomic_write_text(output_root / "latest.json", rendered_json)
    atomic_write_text(output_root / "latest.md", rendered_markdown)
    return report, json_path, markdown_path
