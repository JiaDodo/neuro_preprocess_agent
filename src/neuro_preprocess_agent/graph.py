from __future__ import annotations

import re
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.error import URLError
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, Send, interrupt

from neuro_preprocess_agent.agents.supervisor import Supervisor
from neuro_preprocess_agent.config import (
    PROJECT_ROOT,
    AppConfig,
    extract_absolute_path,
    load_config,
    project_path,
)
from neuro_preprocess_agent.events import emit, make_event
from neuro_preprocess_agent.interaction import (
    parse_human_response,
    parse_input_qc_response,
    parse_qc_review_response,
)
from neuro_preprocess_agent.state import PipelineState
from neuro_preprocess_agent.tools.preprocess import (
    apply_skull_strip_actions,
    finalize_preprocessing,
    inspect_anatomical_inputs,
    run_subject_preprocessing,
)
from neuro_preprocess_agent.tools.workers import build_worker_registry

WORKERS = build_worker_registry()
SUPERVISOR = Supervisor(WORKERS)


def _is_transient_error(exc: Exception) -> bool:
    return (
        isinstance(exc, (ConnectionError, TimeoutError, URLError, subprocess.CalledProcessError))
        or (exc.__class__.__module__.startswith("pymysql") and exc.__class__.__name__ == "OperationalError")
    )


TRANSIENT_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, retry_on=_is_transient_error)


def _validate_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: config[key] for key in ("project_root", "config_path") if key in config}
    payload = {key: value for key, value in config.items() if key not in metadata}
    validated = AppConfig.model_validate(payload).model_dump(mode="json")
    validated.update(metadata)
    return validated


def _apply_request_overrides(config: dict[str, Any], request: str) -> dict[str, Any]:
    requested_path_text = extract_absolute_path(request)
    url_match = re.search(r"https?://[^\s，。；,;]+", request)
    if requested_path_text and not url_match:
        requested_path = Path(requested_path_text).expanduser()
        config["source"] = {"name": requested_path.name or "local_data", "type": "local_path", "path": str(requested_path)}
    else:
        dataset_match = re.search(r"\b(ds\d{6})\b", request, flags=re.IGNORECASE)
        if dataset_match:
            dataset_id = dataset_match.group(1).lower()
            existing = config.get("source", {})
            if existing.get("type") == "openneuro" and existing.get("dataset_id") == dataset_id:
                config["source"] = {**existing, "dataset_id": dataset_id}
            else:
                config["source"] = {"name": dataset_id, "type": "openneuro", "dataset_id": dataset_id, "uri": "https://openneuro.org"}
        else:
            if url_match:
                config["source"] = {"name": "URL dataset", "type": "url_file", "url": url_match.group(0)}

    workers_match = re.search(r"(?:并行|并发|worker|max_workers)[^\d]*(\d+)", request, flags=re.IGNORECASE)
    if workers_match:
        config["preprocess"]["max_workers"] = int(workers_match.group(1))
    if any(phrase in request.lower() for phrase in ("只转换bids", "只整理bids", "bids only")):
        config["preprocess"]["run_fmriprep"] = False
    if any(phrase in request.lower() for phrase in ("只处理结构像", "只处理解剖", "anat-only", "anat only")):
        config["preprocess"]["fmriprep_options"]["anat_only"] = True
    nprocs_match = re.search(r"(?:每个被试|fmriprep)[^\d]*(\d+)[^\d]*(?:核|线程|进程|nprocs)", request, flags=re.IGNORECASE)
    if nprocs_match:
        config["preprocess"]["fmriprep_options"]["nprocs"] = int(nprocs_match.group(1))
    memory_match = re.search(r"(?:内存|memory)[^\d]*(\d+(?:\.\d+)?)\s*(gb|g|mb|m)", request, flags=re.IGNORECASE)
    if memory_match:
        value = float(memory_match.group(1))
        config["preprocess"]["fmriprep_options"]["memory_mb"] = int(value * 1024 if memory_match.group(2).lower() in {"gb", "g"} else value)
    if any(phrase in request.lower() for phrase in ("dry-run", "dry run", "试运行", "只检查计划")):
        config["runtime"]["mode"] = "dry_run"
    elif any(phrase in request for phrase in ("正式运行", "真实运行", "直接执行", "开始处理")):
        config["runtime"]["mode"] = "run"

    return _validate_runtime_config(config)


def load_config_node(state: PipelineState) -> dict[str, Any]:
    config = load_config(state.get("config_path") or "configs/example.json")
    config = _apply_request_overrides(config, state.get("request", ""))
    run_id = state.get("run_id") or f"run_{datetime.now().astimezone():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    event = make_event("load_config", "Configuration loaded and request normalized", run_id=run_id, config_path=config["config_path"])
    emit(event)
    return {"config": config, "run_id": run_id, "stage": "load_config", "events": [event]}


def supervisor_node(state: PipelineState) -> dict[str, Any]:
    decision = SUPERVISOR.decide(state)
    config = state["config"]
    if decision.get("preprocess_updates"):
        config = {
            **config,
            "preprocess": {
                **config["preprocess"],
                "fmriprep_options": {
                    **config["preprocess"]["fmriprep_options"],
                    **decision["preprocess_updates"],
                },
            },
        }
        config = _validate_runtime_config(config)
    event = make_event(
        "supervisor",
        "Execution plan updated",
        plan=decision["execution_plan"],
        next_node=decision["next_node"],
        reason=decision["reason"],
        mode=decision["mode"],
    )
    emit(event)
    return {
        "stage": "supervisor",
        "execution_plan": decision["execution_plan"],
        "completed_steps": decision["completed_steps"],
        "skipped_steps": decision["skipped_steps"],
        "next_node": decision["next_node"],
        "supervisor_decisions": [{
            "time": event["time"],
            "next_node": decision["next_node"],
            "reason": decision["reason"],
            "mode": decision["mode"],
        }],
        "events": [event],
        "config": config,
    }


def human_approval_node(
    state: PipelineState,
) -> Command[Literal["supervisor", "preflight", "source", "fetch", "preprocess", "qc", "db", "report"]]:
    source = state["config"]["source"]
    preprocess = state["config"]["preprocess"]
    fmriprep = preprocess["fmriprep_options"]
    permissions = state["config"]["permissions"]
    plan = state.get("execution_plan", [])
    source_reference = source.get("path") or source.get("dataset_id") or source.get("url") or "未指定"
    approval_steps = [step for step in plan if permissions.get(step) == "ask"]
    message = "\n".join([
        "我已经根据你的需求制定了执行计划：",
        f"1. 数据来源：{source['type']}，{source_reference}",
        f"2. 运行模式：{state['config']['runtime']['mode']}",
        f"3. 执行顺序：{' -> '.join(plan)}",
        f"4. 需要授权的操作：{', '.join(approval_steps) if approval_steps else '无'}",
        f"5. fMRIPrep 输出：{project_path(preprocess['derivatives_dir'], state['config'].get('project_root'))}",
        f"6. 并行被试数：{preprocess['max_workers']}",
        f"7. fMRIPrep 资源：每被试 nprocs={fmriprep.get('nprocs') or '自动'}，内存={fmriprep.get('memory_mb') or '自动'} MB，anat-only={fmriprep.get('anat_only', False)}",
        f"8. 输出空间：{', '.join(fmriprep.get('output_spaces', [])) or '使用 fMRIPrep 默认值'}；多 session 解剖参考={fmriprep.get('subject_anatomical_reference', 'first-lex')}",
        "回复“继续”即可执行；回复“停止”取消；也可以说“并行数改成1”“数据来源改成本地/路径”“跳过入库”。修改后我会重新展示计划供你确认。",
    ])
    approval_prompt = {"message": message, "run_id": state["run_id"], "plan": plan}
    while True:
        approval = interrupt(approval_prompt)
        if isinstance(approval, str):
            approval = parse_human_response(approval)
        if isinstance(approval, bool):
            break
        if isinstance(approval, dict) and any(key in approval for key in ("source", "preprocess", "execution_plan")):
            break
        approval_prompt = {
            **approval_prompt,
            "message": "我没有识别出你的回复。请说“继续”“停止”，或直接说明修改内容，例如“并行数改成1”“跳过入库”。",
        }
    event = make_event("human_approval", "Human response received", response=approval)
    emit(event)

    feedback = [{"time": event["time"], "response": approval}]
    if approval is False:
        skipped = [step for step in plan if step != "report"]
        return Command(
            update={"plan_approved": False, "skipped_steps": skipped, "human_feedback": feedback, "events": [event]},
            goto="report",
        )
    if isinstance(approval, dict):
        config = dict(state["config"])
        if isinstance(approval.get("source"), dict):
            config["source"] = approval["source"]
        if isinstance(approval.get("preprocess"), dict):
            config["preprocess"] = {**config["preprocess"], **approval["preprocess"]}
        config = _validate_runtime_config(config)
        update: dict[str, Any] = {
            "config": config,
            "plan_approved": False,
            "human_feedback": feedback,
            "events": [event],
        }
        if isinstance(approval.get("execution_plan"), list):
            update["execution_plan"] = approval["execution_plan"]
        return Command(update=update, goto="supervisor")
    return Command(
        update={"plan_approved": True, "human_feedback": feedback, "events": [event]},
        goto=state.get("next_node", "source"),
    )


def qc_review_node(
    state: PipelineState,
) -> Command[Literal["supervisor", "report"]]:
    qc_result = dict(state.get("qc_result", {}))
    if not qc_result.get("hard_checks_passed") or not qc_result.get("review_required"):
        return Command(goto="report")

    failed_checks = qc_result.get("failed_checks", [])
    require_note = state["config"].get("qc", {}).get("require_review_note", True)
    prompt = {
        "message": "\n".join([
            "自动 QC 的文件完整性和影像硬检查已经通过，但以下软阈值需要人工复核：",
            *[f"- {item}" for item in failed_checks],
            "请查看 fMRIPrep HTML 报告和 QC 指标后回复：",
            "- “人工通过：原因”允许本次结果入库；",
            "- “拒绝：原因”阻止入库并结束本次任务。",
            "硬检查失败不能通过人工决定绕过。",
        ]),
        "run_id": state["run_id"],
        "qc_status": qc_result.get("status"),
        "failed_checks": failed_checks,
    }
    while True:
        response = interrupt(prompt)
        if isinstance(response, str):
            response = parse_qc_review_response(response)
        if isinstance(response, dict) and response.get("decision") in {"approve", "reject"}:
            note = str(response.get("note", "")).strip()
            if note or not require_note:
                break
        prompt = {
            **prompt,
            "message": "请明确回复“人工通过：原因”或“拒绝：原因”。人工复核必须留下审计说明。",
        }

    reviewed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    review = {"decision": response["decision"], "note": note, "reviewed_at": reviewed_at}
    event = make_event("qc_review", "Human QC review completed", decision=response["decision"], note=note)
    emit(event)
    qc_result["manual_review"] = review
    qc_result["approved_for_database"] = response["decision"] == "approve"
    qc_result["passed"] = response["decision"] == "approve"

    update = {
        "stage": "qc_review",
        "qc_result": qc_result,
        "qc_review": review,
        "human_feedback": [{"time": event["time"], "response": review}],
        "events": [event],
    }
    if response["decision"] == "approve":
        return Command(update=update, goto="supervisor")
    skipped = ["db"] if "db" in state.get("execution_plan", []) else []
    return Command(update={**update, "skipped_steps": skipped}, goto="report")


def _worker_node(name: str, capture_errors: bool = True):
    def execute(state: PipelineState) -> dict[str, Any]:
        spec = WORKERS.get(name)
        try:
            output = WORKERS.execute(name, state)
            status = output.get("status") if isinstance(output, dict) else None
            event = make_event(name, f"Worker {name} completed", status=status)
            emit(event)
            return {spec.output_key: output, "stage": "end" if name == "report" else name, "events": [event]}
        except Exception as exc:
            if not capture_errors:
                raise
            error = {
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "node": name,
                "type": type(exc).__name__,
                "message": str(exc),
                "retryable": _is_transient_error(exc),
            }
            event = make_event(name, f"Worker {name} failed", level="error", error=error)
            emit(event)
            if name == "report":
                raise
            return {"stage": "failed", "errors": [error], "events": [event]}

    execute.__name__ = f"{name}_node"
    return execute


def preflight_node(state: PipelineState) -> dict[str, Any]:
    output = WORKERS.execute("preflight", state)
    event = make_event("preflight", "Runtime preflight completed", status=output["status"], warnings=output["warnings"])
    emit(event)
    if output["passed"]:
        return {"preflight_result": output, "stage": "preflight", "events": [event]}
    errors = [{
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "node": "preflight",
        "type": "PreflightError",
        "message": "; ".join(output["hard_failures"]),
        "retryable": False,
    }]
    return {"preflight_result": output, "stage": "failed", "errors": errors, "events": [event]}


def preflight_review_node(state: PipelineState) -> Command[Literal["supervisor", "report"]]:
    result = state["preflight_result"]
    prompt = {
        "message": "\n".join([
            "运行前硬检查已经通过，但发现以下资源警告：",
            *[f"- {item}" for item in result.get("warnings", [])],
            "回复“继续”接受风险并执行，或回复“停止”取消任务。",
        ]),
        "run_id": state["run_id"],
        "warnings": result.get("warnings", []),
    }
    while True:
        response = interrupt(prompt)
        normalized = response.strip().lower() if isinstance(response, str) else response
        if normalized is True or (
            isinstance(normalized, str) and normalized in {"继续", "确认", "同意", "yes", "y", "true"}
        ):
            approved = True
            break
        if normalized is False or (
            isinstance(normalized, str) and normalized in {"停止", "取消", "拒绝", "no", "n", "false"}
        ):
            approved = False
            break
        prompt = {
            **prompt,
            "message": "我没有识别出你的回复。请明确回复“继续”接受资源风险，或回复“停止”取消任务。",
        }
    event = make_event("preflight_review", "Human reviewed preflight warnings", approved=approved)
    emit(event)
    feedback = [{"time": event["time"], "response": response, "kind": "preflight_warning"}]
    if approved:
        return Command(
            update={"preflight_reviewed": True, "human_feedback": feedback, "events": [event]},
            goto="supervisor",
        )
    completed = set(state.get("completed_steps", [])) | {"preflight", "report"}
    remaining = [step for step in state.get("execution_plan", []) if step not in completed]
    return Command(
        update={
            "plan_approved": False,
            "preflight_reviewed": True,
            "skipped_steps": remaining,
            "human_feedback": feedback,
            "events": [event],
        },
        goto="report",
    )


def route_after_preflight(state: PipelineState) -> Literal["supervisor", "preflight_review"]:
    settings = state["config"]["preflight"]
    if (
        not state.get("errors")
        and state.get("preflight_result", {}).get("warnings")
        and settings.get("require_warning_approval", True)
        and not state.get("preflight_reviewed")
    ):
        return "preflight_review"
    return "supervisor"


def database_node(state: PipelineState) -> dict[str, Any]:
    settings = state["config"]["database"]
    max_attempts = int(settings.get("max_attempts", 3))
    interval = float(settings.get("retry_interval_seconds", 1.0))
    events: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            output = WORKERS.execute("db", state)
            event = make_event("db", "Worker db completed", status=output.get("status"), attempt=attempt)
            emit(event)
            return {"db_result": {**output, "attempts": attempt}, "stage": "db", "events": [*events, event]}
        except Exception as exc:  # noqa: BLE001 - convert exhausted database retries into graph state
            transient = _is_transient_error(exc)
            event = make_event(
                "db", "Database attempt failed", level="warning" if transient else "error",
                attempt=attempt, max_attempts=max_attempts, error=f"{type(exc).__name__}: {exc}",
            )
            emit(event)
            events.append(event)
            if transient and attempt < max_attempts:
                if interval:
                    time.sleep(interval * (2 ** (attempt - 1)))
                continue
            error = {
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "node": "db",
                "type": type(exc).__name__,
                "message": str(exc),
                "retryable": transient,
                "attempts": attempt,
                "exhausted": transient and attempt >= max_attempts,
            }
            return {
                "db_result": {"status": "failed", "backend": settings["backend"], "attempts": attempt},
                "stage": "failed",
                "errors": [error],
                "events": events,
            }


def preprocess_prepare_node(state: PipelineState) -> dict[str, Any]:
    try:
        output = WORKERS.execute("preprocess", state)
        event = make_event("preprocess", "BIDS input prepared and subject jobs created", subjects=len(output["preprocess_jobs"]))
        emit(event)
        return {**output, "events": [event]}
    except Exception as exc:  # noqa: BLE001 - worker failures are serialized into graph state
        error = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "node": "preprocess",
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": _is_transient_error(exc),
        }
        event = make_event("preprocess", "Preprocessing preparation failed", level="error", error=error)
        emit(event)
        return {"stage": "failed", "errors": [error], "events": [event]}


def input_qc_node(state: PipelineState) -> dict[str, Any]:
    try:
        result = inspect_anatomical_inputs(state)
        jobs = state.get("preprocess_jobs", [])
        if not result["review_required"]:
            jobs = apply_skull_strip_actions(jobs, result["recommended_actions"])
            result["final_actions"] = result["recommended_actions"]
        event = make_event(
            "input_qc",
            "Anatomical input screening completed",
            status=result["status"],
            review_required=result["review_required"],
        )
        emit(event)
        return {"stage": "input_qc", "input_qc_result": result, "preprocess_jobs": jobs, "events": [event]}
    except Exception as exc:  # noqa: BLE001 - convert input inspection failure into graph state
        error = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "node": "input_qc",
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": False,
        }
        event = make_event("input_qc", "Anatomical input screening failed", level="error", error=error)
        emit(event)
        return {"stage": "failed", "errors": [error], "events": [event]}


def route_after_preprocess(state: PipelineState) -> Literal["input_qc", "supervisor"]:
    return "supervisor" if state.get("errors") else "input_qc"


def input_qc_review_node(state: PipelineState) -> Command[Literal["preprocess_dispatch", "report"]]:
    result = dict(state["input_qc_result"])
    flagged = [item for item in result["subject_results"] if item["review_required"]]
    lines = []
    for item in flagged:
        fractions = [str(image.get("nonzero_fraction", "读取失败")) for image in item["images"]]
        lines.append(
            f"- sub-{item['subject']}：{item['classification']}，非零体素占比={', '.join(fractions)}，"
            f"建议 --skull-strip-t1w {item['recommended_action']}"
        )
    subjects = [item["subject"] for item in result["subject_results"]]
    prompt = {
        "message": "\n".join([
            "fMRIPrep 执行前发现需要确认的 T1w 输入：",
            *lines,
            "这是基于体素占比的保守筛查，不会把候选结果直接当作医学结论。",
            "回复“继续”采用建议；回复“停止”取消；也可以说“sub-04 使用 skip”或“所有被试使用 auto”。",
        ]),
        "run_id": state["run_id"],
        "flagged_subjects": flagged,
        "recommended_actions": result["recommended_actions"],
    }
    while True:
        response = interrupt(prompt)
        if isinstance(response, bool):
            parsed = {"approved": response, "subject_actions": {}}
        elif isinstance(response, str):
            parsed = parse_input_qc_response(response, subjects)
        elif isinstance(response, dict) and isinstance(response.get("approved"), bool):
            supplied_actions = response.get("subject_actions", {})
            parsed = response if isinstance(supplied_actions, dict) and all(
                subject in subjects and action in {"auto", "skip", "force"}
                for subject, action in supplied_actions.items()
            ) else None
        else:
            parsed = None
        if parsed is not None:
            break
        prompt = {
            **prompt,
            "message": "我没有识别出你的回复。请说“继续”“停止”，或例如“sub-04 使用 skip”“所有被试使用 auto”。",
        }

    reviewed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    review = {
        "approved": parsed["approved"],
        "subject_actions": parsed["subject_actions"],
        "reviewed_at": reviewed_at,
        "response": response,
    }
    event = make_event("input_qc_review", "Human reviewed anatomical input screening", approved=parsed["approved"])
    emit(event)
    if not parsed["approved"]:
        skipped = [step for step in state.get("execution_plan", []) if step in {"preprocess", "qc", "db"}]
        return Command(
            update={
                "stage": "input_qc_review",
                "plan_approved": False,
                "input_qc_review": review,
                "skipped_steps": skipped,
                "human_feedback": [{"time": event["time"], "response": review}],
                "events": [event],
            },
            goto="report",
        )

    final_actions = {**result["recommended_actions"], **parsed["subject_actions"]}
    result.update({"status": "approved", "review_required": False, "final_actions": final_actions})
    return Command(
        update={
            "stage": "input_qc_review",
            "input_qc_result": result,
            "input_qc_review": review,
            "preprocess_jobs": apply_skull_strip_actions(state.get("preprocess_jobs", []), final_actions),
            "human_feedback": [{"time": event["time"], "response": review}],
            "events": [event],
        },
        goto="preprocess_dispatch",
    )


def preprocess_dispatch_node(state: PipelineState) -> dict[str, Any]:
    event = make_event("preprocess_dispatch", "Subject preprocessing jobs ready for dispatch")
    emit(event)
    return {"events": [event]}


def route_after_input_qc(state: PipelineState) -> Literal["input_qc_review", "preprocess_dispatch", "supervisor"]:
    if state.get("errors"):
        return "supervisor"
    if state.get("input_qc_result", {}).get("review_required"):
        return "input_qc_review"
    return "preprocess_dispatch"


def route_preprocess_jobs(state: PipelineState) -> list[Send] | Literal["preprocess_finalize", "supervisor"]:
    if state.get("errors"):
        return "supervisor"
    jobs = state.get("preprocess_jobs", [])
    completed_subjects = {
        item["subject"]
        for item in state.get("preprocess_subject_results", [])
        if item.get("run_id") == state["run_id"]
    }
    pending_jobs = [job for job in jobs if job["subject"] not in completed_subjects]
    if not pending_jobs:
        return "preprocess_finalize"
    max_workers = int(state["preprocess_context"]["max_workers"])
    return [
        Send("preprocess_subject", {"run_id": state["run_id"], "preprocess_job": job})
        for job in pending_jobs[:max_workers]
    ]


def preprocess_subject_node(state: PipelineState) -> dict[str, Any]:
    job = state["preprocess_job"]
    start_event = make_event(
        "preprocess_subject",
        "Subject preprocessing started",
        subject=job["subject"],
        log_file=job["log_file"],
    )
    emit(start_event)
    output = run_subject_preprocessing(state)
    result = output["preprocess_subject_results"][0]
    finish_event = make_event(
        "preprocess_subject",
        "Subject preprocessing finished",
        subject=result["subject"],
        status=result["status"],
        duration_seconds=result.get("duration_seconds"),
    )
    emit(finish_event)
    return {**output, "events": [start_event, finish_event]}


def preprocess_batch_node(state: PipelineState) -> dict[str, Any]:
    completed = {
        item["subject"]
        for item in state.get("preprocess_subject_results", [])
        if item.get("run_id") == state["run_id"]
    }
    event = make_event(
        "preprocess_batch",
        "Subject batch completed",
        completed_subjects=len(completed),
        total_subjects=len(state.get("preprocess_jobs", [])),
    )
    emit(event)
    return {"events": [event]}


def preprocess_finalize_node(state: PipelineState) -> dict[str, Any]:
    output = finalize_preprocessing(state)
    event = make_event("preprocess", "All subject jobs aggregated", status=output["status"], subjects=len(output["subjects"]))
    emit(event)
    return {"processed_data": output, "stage": "preprocess", "events": [event]}


def route_from_supervisor(
    state: PipelineState,
) -> Literal["human_approval", "preflight", "source", "fetch", "preprocess", "qc", "qc_review", "db", "report"]:
    if not state.get("plan_approved") and state.get("next_node") != "report":
        return "human_approval"
    return state.get("next_node", "report")  # type: ignore[return-value]


def build_graph(checkpointer: Any | None = None):
    builder = StateGraph(PipelineState)
    builder.add_node("load_config", load_config_node)
    builder.add_node("supervisor", supervisor_node, retry_policy=TRANSIENT_RETRY)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("preflight", preflight_node)
    builder.add_node("preflight_review", preflight_review_node)
    builder.add_node("source", _worker_node("source"))
    builder.add_node("fetch", _worker_node("fetch", capture_errors=False), retry_policy=TRANSIENT_RETRY)
    builder.add_node("preprocess", preprocess_prepare_node)
    builder.add_node("input_qc", input_qc_node)
    builder.add_node("input_qc_review", input_qc_review_node)
    builder.add_node("preprocess_dispatch", preprocess_dispatch_node)
    builder.add_node("preprocess_subject", preprocess_subject_node, retry_policy=TRANSIENT_RETRY)
    builder.add_node("preprocess_batch", preprocess_batch_node)
    builder.add_node("preprocess_finalize", preprocess_finalize_node)
    builder.add_node("qc", _worker_node("qc"))
    builder.add_node("qc_review", qc_review_node)
    builder.add_node("db", database_node)
    builder.add_node("report", _worker_node("report"))

    builder.add_edge(START, "load_config")
    builder.add_edge("load_config", "supervisor")
    builder.add_conditional_edges("supervisor", route_from_supervisor, {
        "human_approval": "human_approval",
        "preflight": "preflight",
        "source": "source",
        "fetch": "fetch",
        "preprocess": "preprocess",
        "qc": "qc",
        "qc_review": "qc_review",
        "db": "db",
        "report": "report",
    })
    builder.add_conditional_edges(
        "preflight",
        route_after_preflight,
        {"supervisor": "supervisor", "preflight_review": "preflight_review"},
    )
    for worker in ("source", "fetch", "qc", "db"):
        builder.add_edge(worker, "supervisor")
    builder.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        ["input_qc", "supervisor"],
    )
    builder.add_conditional_edges(
        "input_qc",
        route_after_input_qc,
        ["input_qc_review", "preprocess_dispatch", "supervisor"],
    )
    builder.add_conditional_edges(
        "preprocess_dispatch",
        route_preprocess_jobs,
        ["preprocess_subject", "preprocess_finalize", "supervisor"],
    )
    builder.add_edge("preprocess_subject", "preprocess_batch")
    builder.add_conditional_edges(
        "preprocess_batch",
        route_preprocess_jobs,
        ["preprocess_subject", "preprocess_finalize", "supervisor"],
    )
    builder.add_edge("preprocess_finalize", "supervisor")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


def build_local_graph(checkpoint_path: str | Path | None = None):
    path = project_path(checkpoint_path or "data/checkpoints/checkpoints.sqlite", PROJECT_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return build_graph(checkpointer=SqliteSaver(connection))


graph = build_graph()
