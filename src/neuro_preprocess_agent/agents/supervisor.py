from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuro_preprocess_agent.state import PipelineState
from neuro_preprocess_agent.tools.registry import WorkerRegistry

StepName = Literal["preflight", "source", "fetch", "preprocess", "qc", "db", "report"]
ALLOWED_STEPS: tuple[StepName, ...] = ("preflight", "source", "fetch", "preprocess", "qc", "db", "report")


class FMRIPrepPlanUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_spaces: list[str] | None = None
    nprocs: int | None = Field(default=None, ge=1)
    omp_nthreads: int | None = Field(default=None, ge=1)
    memory_mb: int | None = Field(default=None, ge=1024)
    low_mem: bool | None = None
    anat_only: bool | None = None
    level: Literal["minimal", "resampling", "full"] | None = None
    subject_anatomical_reference: Literal["first-lex", "unbiased", "sessionwise"] | None = None


class SupervisorDecision(BaseModel):
    execution_plan: list[StepName] = Field(min_length=1)
    next_node: StepName
    skip_steps: list[StepName] = Field(default_factory=list)
    reason: str
    fmriprep_options: FMRIPrepPlanUpdate | None = None


class Supervisor:
    """Plans once with an LLM and enforces deterministic safety constraints."""

    def __init__(self, workers: WorkerRegistry) -> None:
        self.workers = workers

    def default_plan(self, state: PipelineState) -> list[str]:
        request = state.get("request", "")
        plan = ["preflight", "source", "fetch", "preprocess", "qc", "db", "report"]
        if state["config"].get("database", {}).get("enabled") is False or any(
            phrase in request for phrase in ("跳过入库", "不要入库", "不入库", "不写数据库")
        ):
            plan.remove("db")
        if any(phrase in request for phrase in ("只做预处理", "只跑预处理")):
            plan = ["preflight", "source", "fetch", "preprocess", "report"]
        return plan

    def decide(self, state: PipelineState) -> dict[str, Any]:
        stage = state.get("stage")
        plan = list(state.get("execution_plan") or self.default_plan(state))
        completed = list(dict.fromkeys(state.get("completed_steps", [])))
        skipped = list(dict.fromkeys(state.get("skipped_steps", [])))
        completed_delta: list[str] = []

        if stage in ALLOWED_STEPS and stage != "report" and stage not in completed:
            completed.append(stage)
            completed_delta.append(stage)

        if state.get("errors"):
            return self._result(state, plan, completed, skipped, completed_delta, [], "report", "A worker failed; generate a failure report", "safety")

        qc_result = state.get("qc_result", {})
        if stage == "qc" and not qc_result.get("approved_for_database", qc_result.get("passed", False)):
            if qc_result.get("review_required") and state["config"].get("qc", {}).get("human_review_enabled", True):
                return self._result(
                    state, plan, completed, skipped, completed_delta, [], "qc_review",
                    "QC soft thresholds require an auditable human decision before database routing", "safety",
                )
            skip_delta = ["db"] if "db" in plan and "db" not in skipped else []
            reason = "QC requires human review; database write is blocked" if qc_result.get("review_required") else "QC failed; database write is blocked"
            return self._result(state, plan, completed, skipped, completed_delta, skip_delta, "report", reason, "safety")

        supervisor_config = state["config"].get("supervisor", {})
        should_call_llm = supervisor_config.get("mode") == "llm" and not state.get("execution_plan")
        if should_call_llm:
            try:
                decision = self._llm_decision(state, plan, completed, skipped)
                plan = self._validate_plan(list(decision.execution_plan), state)
                skip_delta = [step for step in decision.skip_steps if step not in skipped]
                next_node = self._safe_next(plan, completed, skipped + skip_delta, decision.next_node)
                updates = decision.fmriprep_options.model_dump(exclude_none=True) if decision.fmriprep_options else {}
                return self._result(
                    state, plan, completed, skipped, completed_delta, skip_delta, next_node,
                    decision.reason, "llm", updates,
                )
            except Exception as exc:
                if not supervisor_config.get("fallback_to_rule", True):
                    raise
                next_node = self._safe_next(plan, completed, skipped)
                return self._result(
                    state, plan, completed, skipped, completed_delta, [], next_node,
                    f"LLM planning failed; deterministic fallback used: {type(exc).__name__}: {exc}", "rule_fallback",
                )

        plan = self._validate_plan(plan, state)
        next_node = self._safe_next(plan, completed, skipped)
        return self._result(state, plan, completed, skipped, completed_delta, [], next_node, "Selected the next pending approved step", "rule")

    def _llm_decision(
        self,
        state: PipelineState,
        plan: list[str],
        completed: list[str],
        skipped: list[str],
    ) -> SupervisorDecision:
        settings = state["config"]["supervisor"]
        if settings.get("provider", "deepseek") != "deepseek":
            raise ValueError(f"Unsupported supervisor provider: {settings.get('provider')}")
        from langchain_deepseek import ChatDeepSeek

        system_prompt = """你是神经影像数据流水线的 Supervisor。你负责理解需求、制定一次可执行计划并选择第一个 worker，不直接执行下载、Docker 或数据库操作。

可用 worker 由输入 catalog 给出。必须遵守：
1. preflight、source 和 fetch 必须在 preprocess 前；需要入库时 qc 必须在 db 前。
2. report 必须是最后一步。
3. 不得编造 catalog 之外的 worker。
4. 用户只要求预处理时可以省略 qc 和 db；否则默认保留 qc。
5. 不要把自然语言命令或 shell 命令放入计划，计划只包含 worker 名称。
	6. 返回符合 SupervisorDecision schema 的结构化结果。
	7. 只在用户明确提出资源、输出空间、解剖-only或纵向处理要求时填写 fmriprep_options；不要猜测扫描参数。
	8. fmriprep_options 只能表达 schema 中的受管参数，不能返回 shell 参数。
"""
        payload = {
            "request": state.get("request"),
            "runtime_mode": state["config"]["runtime"]["mode"],
            "source": state["config"]["source"],
            "default_plan": plan,
            "completed_steps": completed,
            "skipped_steps": skipped,
            "worker_catalog": self.workers.catalog(),
        }
        model_kwargs: dict[str, Any] = {
            "model": settings.get("model", "deepseek-chat"),
            "api_key": settings.get("api_key") or os.environ.get(settings.get("api_key_env", "DEEPSEEK_API_KEY")),
            "temperature": settings.get("temperature", 0),
        }
        if settings.get("extra_body"):
            model_kwargs["extra_body"] = settings["extra_body"]
        model = ChatDeepSeek(**model_kwargs).with_structured_output(SupervisorDecision)
        decision = model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ])
        if isinstance(decision, SupervisorDecision):
            return decision
        return SupervisorDecision.model_validate(decision)

    def _validate_plan(self, plan: list[str], state: PipelineState) -> list[str]:
        normalized = [step for step in dict.fromkeys(plan) if step in ALLOWED_STEPS]
        if "preprocess" in normalized:
            for dependency in reversed(("preflight", "source", "fetch")):
                if dependency not in normalized:
                    normalized.insert(0, dependency)
        if "db" in normalized:
            if state["config"].get("database", {}).get("enabled") is False:
                normalized.remove("db")
            elif "qc" not in normalized:
                normalized.insert(normalized.index("db"), "qc")
        if "report" in normalized:
            normalized.remove("report")
        normalized.append("report")
        order = {name: index for index, name in enumerate(ALLOWED_STEPS)}
        return sorted(normalized, key=order.__getitem__)

    @staticmethod
    def _safe_next(plan: list[str], completed: list[str], skipped: list[str], proposed: str | None = None) -> str:
        done = set(completed) | set(skipped)
        if proposed in plan and proposed not in done:
            first_pending = next((step for step in plan if step not in done), "report")
            if proposed == first_pending:
                return proposed
        return next((step for step in plan if step not in done), "report")

    @staticmethod
    def _result(
        state: PipelineState,
        plan: list[str],
        completed: list[str],
        skipped: list[str],
        completed_delta: list[str],
        skip_delta: list[str],
        next_node: str,
        reason: str,
        mode: str,
        preprocess_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "execution_plan": plan,
            "completed_steps": completed_delta,
            "skipped_steps": skip_delta,
            "next_node": next_node,
            "reason": reason,
            "mode": mode,
            "effective_completed": completed,
            "effective_skipped": list(dict.fromkeys(skipped + skip_delta)),
            "preprocess_updates": preprocess_updates or {},
        }
