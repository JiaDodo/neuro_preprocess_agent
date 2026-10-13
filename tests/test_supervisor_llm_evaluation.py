from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuro_preprocess_agent.agents.supervisor import Supervisor, SupervisorDecision
from neuro_preprocess_agent.supervisor_eval_dataset import SupervisorGoldenCase
from neuro_preprocess_agent.supervisor_llm_evaluation import run_llm_evaluation
from neuro_preprocess_agent.tools.workers import build_worker_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SupervisorLLMEvaluationTestCase(unittest.TestCase):
    def test_replay_scores_repeats_tokens_and_stability(self) -> None:
        case = SupervisorGoldenCase(
            case_id="full",
            split="dev",
            category="planning",
            request="完整处理 OpenNeuro ds000001",
            expected_plan=[
                "preflight",
                "source",
                "fetch",
                "preprocess",
                "qc",
                "db",
                "report",
            ],
            required_steps=[
                "preflight",
                "source",
                "fetch",
                "preprocess",
                "qc",
                "db",
                "report",
            ],
            expected_next_node="preflight",
            rationale="test fixture",
            review_status="approved",
        )

        def planner(_state: dict) -> dict:
            return {
                "raw_decision": {
                    "execution_plan": case.expected_plan,
                    "next_node": "preflight",
                    "skip_steps": [],
                    "reason": "test",
                },
                "constrained_plan": case.expected_plan,
                "constrained_next_node": "preflight",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
                "response": {"model": "fake"},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "golden.jsonl"
            dataset.write_text(case.model_dump_json() + "\n", encoding="utf-8")
            report, json_path, markdown_path = run_llm_evaluation(
                dataset,
                PROJECT_ROOT / "configs/llm_supervisor_example.json",
                root / "reports",
                repeats=2,
                planner=planner,
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())

        self.assertEqual(report["metrics"]["passed_runs"], 2)
        self.assertEqual(report["metrics"]["raw_plan_exact_match_rate"], 1.0)
        self.assertEqual(report["metrics"]["decision_stability_rate"], 1.0)
        self.assertEqual(report["metrics"]["total_tokens"], 240)

    def test_production_planner_exposes_raw_and_constrained_outputs(self) -> None:
        class RawMessage:
            def __init__(self) -> None:
                self.id = "fake-message"
                self.usage_metadata = {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }
                self.response_metadata = {
                    "model_name": "fake-model",
                    "finish_reason": "stop",
                }

        class FakeModel:
            def __init__(self, **_: object) -> None:
                self.include_raw = False

            def with_structured_output(
                self, _: object, *, include_raw: bool = False
            ) -> FakeModel:
                self.include_raw = include_raw
                return self

            def invoke(self, _: object) -> object:
                decision = SupervisorDecision(
                    execution_plan=["db", "preprocess"],
                    next_node="db",
                    reason="fake",
                )
                if self.include_raw:
                    return {
                        "raw": RawMessage(),
                        "parsed": decision,
                        "parsing_error": None,
                    }
                return decision

        config = json.loads(
            (PROJECT_ROOT / "configs/llm_supervisor_example.json").read_text(
                encoding="utf-8"
            )
        )
        state = {"request": "完整处理数据", "config": config, "stage": "load_config"}
        supervisor = Supervisor(build_worker_registry())
        with patch("langchain_deepseek.ChatDeepSeek", FakeModel):
            result = supervisor.invoke_llm_for_evaluation(state)  # type: ignore[arg-type]

        self.assertEqual(result["raw_decision"]["execution_plan"], ["db", "preprocess"])
        self.assertEqual(
            result["constrained_plan"],
            ["preflight", "source", "fetch", "preprocess", "qc", "db", "report"],
        )
        self.assertEqual(result["usage"]["total_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
