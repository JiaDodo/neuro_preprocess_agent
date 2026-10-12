from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neuro_preprocess_agent.evaluation import (
    AgentExpectation,
    AssertionSpec,
    _evaluate_agent_expectation,
    _evaluate_assertion,
    run_suite,
)
from neuro_preprocess_agent.graph import WORKERS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvaluationTestCase(unittest.TestCase):
    def test_offline_smoke_suite_passes_and_restores_workers(self) -> None:
        original_qc = WORKERS.get("qc")
        original_fetch = WORKERS.get("fetch")
        with tempfile.TemporaryDirectory() as temp_dir:
            report, json_path, markdown_path = run_suite(
                PROJECT_ROOT / "evals/suites/offline_smoke.json",
                temp_dir,
            )

            self.assertTrue(report["quality_gate"]["passed"])
            self.assertEqual(report["metrics"]["passed_cases"], 10)
            self.assertEqual(report["metrics"]["false_pass_count"], 0)
            self.assertEqual(report["metrics"]["agent_passed_cases"], 10)
            self.assertEqual(report["metrics"]["plan_exact_match_rate"], 1.0)
            self.assertEqual(report["metrics"]["required_step_recall"], 1.0)
            self.assertEqual(report["metrics"]["step_order_match_rate"], 1.0)
            self.assertEqual(report["metrics"]["forbidden_action_count"], 0)
            self.assertEqual(report["metrics"]["hitl_match_rate"], 1.0)
            self.assertEqual(report["metrics"]["llm_decision_cases"], 0)
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertIs(WORKERS.get("qc"), original_qc)
            self.assertIs(WORKERS.get("fetch"), original_fetch)

    def test_assertion_failure_preserves_expected_and_actual_values(self) -> None:
        result = _evaluate_assertion(
            {"qc_result": {"passed": False}},
            AssertionSpec(path="qc_result.passed", operator="eq", value=True),
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["expected"])
        self.assertFalse(result["actual"])

    def test_agent_expectation_detects_wrong_plan_and_forbidden_worker(self) -> None:
        result = _evaluate_agent_expectation(
            {
                "final_plan": ["preflight", "source", "db", "report"],
                "executed_steps": ["preflight", "source", "db", "report"],
                "interrupt_count": 0,
                "supervisor_decision_count": 5,
            },
            AgentExpectation(
                expected_plan=["preflight", "source", "report"],
                required_steps=["preflight", "source", "report"],
                forbidden_steps=["db"],
                expected_interrupts=1,
                max_supervisor_decisions=4,
            ),
        )

        self.assertIsNotNone(result)
        self.assertFalse(result["passed"])
        self.assertFalse(result["plan_exact_match"])
        self.assertEqual(result["forbidden_action_count"], 1)
        self.assertFalse(result["hitl_match"])
        self.assertTrue(result["step_order_match"])


if __name__ == "__main__":
    unittest.main()
