from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neuro_preprocess_agent.evaluation import (
    AssertionSpec,
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
            self.assertEqual(report["metrics"]["passed_cases"], 8)
            self.assertEqual(report["metrics"]["false_pass_count"], 0)
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


if __name__ == "__main__":
    unittest.main()
