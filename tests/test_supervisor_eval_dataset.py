from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neuro_preprocess_agent.supervisor_eval_dataset import (
    SupervisorGoldenCase,
    generate_candidates,
    import_reviewed_dataset,
    read_review_csv,
    write_review_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SupervisorEvalDatasetTestCase(unittest.TestCase):
    def test_candidates_have_fixed_split_and_unique_identifiers(self) -> None:
        cases = generate_candidates(PROJECT_ROOT / "evals/suites/offline_smoke.json")

        self.assertEqual(len(cases), 30)
        self.assertEqual(sum(case.split == "dev" for case in cases), 20)
        self.assertEqual(sum(case.split == "test" for case in cases), 10)
        self.assertEqual(len({case.case_id for case in cases}), 30)
        self.assertTrue(all(case.review_status == "pending" for case in cases))

    def test_review_csv_round_trip_preserves_unicode_and_parameters(self) -> None:
        cases = generate_candidates()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review.csv"
            write_review_csv(cases, path)
            loaded = read_review_csv(path)

        self.assertEqual(loaded, cases)
        resource_case = next(case for case in loaded if case.case_id == "resources_memory")
        self.assertEqual(resource_case.expected_parameters, {"memory_mb": 24576})

    def test_freeze_requires_review_and_exports_only_approved_rows(self) -> None:
        cases = generate_candidates()
        with tempfile.TemporaryDirectory() as temp_dir:
            review_path = Path(temp_dir) / "review.csv"
            output_path = Path(temp_dir) / "golden.jsonl"
            write_review_csv(cases, review_path)

            with self.assertRaisesRegex(ValueError, "pending review"):
                import_reviewed_dataset(review_path, output_path)

            reviewed = [
                case.model_copy(
                    update={"review_status": "approved" if index < 2 else "rejected"}
                )
                for index, case in enumerate(cases)
            ]
            write_review_csv(reviewed, review_path)
            output, counts = import_reviewed_dataset(review_path, output_path)

            records = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 2)
            self.assertEqual(counts["approved"], 2)
            self.assertEqual(counts["rejected"], 28)
            self.assertEqual(counts["pending"], 0)

    def test_case_schema_rejects_forbidden_step_in_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden_steps"):
            SupervisorGoldenCase(
                case_id="invalid",
                split="dev",
                category="safety",
                request="test",
                expected_plan=["preflight", "db", "report"],
                required_steps=["preflight", "report"],
                forbidden_steps=["db"],
                rationale="invalid fixture",
            )


if __name__ == "__main__":
    unittest.main()
