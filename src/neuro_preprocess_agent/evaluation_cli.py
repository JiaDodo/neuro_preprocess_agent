from __future__ import annotations

import argparse
from pathlib import Path

from neuro_preprocess_agent.evaluation import run_suite

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic neuro-preprocess-agent evaluations"
    )
    parser.add_argument(
        "--suite",
        default=str(PROJECT_ROOT / "evals/suites/offline_smoke.json"),
        help="Evaluation suite JSON path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evals/reports"),
        help="Directory for JSON, Markdown, and isolated case workspaces",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run one case ID; repeat as needed",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Run cases matching a tag; repeat as needed",
    )
    args = parser.parse_args()

    report, json_path, markdown_path = run_suite(
        args.suite,
        args.output_dir,
        case_ids=set(args.case_ids or []),
        tags=set(args.tags or []),
    )
    metrics = report["metrics"]
    print(f"quality_gate={'PASS' if report['quality_gate']['passed'] else 'FAIL'}")
    print(f"cases={metrics['passed_cases']}/{metrics['total_cases']}")
    print(f"assertions={metrics['passed_assertions']}/{metrics['total_assertions']}")
    print(f"false_passes={metrics['false_pass_count']}")
    print(f"json_report={json_path}")
    print(f"markdown_report={markdown_path}")
    raise SystemExit(0 if report["quality_gate"]["passed"] else 1)
