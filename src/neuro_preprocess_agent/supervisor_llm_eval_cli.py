from __future__ import annotations

import argparse
from pathlib import Path

from neuro_preprocess_agent.supervisor_llm_evaluation import run_llm_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the approved golden dataset against the production LLM Supervisor"
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "evals/llm_supervisor/golden.jsonl"),
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs/llm_supervisor_example.json")
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evals/llm_supervisor/reports"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()

    report, json_path, markdown_path = run_llm_evaluation(
        args.dataset,
        args.config,
        args.output_dir,
        repeats=args.repeats,
        split=args.split,
        case_ids=set(args.case_ids or []),
        max_cases=args.max_cases,
    )
    metrics = report["metrics"]
    print(f"runs={metrics['passed_runs']}/{metrics['runs']}")
    print(f"schema_valid_rate={metrics['schema_valid_rate']:.3f}")
    print(f"raw_plan_exact_match_rate={metrics['raw_plan_exact_match_rate']:.3f}")
    print(
        "constrained_plan_exact_match_rate="
        f"{metrics['constrained_plan_exact_match_rate']:.3f}"
    )
    print(f"parameter_case_exact_rate={metrics['parameter_case_exact_rate']:.3f}")
    print(f"escalation_match_rate={metrics['escalation_match_rate']:.3f}")
    print(f"forbidden_action_count={metrics['forbidden_action_count']}")
    print(f"decision_stability_rate={metrics['decision_stability_rate']:.3f}")
    print(
        f"latency_p50_p95={metrics['latency_p50_seconds']}/"
        f"{metrics['latency_p95_seconds']}s"
    )
    print(f"total_tokens={metrics['total_tokens']}")
    print(f"json_report={json_path.resolve()}")
    print(f"markdown_report={markdown_path.resolve()}")


if __name__ == "__main__":
    main()
