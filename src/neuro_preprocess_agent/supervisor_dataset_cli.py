from __future__ import annotations

import argparse
from pathlib import Path

from neuro_preprocess_agent.supervisor_eval_dataset import (
    generate_candidates,
    import_reviewed_dataset,
    read_review_csv,
    write_review_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW = PROJECT_ROOT / "evals/llm_supervisor/review.csv"
DEFAULT_GOLDEN = PROJECT_ROOT / "evals/llm_supervisor/golden.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate the human-reviewed LLM Supervisor dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Create the 30-case review CSV")
    generate.add_argument("--output", default=str(DEFAULT_REVIEW))
    generate.add_argument(
        "--suite", default=str(PROJECT_ROOT / "evals/suites/offline_smoke.json")
    )
    generate.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate a review CSV")
    validate.add_argument("--review", default=str(DEFAULT_REVIEW))

    freeze = subparsers.add_parser(
        "freeze", help="Export approved rows to immutable JSONL"
    )
    freeze.add_argument("--review", default=str(DEFAULT_REVIEW))
    freeze.add_argument("--output", default=str(DEFAULT_GOLDEN))
    freeze.add_argument("--allow-pending", action="store_true")

    args = parser.parse_args()
    if args.command == "generate":
        output = Path(args.output)
        if output.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing review file: {output}")
        cases = generate_candidates(args.suite)
        write_review_csv(cases, output)
        print(f"review_csv={output.resolve()}")
        print("cases=30 dev=20 test=10 pending=30")
        print("next=review rows, set review_status to approved/rejected, then run freeze")
        return

    if args.command == "validate":
        cases = read_review_csv(args.review)
        counts = {
            status: sum(case.review_status == status for case in cases)
            for status in ("approved", "rejected", "pending")
        }
        print(f"review_csv={Path(args.review).resolve()}")
        print(f"cases={len(cases)}")
        print(" ".join(f"{key}={value}" for key, value in counts.items()))
        return

    output, counts = import_reviewed_dataset(
        args.review, args.output, allow_pending=args.allow_pending
    )
    print(f"golden_jsonl={output.resolve()}")
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
