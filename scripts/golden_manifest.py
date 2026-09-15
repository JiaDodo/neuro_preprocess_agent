#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from neuro_preprocess_agent.io_utils import atomic_write_text


def fingerprint(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_digest.hexdigest()}\n".encode())
        file_count += 1
        total_bytes += size
    return {
        "root": str(root.resolve()),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def parse_cases(values: list[str]) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    for value in values:
        case_id, separator, path = value.partition("=")
        if not separator or not case_id or not path:
            raise SystemExit(f"Invalid --case {value!r}; expected CASE_ID=/absolute/path")
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Golden case directory does not exist: {root}")
        cases[case_id] = root
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify aggregate fingerprints for local Golden Datasets")
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--manifest", type=Path, default=Path("evals/manifests/golden.local.json"))
    parser.add_argument("--case", action="append", default=[], help="CASE_ID=/absolute/path; repeat for create")
    args = parser.parse_args()

    if args.command == "create":
        cases = parse_cases(args.case)
        if not cases:
            raise SystemExit("create requires at least one --case")
        manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cases": {case_id: fingerprint(path) for case_id, path in cases.items()},
        }
        atomic_write_text(args.manifest, json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"created={args.manifest} cases={len(cases)}")
        return

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for case_id, expected in manifest["cases"].items():
        actual = fingerprint(Path(expected["root"]))
        passed = all(actual[key] == expected[key] for key in ("file_count", "total_bytes", "tree_sha256"))
        print(f"{'PASS' if passed else 'FAIL'} {case_id}: files={actual['file_count']} bytes={actual['total_bytes']}")
        if not passed:
            failures.append({"case_id": case_id, "expected": expected, "actual": actual})
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
