#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import statistics
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from neuro_preprocess_agent.graph import build_graph
from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.runtime import AgentRuntime


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.999999)))
    return ordered[position]


def real_run_metrics(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    intervals = []
    for item in report.get("processed_data", {}).get("subject_results", []):
        if item.get("status") != "completed" or item.get("returncode") != 0:
            continue
        intervals.append({
            "subject": item.get("subject"),
            "start": datetime.fromisoformat(item["started_at"]),
            "finish": datetime.fromisoformat(item["finished_at"]),
            "duration_seconds": float(item["duration_seconds"]),
        })
    if not intervals:
        return {"report": str(report_path), "completed_subjects": 0}

    events = sorted(
        [(item["start"], 1) for item in intervals] + [(item["finish"], -1) for item in intervals],
        key=lambda item: (item[0], item[1]),
    )
    active = 0
    peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    wall_seconds = (max(item["finish"] for item in intervals) - min(item["start"] for item in intervals)).total_seconds()
    service_seconds = sum(item["duration_seconds"] for item in intervals)
    waves = []
    for started_at in sorted({item["start"] for item in intervals}):
        wave = [item for item in intervals if item["start"] == started_at]
        if len(wave) < 2:
            continue
        wave_wall = (max(item["finish"] for item in wave) - started_at).total_seconds()
        waves.append({
            "started_at": started_at.isoformat(),
            "subjects": [item["subject"] for item in wave],
            "concurrency": len(wave),
            "wall_seconds": wave_wall,
            "summed_subject_seconds": sum(item["duration_seconds"] for item in wave),
            "observed_overlap_factor": sum(item["duration_seconds"] for item in wave) / wave_wall,
            "throughput_subjects_per_hour": len(wave) * 3600 / wave_wall,
        })
    return {
        "report": str(report_path),
        "run_id": report.get("run_id"),
        "completed_subjects": len(intervals),
        "peak_observed_concurrency": peak,
        "wall_seconds": wall_seconds,
        "summed_subject_seconds": service_seconds,
        "parallelism_factor": service_seconds / wall_seconds,
        "throughput_subjects_per_hour": len(intervals) * 3600 / wall_seconds,
        "duration_p50_seconds": statistics.median(item["duration_seconds"] for item in intervals),
        "duration_p95_seconds": percentile([item["duration_seconds"] for item in intervals], 0.95),
        "concurrent_waves": waves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LangGraph subject fan-out at bounded concurrency levels")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--task-seconds", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--real-report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evals/reports/fanout_benchmark.json"))
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 1 or args.task_seconds <= 0:
        raise SystemExit("jobs/repeats must be positive and task-seconds must be greater than zero")
    if any(worker < 1 or worker > args.jobs for worker in args.workers):
        raise SystemExit("each worker count must be between 1 and jobs")

    graph_module = importlib.import_module("neuro_preprocess_agent.graph")
    original_runner = graph_module.run_subject_preprocessing
    results: list[dict[str, Any]] = []
    try:
        for workers in args.workers:
            samples = []
            for repeat in range(args.repeats):
                lock = threading.Lock()
                activity = {"active": 0, "peak": 0}

                def synthetic_runner(state: dict[str, Any]) -> dict[str, Any]:
                    job = state["preprocess_job"]
                    started = time.perf_counter()
                    with lock:
                        activity["active"] += 1
                        activity["peak"] = max(activity["peak"], activity["active"])
                    try:
                        time.sleep(args.task_seconds)
                    finally:
                        with lock:
                            activity["active"] -= 1
                    return {"preprocess_subject_results": [{
                        "run_id": job["run_id"], "subject": job["subject"],
                        "status": "planned", "returncode": 0,
                        "duration_seconds": time.perf_counter() - started,
                    }]}

                graph_module.run_subject_preprocessing = synthetic_runner
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    config_path = root / "config.json"
                    config_path.write_text(json.dumps({
                        "runtime": {"mode": "mock"},
                        "supervisor": {"mode": "rule"},
                        "source": {"type": "openneuro", "dataset_id": "ds000001"},
                        "storage": {
                            "raw_dir": str(root / "raw"), "processed_dir": str(root / "processed"),
                            "report_dir": str(root / "reports"),
                        },
                        "preprocess": {
                            "bids_dir": str(root / "bids"), "derivatives_dir": str(root / "derivatives"),
                            "log_dir": str(root / "logs"), "work_dir": str(root / "work"),
                            "lock_dir": str(root / "locks"), "max_workers": workers,
                            "subjects": [f"{index:03d}" for index in range(args.jobs)],
                        },
                        "qc": {"min_output_items": 1},
                        "database": {"enabled": False, "backend": "jsonl", "jsonl_path": str(root / "records.jsonl")},
                    }), encoding="utf-8")
                    runtime = AgentRuntime(build_graph(checkpointer=InMemorySaver()))
                    thread_id = f"benchmark-{workers}-{repeat}-{uuid4().hex[:8]}"
                    runtime.start("benchmark bounded subject fan-out", config_path, thread_id)
                    started = time.perf_counter()
                    final = runtime.resume(True, thread_id)
                    wall_seconds = time.perf_counter() - started
                    if final.get("stage") != "end" or len(final.get("preprocess_subject_results", [])) != args.jobs:
                        raise RuntimeError(f"benchmark run did not complete: workers={workers}, repeat={repeat}")
                    samples.append({"wall_seconds": wall_seconds, "peak_concurrency": activity["peak"]})

            walls = [sample["wall_seconds"] for sample in samples]
            results.append({
                "workers": workers,
                "repeats": args.repeats,
                "wall_seconds_median": statistics.median(walls),
                "wall_seconds_p95": percentile(walls, 0.95),
                "throughput_jobs_per_second": args.jobs / statistics.median(walls),
                "peak_concurrency": max(sample["peak_concurrency"] for sample in samples),
                "samples": samples,
            })
    finally:
        graph_module.run_subject_preprocessing = original_runner

    baseline = next(item["wall_seconds_median"] for item in results if item["workers"] == 1)
    for item in results:
        item["speedup_vs_one_worker"] = baseline / item["wall_seconds_median"]
        item["parallel_efficiency"] = item["speedup_vs_one_worker"] / item["workers"]

    output = {
        "benchmark_type": "synthetic fixed-duration tasks through the production LangGraph fan-out path",
        "jobs_per_run": args.jobs,
        "task_seconds": args.task_seconds,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "langgraph": importlib.metadata.version("langgraph"),
        },
        "synthetic_results": results,
        "real_fmriprep_observation": real_run_metrics(args.real_report) if args.real_report else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output, json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
