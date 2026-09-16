# Bounded concurrency benchmark

## Purpose

This benchmark verifies the subject-level bounded fan-out implemented by the production
LangGraph graph. It answers two separate questions:

1. Does the `Send`-based scheduler respect `max_workers` and scale as worker count increases?
2. What concurrency and throughput were observed in an existing real fMRIPrep run?

It is not an HTTP QPS benchmark. fMRIPrep is a long-running, CPU-, memory-, and I/O-intensive
workload, so the relevant engineering problem is resource-aware parallel task scheduling.

## Reproduction

```bash
.venv/bin/python scripts/benchmark_fanout.py \
  --workers 1 2 4 \
  --jobs 12 \
  --task-seconds 0.2 \
  --repeats 5 \
  --real-report runs/reports/run_20260914_102449_fde23fda_summary.json \
  --output evals/reports/fanout_benchmark_20260916.json
```

The script drives the normal `AgentRuntime` and compiled production graph, including plan
approval, routing, `Send` fan-out, reducer aggregation, QC, and reporting. Only the expensive
subject preprocessing function is replaced by a fixed-duration task. Each worker count is run
five times and the median wall time is used for speedup calculations.

Environment: Python 3.10.13, LangGraph 1.2.11, Linux 5.15.

## Scheduler result

| max_workers | jobs | median wall time | p95 wall time | throughput | speedup | efficiency | measured peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 2.475 s | 2.480 s | 4.85 jobs/s | 1.00x | 100.0% | 1 |
| 2 | 12 | 1.263 s | 1.264 s | 9.50 jobs/s | 1.96x | 98.0% | 2 |
| 4 | 12 | 0.652 s | 0.653 s | 18.41 jobs/s | 3.80x | 94.9% | 4 |

The observed peak exactly matches the configured bound. This demonstrates that graph routing,
runtime concurrency configuration, and result aggregation do not serialize subject jobs or
exceed the requested worker limit.

## Real fMRIPrep observation

Historical report `run_20260914_102449_fde23fda` contains seven newly computed subjects and
three reused subjects. Metrics below include only the seven completed fMRIPrep processes:

| metric | observed value |
|---|---:|
| Peak concurrent fMRIPrep jobs | 4 |
| Overall compute window | 6,063 s |
| Overall throughput | 4.16 subjects/hour |
| Per-subject duration P50 / P95 | 2,022.8 / 2,043.1 s |
| Saturated four-subject wave throughput | 7.05 subjects/hour |
| Saturated wave overlap factor | 3.95x |

The overall throughput is lower than the saturated-wave value because the seven jobs form waves
of 1, 4, and 2. The 3.95x overlap factor means four independent fMRIPrep processes overlapped
almost fully; it is not a controlled comparison against a sequential fMRIPrep run.

## Interpretation and limits

- The `3.80x` speedup and `94.9%` efficiency characterize orchestration overhead using controlled
  fixed-duration tasks. They must not be described as fMRIPrep compute acceleration.
- The `7.05 subjects/hour` number is an observation from one saturated four-job wave on the current
  server. It may change with image dimensions, modalities, storage bandwidth, CPU and memory limits.
- Increasing concurrency beyond available memory or I/O bandwidth may reduce throughput or make
  jobs fail. `max_workers` therefore remains a configurable resource bound rather than an unlimited
  queue consumer count.
- A future capacity test should run the same uncached subjects at 1, 2, and 4 workers while recording
  CPU, RAM, disk throughput, failure rate, and cost per subject.

## Resume-safe statement

Implemented bounded subject-level parallel scheduling with LangGraph `Send`; a repeatable
12-job scheduler benchmark achieved 3.80x speedup at four workers with 94.9% parallel efficiency,
while a real seven-subject fMRIPrep run reached four concurrent jobs and 4.16 subjects/hour overall.
