from __future__ import annotations

import gzip
import json
import os
import struct
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

from neuro_preprocess_agent.agents.supervisor import Supervisor, SupervisorDecision
from neuro_preprocess_agent.cli import _drive
from neuro_preprocess_agent.config import AppConfig
from neuro_preprocess_agent.db import record_run
from neuro_preprocess_agent.graph import (
    WORKERS,
    _apply_request_overrides,
    build_graph,
    graph,
    qc_review_node,
    route_after_preprocess,
)
from neuro_preprocess_agent.interaction import (
    parse_human_response,
    parse_input_qc_response,
)
from neuro_preprocess_agent.runtime import AgentRuntime, interrupt_message
from neuro_preprocess_agent.state import PipelineState
from neuro_preprocess_agent.tools.preprocess import (
    apply_skull_strip_actions,
    inspect_anatomical_inputs,
    inspect_bids_dataset,
    prepare_preprocessing,
    run_subject_preprocessing,
    validate_bids_with_container,
)
from neuro_preprocess_agent.tools.qc import run_qc
from neuro_preprocess_agent.tools.qc_metrics import compute_quantitative_metrics
from neuro_preprocess_agent.tools.registry import WorkerPolicyError
from neuro_preprocess_agent.tools.source import fetch_data
from neuro_preprocess_agent.tools.workers import (
    build_worker_registry,
    generate_report,
    write_database,
)


def write_test_nifti(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(352)
    dimensions = [len(shape), *shape, *([1] * (7 - len(shape)))]
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, *dimensions)
    struct.pack_into("<h", header, 70, 4)
    struct.pack_into("<h", header, 72, 16)
    with gzip.open(path, "wb") as file_handle:
        file_handle.write(header)
        file_handle.write(os.urandom(2048))


class RuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "config.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_config(
        self,
        *,
        mode: str = "mock",
        min_output_items: int = 1,
        source: dict | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        config = {
            "runtime": {"mode": mode},
            "supervisor": {"mode": "rule"},
            "source": source or {"name": "test", "type": "openneuro", "dataset_id": "ds000001"},
            "storage": {
                "raw_dir": str(self.root / "raw"),
                "processed_dir": str(self.root / "processed"),
                "report_dir": str(self.root / "reports"),
            },
            "preprocess": {
                "bids_dir": str(self.root / "bids"),
                "derivatives_dir": str(self.root / "derivatives"),
                "log_dir": str(self.root / "logs"),
                "work_dir": str(self.root / "work"),
                "lock_dir": str(self.root / "locks"),
                "dcm2bids": "/bin/false",
                "dcm2bids_config": str(self.root / "dcm2bids.json"),
                "freesurfer_license": str(self.root / "license.txt"),
                "fmriprep_image": "nipreps/fmriprep:latest",
                "max_workers": 2,
                "subjects": subjects or ["001"],
            },
            "qc": {"min_output_items": min_output_items},
            "database": {"backend": "jsonl", "jsonl_path": str(self.root / "runs.jsonl")},
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def runtime(self) -> AgentRuntime:
        return AgentRuntime(build_graph(checkpointer=InMemorySaver()))

    def test_mock_pipeline_interrupt_and_complete(self) -> None:
        self.write_config()
        runtime = self.runtime()
        result = runtime.start("帮我处理 OpenNeuro ds000002", self.config_path, "mock-success")
        self.assertIn("执行计划", interrupt_message(result) or "")

        result = runtime.resume(True, "mock-success")
        self.assertEqual(result["report"]["status"], "mocked")
        self.assertTrue(result["qc_result"]["passed"])
        self.assertEqual(result["db_result"]["status"], "planned")
        self.assertEqual(result["source"]["dataset_id"], "ds000002")

    def test_plan_modification_requires_second_confirmation(self) -> None:
        self.write_config()
        runtime = self.runtime()
        runtime.start("处理 ds000001", self.config_path, "modify-plan")
        result = runtime.resume({"preprocess": {"max_workers": 1}}, "modify-plan")
        self.assertIn("并行被试数：1", interrupt_message(result) or "")
        result = runtime.resume(True, "modify-plan")
        self.assertEqual(result["processed_data"]["max_workers"], 1)

    def test_subject_jobs_are_fanned_out_and_aggregated(self) -> None:
        self.write_config(subjects=["001", "002", "003"])
        runtime = self.runtime()
        runtime.start("处理 ds000001", self.config_path, "subject-fanout")
        result = runtime.resume(True, "subject-fanout")
        subject_results = result["processed_data"]["subject_results"]
        self.assertEqual([item["subject"] for item in subject_results], ["001", "002", "003"])
        self.assertTrue(all(item["status"] == "planned" for item in subject_results))
        batches = [event for event in result["events"] if event["node"] == "preprocess_batch"]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[-1]["completed_subjects"], 3)

    def test_interrupt_validates_direct_string_and_unknown_payload(self) -> None:
        self.write_config()
        runtime = self.runtime()
        runtime.start("处理 ds000001", self.config_path, "approval-validation")
        invalid = runtime.resume({"unknown": "value"}, "approval-validation")
        self.assertIn("没有识别", interrupt_message(invalid) or "")
        cancelled = runtime.resume("停止", "approval-validation")
        self.assertEqual(cancelled["report"]["status"], "cancelled")

    def test_qc_failure_skips_database(self) -> None:
        self.write_config(min_output_items=5)
        runtime = self.runtime()
        runtime.start("处理 ds000001", self.config_path, "qc-fail")
        result = runtime.resume(True, "qc-fail")
        self.assertFalse(result["qc_result"]["passed"])
        self.assertEqual(result["report"]["status"], "qc_failed")
        self.assertIn("db", result["skipped_steps"])
        self.assertNotIn("db_result", result)

    def test_worker_failure_becomes_failure_report(self) -> None:
        missing = self.root / "does-not-exist"
        self.write_config(mode="run", source={"name": "missing", "type": "local_path", "path": str(missing)})
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["preflight"] = {"enabled": False}
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        runtime = self.runtime()
        runtime.start(f"处理本地数据 {missing}", self.config_path, "worker-fail")
        result = runtime.resume(True, "worker-fail")
        self.assertEqual(result["report"]["status"], "failed")
        self.assertEqual(result["errors"][0]["node"], "source")

    def test_preprocess_failure_skips_input_qc(self) -> None:
        self.assertEqual(route_after_preprocess({"errors": [{"node": "preprocess"}]}), "supervisor")
        self.assertEqual(route_after_preprocess({"preprocess_context": {}}), "input_qc")

    def test_natural_language_approval_parser(self) -> None:
        parsed = parse_human_response("数据来源改成本地/tmp/demo，并行数改成3，跳过入库")
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["source"]["path"], "/tmp/demo")
        self.assertEqual(parsed["preprocess"]["max_workers"], 3)
        self.assertNotIn("db", parsed["execution_plan"])

    def test_worker_permission_denied(self) -> None:
        registry = build_worker_registry()
        state = {"config": {"permissions": {"source": "deny"}}}
        with self.assertRaises(WorkerPolicyError):
            registry.execute("source", state)  # type: ignore[arg-type]

    def test_jsonl_database_write_is_idempotent(self) -> None:
        path = self.root / "records.jsonl"
        config = {
            "project_root": str(self.root),
            "storage": {"report_dir": "reports"},
            "database": {"backend": "jsonl", "jsonl_path": str(path)},
        }
        state = {"run_id": "same-run", "request": "test", "qc_result": {"passed": True}}
        first = record_run(config, state)
        second = record_run(config, state)
        self.assertEqual(first["status"], "written")
        self.assertEqual(second["status"], "updated")
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_concurrent_jsonl_writes_do_not_lose_runs(self) -> None:
        path = self.root / "concurrent.jsonl"
        config = {
            "project_root": str(self.root),
            "storage": {"report_dir": "reports"},
            "database": {"backend": "jsonl", "jsonl_path": str(path)},
        }

        def write(index: int) -> None:
            record_run(config, {"run_id": f"run-{index}", "request": "test", "qc_result": {"passed": True}})

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write, range(20)))

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 20)
        self.assertEqual({record["run_id"] for record in records}, {f"run-{index}" for index in range(20)})

    def test_platform_graph_does_not_embed_local_checkpointer(self) -> None:
        self.assertIsNone(graph.checkpointer)

    def test_unknown_configuration_field_is_rejected(self) -> None:
        self.write_config()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["preprocess"]["max_worker"] = 8
        with self.assertRaisesRegex(ValidationError, "max_worker"):
            AppConfig.model_validate(config)

    def test_llm_plan_is_constrained_by_pipeline_dependencies(self) -> None:
        class FakeModel:
            def __init__(self, **_: object) -> None:
                pass

            def with_structured_output(self, _: object) -> FakeModel:
                return self

            def invoke(self, _: object) -> SupervisorDecision:
                return SupervisorDecision(
                    execution_plan=["db", "preprocess"],
                    next_node="preprocess",
                    reason="fake decision",
                )

        self.write_config()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["supervisor"] = {"mode": "llm", "provider": "deepseek", "fallback_to_rule": False}
        state = {"request": "完整处理数据", "config": config, "stage": "load_config"}
        with patch("langchain_deepseek.ChatDeepSeek", FakeModel):
            decision = Supervisor(build_worker_registry()).decide(state)  # type: ignore[arg-type]
        self.assertEqual(
            decision["execution_plan"],
            ["preflight", "source", "fetch", "preprocess", "qc", "db", "report"],
        )
        self.assertEqual(decision["next_node"], "preflight")

    def test_preflight_blocks_missing_local_source_before_workers_run(self) -> None:
        missing = self.root / "missing-before-fetch"
        self.write_config(mode="run", source={"name": "missing", "type": "local_path", "path": str(missing)})
        runtime = self.runtime()
        runtime.start(f"处理本地数据 {missing}", self.config_path, "preflight-missing-source")

        result = runtime.resume(True, "preflight-missing-source")

        self.assertEqual(result["report"]["status"], "failed")
        self.assertFalse(result["preflight_result"]["passed"])
        self.assertIn("source_readable", result["preflight_result"]["hard_failures"])
        self.assertEqual(result["errors"][0]["node"], "preflight")
        self.assertNotIn("raw_data", result)

    def test_preflight_warning_requires_human_confirmation(self) -> None:
        self.write_config()
        original = WORKERS.get("preflight")

        def warning(_state):
            return {
                "status": "passed",
                "passed": True,
                "checks": [],
                "hard_failures": [],
                "warnings": ["cpu_budget"],
                "resources": {"requested_cpu": 128, "cpu_count": 96},
            }

        WORKERS._workers["preflight"] = replace(original, handler=warning)
        try:
            runtime = self.runtime()
            runtime.start("处理 ds000001", self.config_path, "preflight-warning")
            paused = runtime.resume(True, "preflight-warning")
            self.assertIn("资源警告", interrupt_message(paused) or "")
            result = runtime.resume("停止", "preflight-warning")
        finally:
            WORKERS._workers["preflight"] = original

        self.assertEqual(result["report"]["status"], "cancelled")
        self.assertTrue(result["preflight_reviewed"])
        self.assertNotIn("raw_data", result)

    def test_database_transient_failure_recovers(self) -> None:
        self.write_config()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["database"]["retry_interval_seconds"] = 0
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        original = WORKERS.get("db")
        attempts = 0

        def fail_once(state):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary database outage")
            return original.handler(state)

        WORKERS._workers["db"] = replace(original, handler=fail_once)
        try:
            runtime = self.runtime()
            runtime.start("处理 ds000001", self.config_path, "db-recovery")
            result = runtime.resume(True, "db-recovery")
        finally:
            WORKERS._workers["db"] = original

        self.assertEqual(result["report"]["status"], "mocked")
        self.assertEqual(result["db_result"]["attempts"], 2)
        self.assertEqual(attempts, 2)

    def test_database_retry_exhaustion_still_generates_report(self) -> None:
        self.write_config()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["database"].update({"max_attempts": 3, "retry_interval_seconds": 0})
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        original = WORKERS.get("db")

        def always_fail(_state):
            raise ConnectionError("database unavailable")

        WORKERS._workers["db"] = replace(original, handler=always_fail)
        try:
            runtime = self.runtime()
            runtime.start("处理 ds000001", self.config_path, "db-exhausted")
            result = runtime.resume(True, "db-exhausted")
        finally:
            WORKERS._workers["db"] = original

        self.assertEqual(result["report"]["status"], "failed")
        self.assertEqual(result["db_result"], {"status": "failed", "backend": "jsonl", "attempts": 3})
        self.assertEqual(result["errors"][0]["node"], "db")
        self.assertTrue(result["errors"][0]["exhausted"])

    def test_local_dry_run_inspects_original_bids_directory(self) -> None:
        bids_dir = self.root / "input_bids"
        anat_dir = bids_dir / "sub-01" / "anat"
        anat_dir.mkdir(parents=True)
        (bids_dir / "dataset_description.json").write_text(
            json.dumps({"Name": "Test dataset", "BIDSVersion": "1.9.0"}), encoding="utf-8"
        )
        (anat_dir / "sub-01_T1w.nii.gz").write_bytes(b"non-empty test image")
        self.write_config(
            mode="dry_run",
            source={"name": "local", "type": "local_path", "path": str(bids_dir)},
            subjects=["01"],
        )

        runtime = self.runtime()
        runtime.start(f"检查本地数据 {bids_dir}，跳过入库", self.config_path, "local-dry-run")
        result = runtime.resume(True, "local-dry-run")

        self.assertEqual(result["raw_data"]["location"], str(bids_dir))
        self.assertNotEqual(result["raw_data"]["location"], result["raw_data"]["planned_location"])
        self.assertTrue(result["processed_data"]["bids_preflight"]["passed"])
        self.assertEqual(result["processed_data"]["subjects"], ["01"])
        self.assertEqual(
            result["processed_data"]["subject_modalities"]["01"],
            {"anat": True, "func": False, "t1w_count": 1, "bold_count": 0},
        )

    def test_converted_nifti_bids_directory_is_run_isolated(self) -> None:
        nifti_dir = self.root / "nifti_input"
        write_test_nifti(nifti_dir / "subject_T1w.nii.gz", (2, 2, 2))
        self.write_config(
            mode="dry_run",
            source={"name": "nifti", "type": "local_path", "path": str(nifti_dir)},
        )

        runtime = self.runtime()
        runtime.start(f"检查本地数据 {nifti_dir}，跳过入库", self.config_path, "nifti-isolation")
        result = runtime.resume(True, "nifti-isolation")

        self.assertEqual(
            Path(result["processed_data"]["bids_dir"]),
            self.root / "bids" / result["run_id"],
        )

    def test_thread_id_cannot_be_reused_or_resumed_after_completion(self) -> None:
        self.write_config()
        runtime = self.runtime()
        runtime.start("处理 ds000001", self.config_path, "thread-lifecycle")
        with self.assertRaisesRegex(ValueError, "already exists"):
            runtime.start("另一个任务", self.config_path, "thread-lifecycle")
        runtime.resume(True, "thread-lifecycle")
        with self.assertRaisesRegex(ValueError, "already completed"):
            runtime.resume(True, "thread-lifecycle")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            runtime.resume(True, "missing-thread")

    def test_bids_preflight_rejects_bold_without_repetition_time(self) -> None:
        bids_dir = self.root / "invalid_bids"
        func_dir = bids_dir / "sub-01" / "func"
        func_dir.mkdir(parents=True)
        (bids_dir / "dataset_description.json").write_text(
            json.dumps({"Name": "Test dataset", "BIDSVersion": "1.9.0"}), encoding="utf-8"
        )
        (func_dir / "sub-01_task-rest_bold.nii.gz").write_bytes(b"non-empty test image")
        (func_dir / "sub-01_task-rest_bold.json").write_text(
            json.dumps({"TaskName": "rest"}), encoding="utf-8"
        )

        result = inspect_bids_dataset(bids_dir)

        self.assertFalse(result["passed"])
        self.assertTrue(any("RepetitionTime" in error for error in result["errors"]))

    def test_container_bids_validator_errors_are_structured(self) -> None:
        payload = {
            "issues": {
                "errors": [{
                    "reason": "Invalid JSON file",
                    "files": [{
                        "file": {"relativePath": "/task-demo_bold.json"},
                        "evidence": ".Field should match format uri",
                    }],
                }],
                "warnings": [],
            },
            "summary": {"subjects": ["01"]},
        }
        completed = subprocess.CompletedProcess(["docker"], 1, stdout=json.dumps(payload), stderr="")
        with patch("neuro_preprocess_agent.tools.preprocess.subprocess.run", return_value=completed):
            result = validate_bids_with_container(self.root, "nipreps/fmriprep:test")
        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"], ["/task-demo_bold.json: .Field should match format uri"])

    def test_t1_only_qc_does_not_require_functional_outputs(self) -> None:
        derivatives = self.root / "derivatives"
        subject_dir = derivatives / "sub-01"
        anat_dir = subject_dir / "anat"
        figures_dir = subject_dir / "figures"
        anat_dir.mkdir(parents=True)
        figures_dir.mkdir(parents=True)
        for filename in ("sub-01_desc-preproc_T1w.nii.gz", "sub-01_desc-brain_mask.nii.gz"):
            write_test_nifti(anat_dir / filename, (2, 2, 2))
        (derivatives / "sub-01.html").write_text("fMRIPrep report", encoding="utf-8")
        for index in range(3):
            (figures_dir / f"figure-{index}.svg").write_text("<svg></svg>", encoding="utf-8")
        log_path = self.root / "sub-01.log"
        reuse_evidence = derivatives / ".neuro_preprocess_agent/sub-01.complete.json"
        reuse_evidence.parent.mkdir(parents=True)
        reuse_evidence.write_text("{}", encoding="utf-8")
        state = {
            "config": {
                "runtime": {"mode": "run"},
                "qc": {"engine": "rule_file_image_qc", "min_output_items": 1, "min_visual_artifacts": 3},
            },
            "processed_data": {
                "output_items": 2,
                "fmriprep_dir": str(derivatives),
                "subjects": ["01"],
                "subject_modalities": {"01": {"anat": True, "func": False}},
                "subject_results": [{
                    "subject": "01",
                    "returncode": 0,
                    "status": "reused",
                    "log_file": str(log_path),
                    "reuse_evidence": str(reuse_evidence),
                }],
                "bids_preflight": {"status": "completed", "passed": True, "errors": [], "warnings": []},
            },
        }

        result = run_qc(state)  # type: ignore[arg-type]

        self.assertTrue(result["passed"])
        names = {item["name"] for item in result["subject_qc"][0]["checks"]}
        self.assertFalse(any("bold" in name or "confounds" in name for name in names))

    def test_motion_thresholds_require_review_and_block_database(self) -> None:
        derivatives = self.root / "derivatives"
        func_dir = derivatives / "sub-01" / "func"
        figures_dir = derivatives / "sub-01" / "figures"
        write_test_nifti(func_dir / "sub-01_task-rest_desc-preproc_bold.nii.gz", (2, 2, 2, 4))
        write_test_nifti(func_dir / "sub-01_task-rest_desc-brain_mask.nii.gz", (2, 2, 2))
        figures_dir.mkdir(parents=True)
        (derivatives / "sub-01.html").write_text("report", encoding="utf-8")
        for index in range(3):
            (figures_dir / f"figure-{index}.svg").write_text("<svg></svg>", encoding="utf-8")
        confounds = func_dir / "sub-01_task-rest_desc-confounds_timeseries.tsv"
        confounds.write_text(
            "framewise_displacement\tstd_dvars\tdvars\n"
            "n/a\tn/a\tn/a\n"
            "1.0\t2.0\t10\n"
            "1.0\t2.0\t10\n",
            encoding="utf-8",
        )
        log_path = self.root / "sub-01.log"
        log_path.write_text("fMRIPrep finished successfully!", encoding="utf-8")
        config = {
            "runtime": {"mode": "run"},
            "storage": {"report_dir": str(self.root / "reports")},
            "qc": {
                "engine": "rule_file_image_qc",
                "min_output_items": 1,
                "min_visual_artifacts": 3,
                "fd_threshold_mm": 0.5,
                "max_mean_fd_mm": 0.5,
                "max_fd_outlier_fraction": 0.2,
                "max_mean_std_dvars": 1.5,
                "block_database_on_review": True,
            },
            "database": {"backend": "jsonl", "jsonl_path": str(self.root / "records.jsonl")},
        }
        state = {
            "run_id": "motion-review",
            "request": "test",
            "config": config,
            "plan_approved": True,
            "processed_data": {
                "output_items": 3,
                "fmriprep_dir": str(derivatives),
                "subjects": ["01"],
                "subject_modalities": {"01": {"anat": False, "func": True}},
                "subject_results": [{"subject": "01", "returncode": 0, "log_file": str(log_path)}],
                "bids_preflight": {"status": "completed", "passed": True, "errors": [], "warnings": []},
            },
        }

        qc_result = run_qc(state)  # type: ignore[arg-type]
        state["qc_result"] = qc_result
        db_result = write_database(state)  # type: ignore[arg-type]
        report = generate_report(state)  # type: ignore[arg-type]

        self.assertTrue(qc_result["hard_checks_passed"])
        self.assertTrue(qc_result["review_required"])
        self.assertFalse(qc_result["approved_for_database"])
        self.assertEqual(db_result["status"], "blocked_by_qc")
        self.assertEqual(report["status"], "review_required")

    def test_qc_discovers_functional_outputs_in_multiple_sessions(self) -> None:
        derivatives = self.root / "derivatives"
        anat_dir = derivatives / "sub-01" / "anat"
        figures_dir = derivatives / "sub-01" / "figures"
        write_test_nifti(anat_dir / "sub-01_desc-preproc_T1w.nii.gz", (2, 2, 2))
        write_test_nifti(anat_dir / "sub-01_desc-brain_mask.nii.gz", (2, 2, 2))
        for session in ("test", "retest"):
            func_dir = derivatives / "sub-01" / f"ses-{session}" / "func"
            prefix = f"sub-01_ses-{session}_task-demo"
            write_test_nifti(func_dir / f"{prefix}_space-MNI_desc-preproc_bold.nii.gz", (2, 2, 2, 4))
            write_test_nifti(func_dir / f"{prefix}_space-MNI_desc-brain_mask.nii.gz", (2, 2, 2))
            (func_dir / f"{prefix}_desc-confounds_timeseries.tsv").write_text(
                "framewise_displacement\tstd_dvars\n0.1\t1.0\n", encoding="utf-8"
            )
        figures_dir.mkdir(parents=True)
        for index in range(3):
            (figures_dir / f"figure-{index}.svg").write_text("<svg></svg>", encoding="utf-8")
        (derivatives / "sub-01.html").write_text("report", encoding="utf-8")
        log_path = self.root / "sub-01.log"
        log_path.write_text("fMRIPrep finished successfully!", encoding="utf-8")
        state = {
            "run_id": "multi-session-qc",
            "config": {
                "runtime": {"mode": "run"},
                "qc": {
                    "engine": "rule_file_image_qc", "min_output_items": 1,
                    "min_visual_artifacts": 3, "quantitative": {"enabled": False},
                },
            },
            "processed_data": {
                "output_items": 8, "fmriprep_dir": str(derivatives), "subjects": ["01"],
                "subject_modalities": {"01": {"anat": True, "func": True}},
                "subject_sessions": {"01": ["retest", "test"]},
                "subject_results": [{"subject": "01", "returncode": 0, "status": "completed", "log_file": str(log_path)}],
                "bids_preflight": {"status": "completed", "passed": True, "errors": [], "warnings": []},
            },
        }

        result = run_qc(state)  # type: ignore[arg-type]

        self.assertTrue(result["passed"])
        names = {item["name"] for item in result["subject_qc"][0]["checks"]}
        self.assertTrue(any("ses-test" in name for name in names))
        self.assertTrue(any("ses-retest" in name for name in names))

    def test_spatial_shape_mismatch_is_hard_qc_failure(self) -> None:
        derivatives = self.root / "derivatives"
        anat_dir = derivatives / "sub-01" / "anat"
        figures_dir = derivatives / "sub-01" / "figures"
        write_test_nifti(anat_dir / "sub-01_desc-preproc_T1w.nii.gz", (2, 2, 2))
        write_test_nifti(anat_dir / "sub-01_desc-brain_mask.nii.gz", (3, 3, 3))
        figures_dir.mkdir(parents=True)
        (derivatives / "sub-01.html").write_text("report", encoding="utf-8")
        for index in range(3):
            (figures_dir / f"figure-{index}.svg").write_text("<svg></svg>", encoding="utf-8")
        log_path = self.root / "sub-01.log"
        log_path.write_text("fMRIPrep finished successfully!", encoding="utf-8")
        state = {
            "config": {
                "runtime": {"mode": "run"},
                "qc": {"engine": "rule_file_image_qc", "min_output_items": 1, "min_visual_artifacts": 3},
            },
            "processed_data": {
                "output_items": 2,
                "fmriprep_dir": str(derivatives),
                "subjects": ["01"],
                "subject_modalities": {"01": {"anat": True, "func": False}},
                "subject_results": [{"subject": "01", "returncode": 0, "log_file": str(log_path)}],
                "bids_preflight": {"status": "completed", "passed": True, "errors": [], "warnings": []},
            },
        }

        result = run_qc(state)  # type: ignore[arg-type]

        self.assertFalse(result["hard_checks_passed"])
        self.assertEqual(result["status"], "failed")
        shape_check = next(check for check in result["checks"] if "spatial_shape_match" in check["name"])
        self.assertFalse(shape_check["passed"])

    def test_quantitative_qc_computes_mask_tsnr_and_overlap(self) -> None:
        import nibabel as nib
        import numpy as np

        shape = (5, 5, 5)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        bold = np.stack([mask * value for value in (10, 11, 9, 10)], axis=3).astype(np.float32)
        paths = {
            "preproc_t1w": self.root / "t1w.nii.gz",
            "t1w_brain_mask": self.root / "t1w_mask.nii.gz",
            "preproc_bold": self.root / "bold.nii.gz",
            "bold_brain_mask": self.root / "bold_mask.nii.gz",
        }
        nib.save(nib.Nifti1Image(mask.astype(np.float32) * 100, np.eye(4)), paths["preproc_t1w"])
        nib.save(nib.Nifti1Image(mask, np.eye(4)), paths["t1w_brain_mask"])
        nib.save(nib.Nifti1Image(bold, np.eye(4)), paths["preproc_bold"])
        nib.save(nib.Nifti1Image(mask, np.eye(4)), paths["bold_brain_mask"])

        result = compute_quantitative_metrics(
            "01",
            paths,
            {
                "enabled": True,
                "max_nonfinite_fraction": 0.0,
                "min_mask_voxels": 10,
                "max_mask_edge_fraction": 0.0,
                "min_bold_tsnr": 5.0,
                "min_anat_bold_mask_dice": 0.9,
            },
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["images"]["t1w_brain_mask"]["mask_voxels"], 27)
        self.assertGreater(result["images"]["bold_tsnr"]["median"], 5.0)
        self.assertEqual(result["images"]["anat_bold_mask_overlap"]["dice"], 1.0)
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_quantitative_qc_flags_nonfinite_data_and_disjoint_masks(self) -> None:
        import nibabel as nib
        import numpy as np

        anat_mask = np.zeros((5, 5, 5), dtype=np.uint8)
        bold_mask = np.zeros((5, 5, 5), dtype=np.uint8)
        anat_mask[1:3, 1:3, 1:3] = 1
        bold_mask[3:5, 3:5, 3:5] = 1
        t1w = anat_mask.astype(np.float32)
        t1w[0, 0, 0] = np.nan
        paths = {
            "preproc_t1w": self.root / "bad_t1w.nii.gz",
            "t1w_brain_mask": self.root / "bad_t1w_mask.nii.gz",
            "preproc_bold": self.root / "bad_bold.nii.gz",
            "bold_brain_mask": self.root / "bad_bold_mask.nii.gz",
        }
        nib.save(nib.Nifti1Image(t1w, np.eye(4)), paths["preproc_t1w"])
        nib.save(nib.Nifti1Image(anat_mask, np.eye(4)), paths["t1w_brain_mask"])
        nib.save(nib.Nifti1Image(np.repeat(bold_mask[..., None], 4, axis=3), np.eye(4)), paths["preproc_bold"])
        nib.save(nib.Nifti1Image(bold_mask, np.eye(4)), paths["bold_brain_mask"])

        result = compute_quantitative_metrics(
            "bad",
            paths,
            {
                "enabled": True,
                "max_nonfinite_fraction": 0.0,
                "min_mask_voxels": 1,
                "max_mask_edge_fraction": None,
                "min_bold_tsnr": None,
                "min_anat_bold_mask_dice": 0.8,
            },
        )
        checks = {item["name"]: item for item in result["checks"]}

        self.assertFalse(checks["bad:nonfinite_fraction:preproc_t1w"]["passed"])
        self.assertEqual(checks["bad:nonfinite_fraction:preproc_t1w"]["severity"], "error")
        self.assertFalse(checks["bad:anat_bold_mask_dice"]["passed"])
        self.assertEqual(checks["bad:anat_bold_mask_dice"]["severity"], "warning")

    def test_completed_subject_is_reused_without_running_command(self) -> None:
        derivatives = self.root / "derivatives"
        subject_dir = derivatives / "sub-01" / "anat"
        subject_dir.mkdir(parents=True)
        (derivatives / "sub-01.html").write_text("report", encoding="utf-8")
        (subject_dir / "sub-01_desc-preproc_T1w.nii.gz").write_bytes(b"output")
        log_path = self.root / "logs" / "sub-01.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text("fMRIPrep finished successfully!", encoding="utf-8")
        state = {
            "preprocess_job": {
                "run_id": "same-run",
                "subject": "01",
                "mode": "run",
                "command": ["/bin/false"],
                "log_file": str(log_path),
                "lock_file": str(self.root / "locks" / "sub-01.lock"),
                "fmriprep_dir": str(derivatives),
                "reuse_completed": True,
            }
        }

        result = run_subject_preprocessing(state)  # type: ignore[arg-type]

        subject_result = result["preprocess_subject_results"][0]
        self.assertEqual(subject_result["status"], "reused")
        self.assertEqual(subject_result["returncode"], 0)
        self.assertTrue((derivatives / ".neuro_preprocess_agent/sub-01.complete.json").is_file())

    def test_qc_review_interrupt_records_approval_note(self) -> None:
        builder = StateGraph(PipelineState)
        builder.add_node("qc_review", qc_review_node)
        builder.add_node("supervisor", lambda state: {"stage": "end"})
        builder.add_node("report", lambda state: {"stage": "end"})
        builder.add_edge(START, "qc_review")
        builder.add_edge("supervisor", END)
        builder.add_edge("report", END)
        review_graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "qc-review-approval"}}
        state = {
            "run_id": "review-run",
            "config": {"qc": {"require_review_note": True}},
            "execution_plan": ["qc", "db", "report"],
            "qc_result": {
                "status": "review_required",
                "hard_checks_passed": True,
                "review_required": True,
                "approved_for_database": False,
                "passed": False,
                "failed_checks": ["mean FD exceeded review threshold"],
            },
        }

        paused = review_graph.invoke(state, config)
        self.assertIn("自动 QC", interrupt_message(paused) or "")
        completed = review_graph.invoke(Command(resume="人工通过：已检查 HTML 配准和运动曲线"), config)

        self.assertTrue(completed["qc_result"]["approved_for_database"])
        self.assertEqual(completed["qc_review"]["decision"], "approve")
        self.assertIn("HTML", completed["qc_review"]["note"])

    def test_pipeline_routes_soft_qc_warning_through_review_before_database(self) -> None:
        self.write_config()
        original = WORKERS.get("qc")

        def soft_warning(_state):
            return {
                "status": "review_required",
                "passed": False,
                "hard_checks_passed": True,
                "review_required": True,
                "approved_for_database": False,
                "failed_checks": ["01:motion_review_thresholds"],
                "checks": [],
                "subject_qc": [],
                "visual_qc_artifacts": [],
            }

        WORKERS._workers["qc"] = replace(original, handler=soft_warning)
        try:
            runtime = self.runtime()
            runtime.start("处理 ds000001", self.config_path, "soft-qc-route")
            paused = runtime.resume(True, "soft-qc-route")
            self.assertIn("自动 QC", interrupt_message(paused) or "")
            result = runtime.resume("人工通过：已人工检查预处理报告", "soft-qc-route")
        finally:
            WORKERS._workers["qc"] = original

        self.assertEqual(result["qc_review"]["decision"], "approve")
        self.assertEqual(result["db_result"]["status"], "planned")
        self.assertTrue(result["qc_result"]["approved_for_database"])

    def test_fmriprep_managed_options_cannot_be_overridden_by_raw_args(self) -> None:
        with self.assertRaises(ValidationError):
            AppConfig.model_validate({
                "source": {"type": "openneuro", "dataset_id": "ds000001"},
                "preprocess": {"fmriprep_args": ["--nprocs", "64"]},
            })

    def test_fmriprep_runs_as_current_user(self) -> None:
        self.write_config(mode="mock", subjects=["001"])
        config = AppConfig.model_validate_json(self.config_path.read_text(encoding="utf-8")).model_dump(mode="json")
        config["project_root"] = str(self.root)

        output = prepare_preprocessing({
            "run_id": "docker-user-run",
            "config": config,
            "raw_data": {"location": str(self.root / "not-created-in-mock-mode")},
        })
        command = output["preprocess_jobs"][0]["command"]
        user_index = command.index("--user")
        skull_strip_index = command.index("--skull-strip-t1w")

        self.assertEqual(command[user_index + 1], f"{os.getuid()}:{os.getgid()}")
        self.assertEqual(command[skull_strip_index + 1], "auto")

    def test_input_qc_flags_pre_skull_stripped_t1w_for_review(self) -> None:
        import nibabel as nib
        import numpy as np

        bids_dir = self.root / "input-qc-bids"
        for subject, data in (
            ("01", np.ones((16, 16, 16), dtype=np.float32)),
            ("02", np.pad(np.ones((4, 4, 4), dtype=np.float32), 6)),
        ):
            path = bids_dir / f"sub-{subject}" / "anat" / f"sub-{subject}_T1w.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        config = AppConfig.model_validate({
            "runtime": {"mode": "run"},
            "source": {"type": "local_path", "path": str(bids_dir)},
            "preprocess": {"pre_skull_stripped_nonzero_fraction": 0.3},
        }).model_dump(mode="json")
        result = inspect_anatomical_inputs({
            "config": config,
            "preprocess_context": {
                "mode": "run",
                "run_fmriprep": True,
                "bids_dir": str(bids_dir),
                "subjects": ["01", "02"],
            },
        })

        self.assertTrue(result["review_required"])
        self.assertEqual(result["recommended_actions"], {"01": "force", "02": "skip"})
        self.assertEqual(result["subject_results"][1]["classification"], "suspected_pre_skull_stripped")

    def test_input_qc_actions_rewrite_command_and_fingerprint(self) -> None:
        job = {
            "subject": "04",
            "command": ["fmriprep", "--skull-strip-t1w", "auto"],
            "fmriprep_args": ["--skull-strip-t1w", "auto"],
            "skull_strip_t1w": "auto",
            "bids_dir": "/data/bids",
            "fmriprep_image": "nipreps/fmriprep:latest",
            "configuration_fingerprint": "old",
        }
        updated = apply_skull_strip_actions([job], {"04": "skip"})[0]

        self.assertEqual(updated["command"][-1], "skip")
        self.assertEqual(updated["fmriprep_args"][-1], "skip")
        self.assertNotEqual(updated["configuration_fingerprint"], "old")
        self.assertEqual(job["command"][-1], "auto")

    def test_input_qc_natural_language_review_parser(self) -> None:
        self.assertEqual(
            parse_input_qc_response("sub-04 使用 skip", ["01", "04"]),
            {"approved": True, "subject_actions": {"04": "skip"}},
        )
        self.assertEqual(
            parse_input_qc_response("所有被试使用 auto", ["01", "04"]),
            {"approved": True, "subject_actions": {"01": "auto", "04": "auto"}},
        )

    def test_input_qc_interrupt_applies_human_subject_override(self) -> None:
        self.write_config(subjects=["001"])
        runtime = self.runtime()
        runtime.start("处理 ds000001", self.config_path, "input-qc-review")
        screening = {
            "status": "review_required",
            "review_required": True,
            "subject_results": [{
                "subject": "001",
                "classification": "suspected_pre_skull_stripped",
                "recommended_action": "skip",
                "review_required": True,
                "images": [{"nonzero_fraction": 0.12}],
            }],
            "recommended_actions": {"001": "skip"},
        }
        with patch("neuro_preprocess_agent.graph.inspect_anatomical_inputs", return_value=screening):
            paused = runtime.resume(True, "input-qc-review")
        self.assertIn("T1w 输入", interrupt_message(paused) or "")
        guarded = _drive(runtime, paused, "input-qc-review", interactive=False, auto_approve=True)
        self.assertIsNotNone(interrupt_message(guarded))

        result = runtime.resume("sub-001 使用 auto", "input-qc-review")
        self.assertEqual(result["input_qc_result"]["final_actions"], {"001": "auto"})
        self.assertEqual(result["processed_data"]["subject_results"][0]["status"], "planned")

    def test_nifti_manifest_supports_multiple_subjects_and_sessions(self) -> None:
        config = AppConfig.model_validate({
            "source": {"type": "local_path", "path": "/tmp/input"},
            "preprocess": {
                "nifti_manifest": [
                    {"path": "a.nii.gz", "participant_label": "01", "session_label": "baseline", "datatype": "anat", "suffix": "T1w"},
                    {"path": "b.nii.gz", "participant_label": "02", "session_label": "followup", "datatype": "func", "suffix": "bold", "task": "rest", "metadata": {"RepetitionTime": 2.0}},
                ]
            },
        })

        self.assertEqual(len(config.preprocess.nifti_manifest), 2)
        self.assertEqual(config.preprocess.nifti_manifest[1].session_label, "followup")

    def test_nifti_manifest_writes_multi_subject_session_bids(self) -> None:
        source = self.root / "nifti_source"
        write_test_nifti(source / "anatomical.nii.gz", (2, 2, 2))
        write_test_nifti(source / "functional.nii.gz", (2, 2, 2, 3))
        config = AppConfig.model_validate({
            "runtime": {"mode": "run"},
            "source": {"type": "local_path", "path": str(source)},
            "storage": {
                "raw_dir": str(self.root / "raw"),
                "processed_dir": str(self.root / "processed"),
                "report_dir": str(self.root / "reports"),
            },
            "preprocess": {
                "input_format": "nifti",
                "run_fmriprep": False,
                "bids_dir": str(self.root / "bids"),
                "derivatives_dir": str(self.root / "derivatives"),
                "log_dir": str(self.root / "logs"),
                "work_dir": str(self.root / "work"),
                "lock_dir": str(self.root / "locks"),
                "nifti_manifest": [
                    {"path": "anatomical.nii.gz", "participant_label": "01", "session_label": "baseline", "datatype": "anat", "suffix": "T1w"},
                    {"path": "functional.nii.gz", "participant_label": "02", "session_label": "followup", "datatype": "func", "suffix": "bold", "task": "rest", "metadata": {"RepetitionTime": 2.0}},
                ],
            },
        }).model_dump(mode="json")
        config["project_root"] = str(self.root)

        output = prepare_preprocessing({
            "run_id": "manifest-run",
            "config": config,
            "raw_data": {"location": str(source)},
        })
        context = output["preprocess_context"]

        self.assertTrue(context["bids_preflight"]["passed"])
        self.assertEqual(context["subjects"], ["01", "02"])
        self.assertEqual(context["subject_sessions"], {"01": ["baseline"], "02": ["followup"]})
        bids_dir = Path(context["bids_dir"])
        self.assertTrue((bids_dir / "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz").is_file())
        self.assertTrue((bids_dir / "sub-02/ses-followup/func/sub-02_ses-followup_task-rest_bold.json").is_file())

    def test_same_openneuro_request_preserves_selective_download_config(self) -> None:
        config = AppConfig.model_validate({
            "source": {
                "type": "openneuro",
                "name": "sample",
                "dataset_id": "ds000001",
                "include": ["dataset_description.json", "sub-01"],
                "max_concurrency": 2,
            }
        }).model_dump(mode="json")

        updated = _apply_request_overrides(config, "处理 OpenNeuro ds000001")

        self.assertEqual(updated["source"]["include"], ["dataset_description.json", "sub-01"])
        self.assertEqual(updated["source"]["max_concurrency"], 2)

    def test_openneuro_selective_cache_is_reused_without_redownload(self) -> None:
        dataset = self.root / "raw" / "ds000102"
        (dataset / "sub-01" / "anat").mkdir(parents=True)
        (dataset / "dataset_description.json").write_text('{"Name":"legacy dataset"}', encoding="utf-8")
        (dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"nifti")
        state = {
            "run_id": "cache-test",
            "config": {
                "project_root": str(self.root),
                "runtime": {"mode": "run"},
                "storage": {"raw_dir": "raw"},
            },
            "source": {
                "type": "openneuro", "dataset_id": "ds000102", "tag": None,
                "include": ["dataset_description.json", "sub-01"], "exclude": [],
                "max_concurrency": 2, "reuse_existing": True,
            },
        }
        with (
            patch("neuro_preprocess_agent.tools.source.find_spec", return_value=None),
            patch("neuro_preprocess_agent.tools.source.subprocess.run") as run,
        ):
            result = fetch_data(state)  # type: ignore[arg-type]
        self.assertEqual(result["status"], "reused")
        self.assertEqual(result["items"], 2)
        run.assert_not_called()

        state["source"]["reuse_existing"] = False
        with (
            patch("neuro_preprocess_agent.tools.source.find_spec", return_value=None),
            patch("neuro_preprocess_agent.tools.source.subprocess.run") as run,
            self.assertRaisesRegex(RuntimeError, "openneuro-py not found"),
        ):
            fetch_data(state)  # type: ignore[arg-type]
        run.assert_not_called()

        with (
            patch("neuro_preprocess_agent.tools.source.find_spec", return_value=object()),
            patch("neuro_preprocess_agent.tools.source.subprocess.run") as run,
        ):
            refreshed = fetch_data(state)  # type: ignore[arg-type]
        self.assertEqual(refreshed["status"], "fetched")
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
