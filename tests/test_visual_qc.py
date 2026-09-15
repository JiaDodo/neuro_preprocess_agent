from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image

from neuro_preprocess_agent.tools.visual_qc import (
    LABELS,
    _parse_visual_assessment,
    build_subject_packet,
    fit_normal_reference,
    generate_synthetic_dataset,
    record_visual_review,
    run_visual_qc_shadow,
    score_vlm_predictions,
)


class VisualQCTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.derivatives = self.root / "derivatives"
        self.bids = self.root / "bids"
        anat = self.derivatives / "sub-01" / "anat"
        func = self.derivatives / "sub-01" / "func"
        raw_anat = self.bids / "sub-01" / "anat"
        raw_func = self.bids / "sub-01" / "func"
        anat.mkdir(parents=True)
        func.mkdir(parents=True)
        raw_anat.mkdir(parents=True)
        raw_func.mkdir(parents=True)
        shape = (24, 26, 28)
        grid = np.indices(shape)
        radius = ((grid[0] - 12) / 9) ** 2 + ((grid[1] - 13) / 10) ** 2 + ((grid[2] - 14) / 11) ** 2
        mask = (radius <= 1).astype(np.uint8)
        anatomical = mask * (80 + grid[2] * 4).astype(np.float32)
        functional = mask * (100 + grid[0] * 2).astype(np.float32)
        nib.save(nib.Nifti1Image(anatomical, np.eye(4)), anat / "sub-01_desc-preproc_T1w.nii.gz")
        nib.save(nib.Nifti1Image(mask, np.eye(4)), anat / "sub-01_desc-brain_mask.nii.gz")
        nib.save(nib.Nifti1Image(functional, np.eye(4)), func / "sub-01_task-rest_boldref.nii.gz")
        nib.save(nib.Nifti1Image(mask, np.eye(4)), func / "sub-01_task-rest_desc-brain_mask.nii.gz")
        nib.save(nib.Nifti1Image(anatomical, np.eye(4)), raw_anat / "sub-01_T1w.nii.gz")
        nib.save(
            nib.Nifti1Image(np.repeat(functional[..., None], 3, axis=3), np.eye(4)),
            raw_func / "sub-01_task-rest_bold.nii.gz",
        )
        (func / "sub-01_task-rest_desc-confounds_timeseries.tsv").write_text(
            "framewise_displacement\nna\n0.1\n0.2\n0.3\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_packet_and_synthetic_manifest_are_generated(self) -> None:
        packet_dir = self.root / "packets"
        record = build_subject_packet("01", self.derivatives, packet_dir, self.bids)
        with Image.open(record["packet_path"]) as packet:
            self.assertEqual(packet.width, 672)
            self.assertGreater(packet.height, 600)
        self.assertEqual(len(record["panel_paths"]), 3)
        self.assertEqual(len(record["raw_panel_paths"]), 2)
        self.assertEqual(record["image_metadata"]["raw_bold"]["volumes"], 3)

        manifest = packet_dir / "manifest.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = generate_synthetic_dataset(manifest, self.root / "synthetic", seed=7)
        generated = [json.loads(line) for line in Path(result["manifest_path"]).read_text().splitlines()]
        self.assertEqual(result["samples"], 5)
        self.assertEqual({item["synthetic_transform"] for item in generated}, {
            "brain_mask_error", "coverage_error", "ghosting_dropout", "coregistration_error", "severe_motion",
        })
        self.assertTrue(all(set(item["labels"]) == set(LABELS) for item in generated))

    def test_shadow_packet_generation_never_changes_gate(self) -> None:
        state = {
            "run_id": "visual-test",
            "config": {
                "project_root": str(self.root),
                "runtime": {"mode": "run"},
                "qc": {"visual": {
                    "enabled": True,
                    "packet_dir": "visual",
                    "model_backend": "none",
                    "model_id": "facebook/dinov2-small",
                }},
            },
            "processed_data": {"fmriprep_dir": str(self.derivatives), "subjects": ["01"]},
        }
        result = run_visual_qc_shadow(state)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["affects_gate"])
        self.assertEqual(result["subjects"][0]["status"], "packet_only")

    def test_vlm_json_is_strict_and_shadow_result_never_changes_gate(self) -> None:
        assessment = _parse_visual_assessment(
            '```json\n{"decision":"review","confidence":0.8,"observations":['
            '{"category":"coverage_error","abnormal":true,"severity":"medium",'
            '"evidence":"Functional panel is truncated"}'
            '],"summary":"Human review required"}\n```'
        )
        self.assertEqual(assessment.decision, "review")
        state = {
            "run_id": "vlm-test",
            "config": {
                "project_root": str(self.root),
                "runtime": {"mode": "run"},
                "qc": {"visual": {
                    "enabled": True,
                    "packet_dir": "visual",
                    "model_backend": "qwen2_5_vl",
                    "model_id": "test-model",
                }},
            },
            "processed_data": {"fmriprep_dir": str(self.derivatives), "subjects": ["01"]},
        }
        predicted = {
            "decision": "review",
            "confidence": 0.8,
            "findings": [{
                "category": "coverage_error", "severity": "medium", "evidence": "visible truncation",
            }],
            "summary": "Review required",
            "raw_response": "{}",
            "panels_scored": ["functional_mask.png"],
            "prompt_version": "fmriprep-visual-qc-v2",
        }
        with patch("neuro_preprocess_agent.tools.visual_qc._qwen_vl_assessment", return_value=predicted):
            result = run_visual_qc_shadow(state)
        self.assertEqual(result["subjects"][0]["decision"], "review")
        self.assertEqual(result["subjects"][0]["status"], "completed")
        self.assertFalse(result["affects_gate"])

    def test_duplicate_vlm_observations_keep_the_worst_result(self) -> None:
        assessment = _parse_visual_assessment(
            '{"decision":"review","confidence":0.9,"observations":['
            '{"category":"severe_motion","abnormal":false,"severity":"low",'
            '"evidence":"motion is acceptable"},'
            '{"category":"severe_motion","abnormal":true,"severity":"medium",'
            '"evidence":"motion is uncertain"}'
            '],"summary":"Review motion"}'
        )
        self.assertEqual(len(assessment.observations), 1)
        self.assertTrue(assessment.observations[0].abnormal)

    def test_review_ledger_controls_normal_reference(self) -> None:
        embedding_a = self.root / "a.npy"
        embedding_b = self.root / "b.npy"
        np.save(embedding_a, np.array([1.0, 0.0], dtype=np.float32))
        np.save(embedding_b, np.array([0.99, 0.01], dtype=np.float32))
        manifest = self.root / "manifest.jsonl"
        records = [
            {"subject": "01", "embedding_path": str(embedding_a), "embedding_model": "test-model"},
            {"subject": "02", "embedding_path": str(embedding_b), "embedding_model": "test-model"},
        ]
        manifest.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        labels = self.root / "labels.jsonl"
        for record in records:
            record_visual_review(labels, record, "pass", [], "tester", "checked")
        result = fit_normal_reference(manifest, labels, self.root / "reference.npz")
        self.assertEqual(result["sample_count"], 2)
        self.assertTrue(Path(result["reference_path"]).is_file())

    def test_review_ledger_validates_and_preserves_audit_fields(self) -> None:
        labels = self.root / "labels.jsonl"
        sample = {"sample_id": "case-01", "subject": "01", "packet_path": "packet.png"}
        with self.assertRaisesRegex(ValueError, "require at least one"):
            record_visual_review(labels, sample, "fail", [], "tester", "visible artifact")
        with self.assertRaisesRegex(ValueError, "require at least one"):
            record_visual_review(labels, sample, "uncertain", [], "tester", "unclear image")
        with self.assertRaisesRegex(ValueError, "require a note"):
            record_visual_review(labels, sample, "uncertain", ["severe_motion"], "tester", "")
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            record_visual_review(labels, sample, "pass", ["severe_motion"], "tester", "checked")
        with self.assertRaisesRegex(ValueError, "anatomical_input_state"):
            record_visual_review(
                labels,
                sample,
                "pass",
                [],
                "tester",
                "checked",
                anatomical_input_state="invalid",
            )

        result = record_visual_review(
            labels,
            sample,
            "fail",
            ["severe_motion"],
            " tester ",
            " excessive displacement ",
            severity="high",
            exclude_subject=True,
            anatomical_input_state="defaced",
        )
        self.assertEqual(result["schema_version"], "1.2")
        self.assertEqual(result["annotator"], "tester")
        self.assertEqual(result["note"], "excessive displacement")
        self.assertEqual(result["severity"], "high")
        self.assertTrue(result["exclude_subject"])
        self.assertEqual(result["anatomical_input_state"], "defaced")

    def test_vlm_scoring_reports_false_positives_and_false_negatives(self) -> None:
        manifest = self.root / "labeled.jsonl"
        predictions = self.root / "predictions.jsonl"
        manifest.write_text(
            json.dumps({
                "sample_id": "one",
                "labels": {"coverage_error": True, "severe_motion": False},
            }) + "\n",
            encoding="utf-8",
        )
        predictions.write_text(
            json.dumps({
                "sample_id": "one",
                "status": "completed",
                "findings": [
                    {"category": "severe_motion", "severity": "medium", "evidence": "suspected"},
                ],
            }) + "\n",
            encoding="utf-8",
        )
        result = score_vlm_predictions(manifest, predictions)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["micro_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
