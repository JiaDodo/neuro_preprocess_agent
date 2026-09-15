from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuro_preprocess_agent.labeling_app import (
    _best_prediction_file,
    _display_name,
    _read_jsonl,
)


class LabelingAppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_synthetic_display_name_does_not_reveal_label(self) -> None:
        sample = {
            "sample_id": "sub-01_brain_mask_error",
            "subject": "01",
            "synthetic": True,
            "synthetic_transform": "brain_mask_error",
        }
        name = _display_name(sample, 0)
        self.assertTrue(name.startswith("盲标样本 "))
        self.assertNotIn("brain_mask_error", name)
        self.assertNotIn("sub-01", name)

    def test_read_jsonl_reports_bad_line(self) -> None:
        path = self.root / "bad.jsonl"
        path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, ":2"):
            _read_jsonl(path)

    def test_prediction_discovery_prefers_matching_samples(self) -> None:
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps({"sample_id": "case-a"}) + "\n", encoding="utf-8")
        reviews = self.root / "data/visual_qc/vlm_reviews"
        reviews.mkdir(parents=True)
        (reviews / "wrong.jsonl").write_text(json.dumps({"sample_id": "case-b"}) + "\n", encoding="utf-8")
        expected = reviews / "right.jsonl"
        expected.write_text(json.dumps({"sample_id": "case-a", "status": "completed"}) + "\n", encoding="utf-8")
        with patch("neuro_preprocess_agent.labeling_app.PROJECT_ROOT", self.root):
            self.assertEqual(_best_prediction_file(manifest), expected)


if __name__ == "__main__":
    unittest.main()
