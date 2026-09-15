from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuro_preprocess_agent.config import load_config


class ConfigPortabilityTestCase(unittest.TestCase):
    def test_placeholder_paths_use_environment_and_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({"source": {"type": "openneuro", "dataset_id": "ds000001"}}))
            with patch.dict(os.environ, {"FS_LICENSE": "/custom/license.txt"}), patch(
                "neuro_preprocess_agent.config.find_executable", return_value="/custom/bin/dcm2bids",
            ):
                config = load_config(config_path)
            self.assertEqual(config["preprocess"]["dcm2bids"], "/custom/bin/dcm2bids")
            self.assertEqual(config["preprocess"]["freesurfer_license"], "/custom/license.txt")

    def test_explicit_paths_are_not_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({
                "source": {"type": "openneuro", "dataset_id": "ds000001"},
                "preprocess": {"dcm2bids": "/explicit/dcm2bids", "freesurfer_license": "/explicit/license.txt"},
            }))
            with patch.dict(os.environ, {"FS_LICENSE": "/other/license.txt"}):
                config = load_config(config_path)
            self.assertEqual(config["preprocess"]["dcm2bids"], "/explicit/dcm2bids")
            self.assertEqual(config["preprocess"]["freesurfer_license"], "/explicit/license.txt")
