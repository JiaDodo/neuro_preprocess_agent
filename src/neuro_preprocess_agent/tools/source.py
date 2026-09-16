from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from neuro_preprocess_agent.config import RuntimeMode, project_path
from neuro_preprocess_agent.state import PipelineState


def count_files(path: Path) -> int:
    if path.is_file():
        return 1
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def resolve_source(state: PipelineState) -> dict[str, Any]:
    source_config = state["config"]["source"]
    source_type = source_config["type"]
    source = {
        "name": source_config.get("name") or source_config.get("dataset_id") or "dataset",
        "type": source_type,
        "dataset_id": source_config.get("dataset_id"),
        "url": source_config.get("url"),
        "path": source_config.get("path"),
        "uri": source_config.get("uri"),
        "tag": source_config.get("tag"),
        "include": source_config.get("include", []),
        "exclude": source_config.get("exclude", []),
        "max_concurrency": source_config.get("max_concurrency", 5),
        "reuse_existing": source_config.get("reuse_existing", True),
    }
    if source_type == "local_path":
        path = project_path(source["path"], state["config"].get("project_root"))
        if not path.exists():
            raise FileNotFoundError(f"Local source path not found: {path}")
        source["path"] = str(path.resolve())
        source["name"] = source_config.get("name") or path.name
    return source


def fetch_data(state: PipelineState) -> dict[str, Any]:
    config = state["config"]
    source = state["source"]
    mode: RuntimeMode = config["runtime"]["mode"]
    raw_dir = project_path(config["storage"]["raw_dir"], config.get("project_root"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    if mode == "mock":
        mock_location = raw_dir / state["run_id"] / "mock_dataset"
        return {
            "status": "mocked",
            "mode": mode,
            "source_type": source["type"],
            "location": str(mock_location),
            "items": config.get("mock", {}).get("raw_items", 3),
        }

    if source["type"] == "openneuro":
        dataset_id = source["dataset_id"]
        output_dir = raw_dir / dataset_id
        installed = find_spec("openneuro") is not None
        command = [sys.executable, "-m", "openneuro", "download", "--dataset", dataset_id, "--target-dir", str(output_dir)]
        if source.get("tag"):
            command.extend(["--tag", source["tag"]])
        for value in source.get("include", []):
            command.extend(["--include", value])
        for value in source.get("exclude", []):
            command.extend(["--exclude", value])
        command.extend(["--max-concurrent-downloads", str(source.get("max_concurrency", 5))])
        includes = source.get("include", [])
        cache_complete = bool(includes) and all(
            (candidate := output_dir / value).is_file()
            or (candidate.is_dir() and any(item.is_file() for item in candidate.rglob("*")))
            for value in includes
        )
        reused = mode == "run" and source.get("reuse_existing", True) and cache_complete
        if mode == "run":
            if not installed:
                raise RuntimeError("openneuro-py not found. Install it in the active project environment.")
            if not reused:
                subprocess.run(command, check=True)
        return {
            "status": "reused" if reused else ("fetched" if mode == "run" else "planned"),
            "mode": mode,
            "source_type": "openneuro",
            "dataset_id": dataset_id,
            "location": str(output_dir),
            "command": command,
            "items": count_files(output_dir) if mode == "run" else 0,
        }

    if source["type"] == "url_file":
        url = source["url"]
        filename = Path(str(url).split("?", 1)[0]).name or "downloaded_file"
        output_path = raw_dir / state["run_id"] / filename
        if mode == "run":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, output_path)
        return {
            "status": "fetched" if mode == "run" else "planned",
            "mode": mode,
            "source_type": "url_file",
            "url": url,
            "location": str(output_path),
            "items": 1 if mode == "run" else 0,
        }

    input_path = Path(source["path"])
    copy_local = bool(config["storage"].get("copy_local_data", True))
    output_path = raw_dir / state["run_id"] / input_path.name if copy_local else input_path
    if mode == "run" and copy_local:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if input_path.is_dir():
            shutil.copytree(input_path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(input_path, output_path)
    effective_location = input_path if mode == "dry_run" else output_path
    return {
        "status": "fetched" if mode == "run" else "planned",
        "mode": mode,
        "source_type": "local_path",
        "input_path": str(input_path),
        "location": str(effective_location),
        "planned_location": str(output_path),
        "copied": copy_local and mode == "run",
        "items": count_files(effective_location),
    }
