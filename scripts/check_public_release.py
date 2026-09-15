from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRECTORIES = {"data", "runs", "local", "example_data", ".venv", ".venv-server"}
PRIVATE_SUFFIXES = (
    ".csv", ".dcm", ".dicom", ".nii", ".nii.gz", ".safetensors", ".pt", ".pth",
    ".ckpt", ".sqlite", ".sqlite3", ".db", ".npy", ".npz", ".joblib",
)
SECRET_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})\b"
    r"|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
)


def main() -> None:
    # Include tracked files even if an ignore rule was added after staging them.
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    )
    names = sorted(set(filter(None, result.stdout.decode().split("\0"))))
    problems: list[str] = []
    for name in names:
        path = Path(name)
        is_private = (
            path.parts[0] in PRIVATE_DIRECTORIES
            or (path.name.startswith(".env") and path.name != ".env.example")
            or path.name.endswith(".local.json")
            or path.name == "license.txt"
            or name.lower().endswith(PRIVATE_SUFFIXES)
        )
        if is_private:
            problems.append(f"Private data/config candidate: {name}")
            continue
        file_path = ROOT / path
        if file_path.is_symlink():
            problems.append(f"Symlink requires manual review: {name}")
            continue
        if not file_path.is_file():
            continue
        if file_path.stat().st_size > 10 * 1024 * 1024:
            problems.append(f"Large file requires manual review: {name}")
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeError:
            problems.append(f"Binary file requires manual review: {name}")
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            if SECRET_PATTERN.search(line):
                problems.append(f"Possible credential: {name}:{line_number}")
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print(f"PASS: {len(names)} public candidate files; no blocked data files or obvious credential patterns.")
    print("Pattern scanning does not replace manual provenance and privacy review.")
    if not (ROOT / "LICENSE").is_file():
        print("WARNING: project owner must select a source license before open-source release.")


if __name__ == "__main__":
    main()
