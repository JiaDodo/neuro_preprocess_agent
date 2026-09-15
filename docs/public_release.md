# Public Release Boundary

The public repository contains source code, synthetic test fixtures, example configuration,
and aggregate validation summaries. It must not contain individual imaging files, phenotypic
tables, downloaded model weights, database files, credentials, or patient-level runtime records.

## Local files

- `data/`, `runs/`, and `example_data/` remain ignored.
- The phenotypic CSV was relocated to `data/qc_labels/` without deleting the local copy.
- Original server configuration is preserved in `local/configs_original/`.
- Local DICOM case configuration is in `local/configs/`.
- Personal environment scripts and dependency snapshots are in `local/scripts/` and `local/env/`.
- Server-specific real-data regression suites are in `local/evals/` and are not redistributed.
- The unverified third-party ABIDE script is quarantined in `local/third_party/`.

For existing server runs, use an original local configuration if necessary. Public configuration
uses PATH discovery for dcm2bids and `data/private/license.txt` or `FS_LICENSE` for the license.
The DICOM example uses a placeholder SeriesDescription and requires scanner-specific validation.

## Pre-commit checks

```bash
python scripts/check_public_release.py
python -m unittest discover -s tests -v
git status --short
```

The release scanner checks tracked and untracked non-ignored candidates, rejects data/model/database
extensions and private directories, and reports secret-pattern locations without revealing values.
Never force-add ignored files. Do not publish a ZIP of the entire working directory.
Git ignore rules do not remove already tracked data; the scanner also checks tracked files.

License selection for the project's source remains an owner decision. Third-party and dataset terms
are separate; see `THIRD_PARTY_NOTICES.md`. No claim of comprehensive legal or privacy review is made.
