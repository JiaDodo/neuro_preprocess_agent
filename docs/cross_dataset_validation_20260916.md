# OpenNeuro cross-dataset validation

Date: 2026-09-16

## Purpose

The earlier 10-subject run used only `ds000001`. This pilot tests whether the workflow handles
different BIDS layouts and acquisition designs instead of treating more subjects from one source
as evidence of cross-dataset compatibility.

## Pilot matrix

| Dataset | Pilot subject | Distinguishing condition | Selected workload | BIDS Validator | Pipeline outcome |
| --- | --- | --- | --- | --- | --- |
| `ds000114` v1.0.2 | `sub-01` | test/retest sessions, two T1w images | line-bisection task in both sessions | passed after restoring inherited `dwi.bval`/`dwi.bvec` | fMRIPrep completed; multi-session QC passed; MySQL written |
| `ds000030` v1.0.0 | `sub-10159` | psychiatric cohort, several tasks | resting-state only | failed: legacy `CogAtlasID`/`CogPOID` values are not URIs | blocked before fMRIPrep and database |
| `ds000102` | `sub-01` | one session, two functional runs | both flanker runs | failed: legacy `CogAtlasID` value is not a URI | blocked before fMRIPrep and database |
| `ds000228` v1.1.1 | `sub-pixar001` | pediatric cohort, movie fMRI | one movie run | passed | fMRIPrep completed; QC passed; MySQL written |

The selective downloads contain 75 files and approximately 710 MB. The structural PyBIDS check
found T1w and BOLD data for every selected subject. The containerized BIDS Validator is treated as
the authoritative compatibility gate before fMRIPrep.

Two of four datasets passed the current BIDS schema. Both compatible cases completed fMRIPrep,
passed the postprocessing gate, and were written to MySQL (2/2). Both incompatible cases were
stopped before fMRIPrep and produced failure reports without database records (0 false passes).
The multi-session run took 2,339 seconds and the pediatric run took 1,882 seconds on the tested
server with a 16-process, 32-GB per-subject budget.

## Defects exposed by the pilot

1. A failed preprocessing-preparation node still entered input QC and produced a secondary missing-
   state error. The graph now routes preprocessing failures back to the supervisor and report.
2. `openneuro-py` could not revisit a successfully downloaded legacy dataset whose
   `dataset_description.json` lacked `DatasetDOI`. Selective caches are now reused only when every
   requested file or non-empty directory is present.
3. The previous BIDS preflight used PyBIDS for structure and inherited metadata but did not apply the
   complete BIDS schema. A containerized `bids-validator --json` check now runs before fMRIPrep and
   stores structured errors and warnings in state.
4. The first `ds000114` include list copied DWI images but omitted inherited top-level gradient files.
   Adding `dwi.bval` and `dwi.bvec` changed the official validator result from four errors to zero.
5. The postprocessing QC looked only under `sub-*/func` and therefore missed valid outputs under
   `sub-*/ses-*/func`. Output discovery now checks every expected session and every selected run;
   the already completed multi-session result then passed 32 checks and was written to MySQL.

Each defect has a regression test. The two legacy metadata failures are retained as negative cases;
the workflow must report them and prevent database writes rather than silently adding
`--skip-bids-validation`.

## Expansion rule

Do not immediately scale every dataset. First complete one subject for each validator-compatible
layout. Then expand successful datasets to 5 subjects and add another genuinely different dataset.
Report dataset-level pass rate, subject-level preprocessing success, QC rejection/review rate,
database-blocking false passes, runtime, and recovery/reuse behavior. Visual QC labels are requested
only for completed outputs that enter manual review.
