# Third-Party Notices

Runtime dependencies and external programs retain their own licenses. This project does not redistribute
fMRIPrep container images, FSL, FreeSurfer, model weights, imaging datasets, or FreeSurfer license files.
Install or obtain them separately under their respective terms.

The ABIDE download script attributed to Daniel Clark (2015) and Cameron Craddock (2019) was moved to
the ignored `local/third_party/` directory pending license verification. It is not part of the public source release.
Source project: https://github.com/preprocessed-connectomes-project/abide

Public dataset descriptions and download instructions are references, not permission to redistribute their
contents. Phenotypic tables, individual images, labels, and generated patient-level records are kept in ignored
local directories. Documentation may contain aggregate validation results and anonymized public case descriptions.

## External preprocessing software

- fMRIPrep 21.0 and later: Apache License 2.0; earlier versions have different terms.
  https://fmriprep.org/en/stable/usage.html#license-information
- FSL: most components are available free for non-commercial use only. Commercial use requires
  separate authorization; some components and atlases have their own terms.
  https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html
- FreeSurfer: subject to the MGH FreeSurfer Software License Agreement and applicable bundled
  third-party terms. Users must obtain their own registration license file.
  https://surfer.nmr.mgh.harvard.edu/registration.html
- dcm2bids: the current upstream LICENSE.txt contains GNU GPL version 3. Check the installed
  release's license before copying, modifying, or redistributing upstream code.
  https://github.com/unfmontreal/Dcm2Bids/blob/master/LICENSE.txt

This repository provides orchestration code for learning and research, not an unrestricted commercial
license for the complete toolchain. Running tools in Docker does not override their licenses.
Users are responsible for checking the exact installed versions, container contents, templates,
and dataset terms. Pin an image version or digest for reproducible deployments.

## Project source

No license grant for this project's original source has been selected yet. Public visibility is for
portfolio inspection and does not itself grant general permission to reuse or redistribute the source.
Third-party licenses remain separate regardless of any future license chosen for original code.
