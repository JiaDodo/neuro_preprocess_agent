from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuro_preprocess_agent.io_utils import atomic_write_text
from neuro_preprocess_agent.tools.visual_qc import (
    ANATOMICAL_INPUT_STATES,
    assess_manifest_with_vlm,
    build_subject_packet,
    embed_manifest,
    fit_normal_reference,
    generate_synthetic_dataset,
    record_visual_review,
    score_vlm_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and manage visual QC datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packets = subparsers.add_parser("packets", help="Generate packets from an fMRIPrep derivatives directory")
    packets.add_argument("--fmriprep-dir", type=Path, required=True)
    packets.add_argument("--bids-dir", type=Path)
    packets.add_argument("--output-dir", type=Path, required=True)
    packets.add_argument("--subjects", nargs="*")

    synthetic = subparsers.add_parser("synthetic", help="Generate traceable synthetic QC failures")
    synthetic.add_argument("--manifest", type=Path, required=True)
    synthetic.add_argument("--output-dir", type=Path, required=True)
    synthetic.add_argument("--seed", type=int, default=42)

    embed = subparsers.add_parser("embed", help="Extract DINOv2 embeddings for a packet manifest")
    embed.add_argument("--manifest", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--model-id", default="facebook/dinov2-small")
    embed.add_argument("--device", default="cuda:0")
    embed.add_argument("--allow-model-download", action="store_true")

    vlm = subparsers.add_parser("vlm-review", help="Run shadow VLM review for a packet manifest")
    vlm.add_argument("--manifest", type=Path, required=True)
    vlm.add_argument("--output", type=Path, required=True)
    vlm.add_argument("--model-id", default="data/models/Qwen2.5-VL-3B-Instruct")
    vlm.add_argument("--device", default="cuda:0")
    vlm.add_argument("--allow-model-download", action="store_true")
    vlm.add_argument("--max-new-tokens", type=int, default=384)
    vlm.add_argument("--min-pixels", type=int, default=200704)
    vlm.add_argument("--max-pixels", type=int, default=401408)
    vlm.add_argument("--max-panels", type=int, default=8)

    score = subparsers.add_parser("score-vlm", help="Score VLM findings against manifest labels")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)

    reference = subparsers.add_parser("fit-reference", help="Fit a normal DINOv2 embedding reference")
    reference.add_argument("--manifest", type=Path, required=True)
    reference.add_argument("--labels-path", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--percentile", type=float, default=99.0)

    label = subparsers.add_parser("label", help="Append a human review to the label ledger")
    label.add_argument("--manifest", type=Path, required=True)
    label.add_argument("--sample-id", required=True)
    label.add_argument("--labels-path", type=Path, required=True)
    label.add_argument("--decision", choices=("pass", "uncertain", "fail"), required=True)
    label.add_argument("--labels", nargs="*", default=[])
    label.add_argument("--annotator", required=True)
    label.add_argument("--note", required=True)
    label.add_argument("--severity", choices=("low", "medium", "high"))
    label.add_argument("--exclude-subject", action="store_true")
    label.add_argument(
        "--anatomical-input-state",
        choices=ANATOMICAL_INPUT_STATES,
        default="unknown",
    )
    args = parser.parse_args()

    if args.command == "packets":
        subjects = args.subjects or sorted(
            path.name.removeprefix("sub-")
            for path in args.fmriprep_dir.glob("sub-*")
            if path.is_dir()
        )
        records = [
            build_subject_packet(subject, args.fmriprep_dir, args.output_dir, args.bids_dir)
            for subject in subjects
        ]
        manifest = args.output_dir / "manifest.jsonl"
        atomic_write_text(manifest, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
        result = {"manifest_path": str(manifest), "subjects": len(records)}
    elif args.command == "synthetic":
        result = generate_synthetic_dataset(args.manifest, args.output_dir, args.seed)
    elif args.command == "embed":
        result = embed_manifest(
            args.manifest,
            args.output,
            args.model_id,
            args.device,
            args.allow_model_download,
        )
    elif args.command == "vlm-review":
        result = assess_manifest_with_vlm(
            args.manifest,
            args.output,
            args.model_id,
            args.device,
            args.allow_model_download,
            args.max_new_tokens,
            args.min_pixels,
            args.max_pixels,
            args.max_panels,
        )
    elif args.command == "score-vlm":
        result = score_vlm_predictions(args.manifest, args.predictions, args.output)
    elif args.command == "fit-reference":
        result = fit_normal_reference(args.manifest, args.labels_path, args.output, args.percentile)
    else:
        records = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sample = next(
            (
                item for item in records
                if str(item.get("sample_id") or item.get("subject")) == args.sample_id
            ),
            None,
        )
        if sample is None:
            raise SystemExit(f"sample not found: {args.sample_id}")
        result = record_visual_review(
            args.labels_path,
            sample,
            args.decision,
            args.labels,
            args.annotator,
            args.note,
            args.severity,
            args.exclude_subject,
            args.anatomical_input_state,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
