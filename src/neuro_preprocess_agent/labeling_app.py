from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from neuro_preprocess_agent.config import PROJECT_ROOT
from neuro_preprocess_agent.tools.visual_qc import (
    ANATOMICAL_INPUT_STATES,
    LABELS,
    record_visual_review,
)

LABEL_TEXT = {
    "brain_mask_error": "脑掩膜错误",
    "normalization_error": "标准化错误",
    "coregistration_error": "功能-结构配准错误",
    "coverage_error": "视野或脑区覆盖不完整",
    "ghosting_dropout": "重影或信号缺失",
    "segmentation_error": "组织分割错误",
    "severe_motion": "严重头动",
}

PANEL_TEXT = {
    "anatomical_mask": "预处理 T1w 与脑掩膜",
    "functional_mask": "预处理 BOLD reference 与脑掩膜",
    "motion": "逐帧头动",
    "segmentation": "组织分割",
    "normalization": "标准空间配准",
    "coregistration": "功能-结构配准",
    "functional_rois": "功能像覆盖与 ROI",
    "carpetplot": "Carpet plot",
}

INPUT_STATE_TEXT = {
    "unknown": "无法判断",
    "full_head": "完整头部",
    "defaced": "面部已脱敏",
    "pre_skull_stripped": "输入已去颅骨",
}


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _latest_file(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _best_prediction_file(manifest_path: Path | None) -> Path | None:
    if manifest_path is None or not manifest_path.is_file():
        return _latest_file("data/visual_qc/vlm_reviews/*.jsonl")
    sample_ids = {
        str(item.get("sample_id") or item.get("subject"))
        for item in _read_jsonl(manifest_path)
    }
    candidates = list(PROJECT_ROOT.glob("data/visual_qc/vlm_reviews/*.jsonl"))
    scored: list[tuple[int, float, Path]] = []
    for path in candidates:
        try:
            prediction_ids = {
                str(item.get("sample_id") or item.get("subject"))
                for item in _read_jsonl(path)
                if item.get("status") in {None, "completed", "unavailable"}
            }
        except (OSError, ValueError):
            continue
        scored.append((len(sample_ids & prediction_ids), path.stat().st_mtime, path))
    matches = [item for item in scored if item[0] > 0]
    return max(matches, default=(0, 0.0, None))[2]


def _display_name(sample: dict[str, Any], index: int) -> str:
    sample_id = str(sample.get("sample_id") or sample.get("subject") or index)
    if sample.get("synthetic"):
        token = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8].upper()
        return f"盲标样本 {token}"
    return sample_id if sample.get("sample_id") else f"sub-{sample_id}"


def _subject_metrics(report_path: Path | None) -> dict[str, dict[str, Any]]:
    if report_path is None or not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        str(item.get("subject")): item
        for item in report.get("qc_result", {}).get("subject_qc", [])
    }


def main() -> None:
    import streamlit as st

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels-path", type=Path, default=PROJECT_ROOT / "data/visual_qc/labels.jsonl")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--report", type=Path)
    args, _ = parser.parse_known_args()

    default_manifest = args.manifest or _latest_file("data/visual_qc/**/manifest.jsonl")
    default_predictions = args.predictions or _best_prediction_file(default_manifest)
    default_report = args.report or _latest_file("runs/reports/*_summary.json")

    st.set_page_config(page_title="影像质量标注", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1500px; padding-top: 1.5rem;}
        h1 {font-size: 1.65rem !important; letter-spacing: 0 !important;}
        h2, h3 {letter-spacing: 0 !important;}
        [data-testid="stImage"] img {border: 1px solid #d8dde3; background: white;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("标注任务")
        manifest_text = st.text_input("Manifest", str(default_manifest or ""))
        labels_text = st.text_input("标签账本", str(args.labels_path))
        predictions_text = st.text_input("VLM 结果", str(default_predictions or ""))
        report_text = st.text_input("运行报告", str(default_report or ""))
        annotator = st.text_input("标注者", value="dodo")
        status_filter = st.radio("样本范围", ("未标注", "全部", "已标注"), horizontal=True)
        reveal_vlm = st.checkbox("显示 VLM 建议", value=False)

    manifest_path = Path(manifest_text).expanduser() if manifest_text else None
    labels_path = Path(labels_text).expanduser()
    predictions_path = Path(predictions_text).expanduser() if predictions_text else None
    report_path = Path(report_text).expanduser() if report_text else None
    try:
        samples = _read_jsonl(manifest_path)
        review_history = _read_jsonl(labels_path)
        prediction_records = _read_jsonl(predictions_path)
        metrics_by_subject = _subject_metrics(report_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()
    if not samples:
        st.warning("没有可标注样本。")
        st.stop()

    annotator_reviews = [
        item for item in review_history
        if str(item.get("annotator", "")).strip() == annotator.strip()
    ]
    latest_reviews = {str(item.get("sample_id")): item for item in annotator_reviews}
    predictions = {
        str(item.get("sample_id") or item.get("subject")): item
        for item in prediction_records
    }
    if status_filter == "未标注":
        visible = [item for item in samples if str(item.get("sample_id") or item.get("subject")) not in latest_reviews]
    elif status_filter == "已标注":
        visible = [item for item in samples if str(item.get("sample_id") or item.get("subject")) in latest_reviews]
    else:
        visible = samples

    reviewed_count = sum(
        str(item.get("sample_id") or item.get("subject")) in latest_reviews
        for item in samples
    )
    st.title("影像质量标注")
    progress_columns = st.columns(4)
    progress_columns[0].metric("样本", len(samples))
    progress_columns[1].metric("已标注", reviewed_count)
    progress_columns[2].metric("未标注", len(samples) - reviewed_count)
    progress_columns[3].metric("进度", f"{reviewed_count / len(samples):.0%}")
    st.progress(reviewed_count / len(samples))
    if not visible:
        st.success("当前范围内没有待处理样本。")
        st.stop()

    index_key = f"sample_index_{status_filter}"
    st.session_state[index_key] = min(st.session_state.get(index_key, 0), len(visible) - 1)
    navigation = st.columns((1, 4, 1))
    if navigation[0].button("上一例", disabled=st.session_state[index_key] == 0, width="stretch"):
        st.session_state[index_key] -= 1
        st.rerun()
    selected_index = navigation[1].selectbox(
        "当前样本",
        range(len(visible)),
        index=st.session_state[index_key],
        format_func=lambda value: _display_name(visible[value], value),
        label_visibility="collapsed",
    )
    if selected_index != st.session_state[index_key]:
        st.session_state[index_key] = selected_index
        st.rerun()
    if navigation[2].button(
        "下一例",
        disabled=st.session_state[index_key] >= len(visible) - 1,
        width="stretch",
    ):
        st.session_state[index_key] += 1
        st.rerun()

    sample = visible[st.session_state[index_key]]
    sample_id = str(sample.get("sample_id") or sample.get("subject"))
    subject = str(sample.get("subject") or sample_id)
    existing = latest_reviews.get(sample_id)
    st.subheader(_display_name(sample, st.session_state[index_key]))
    if existing:
        st.caption(f"最近标注：{existing.get('decision')} · {existing.get('reviewed_at')}")

    if not sample.get("synthetic"):
        raw_t1w = sample.get("image_metadata", {}).get("raw_t1w", {})
        raw_nonzero_fraction = raw_t1w.get("nonzero_fraction")
        if raw_nonzero_fraction is not None and float(raw_nonzero_fraction) < 0.3:
            st.warning(
                f"原始 T1w 非零体素占比仅 {float(raw_nonzero_fraction):.1%}，"
                "输入可能已经去颅骨或被明显裁剪，请重点检查脑提取结果。"
            )

        current_images = metrics_by_subject.get(subject, {}).get("quantitative_metrics", {}).get("images", {})
        current_mask_volume = current_images.get("t1w_brain_mask", {}).get("mask_volume_mm3")
        batch_mask_volumes = [
            item.get("quantitative_metrics", {}).get("images", {}).get("t1w_brain_mask", {}).get("mask_volume_mm3")
            for item in metrics_by_subject.values()
        ]
        batch_mask_volumes = [float(value) for value in batch_mask_volumes if value is not None]
        if current_mask_volume is not None and batch_mask_volumes:
            batch_median = median(batch_mask_volumes)
            if float(current_mask_volume) < batch_median * 0.6:
                st.warning(
                    f"T1w 脑掩膜体积为 {float(current_mask_volume) / 1000:.1f} mL，"
                    f"低于本批次中位数 {batch_median / 1000:.1f} mL 的 60%，请检查是否过度脑提取。"
                )

    raw_image_paths = [Path(path) for path in sample.get("raw_panel_paths", []) if Path(path).is_file()]
    if raw_image_paths:
        st.markdown("#### 原始输入")
        raw_columns = st.columns(2)
        for index, image_path in enumerate(raw_image_paths):
            caption = "原始 T1w" if "anatomical" in image_path.stem else "原始 BOLD 时间均值"
            raw_columns[index % 2].image(str(image_path), caption=caption, width="stretch")
    else:
        st.info("该历史 manifest 未包含原始数据面板；重新生成 packet 后可显示。")

    st.markdown("#### 预处理结果与质控证据")
    image_paths = [Path(path) for path in sample.get("panel_paths", []) if Path(path).is_file()]
    if not image_paths and sample.get("packet_path") and Path(sample["packet_path"]).is_file():
        image_paths = [Path(sample["packet_path"])]
    image_columns = st.columns(2)
    for index, image_path in enumerate(image_paths):
        caption = PANEL_TEXT.get(image_path.stem, f"质控面板 {index + 1}")
        image_columns[index % 2].image(str(image_path), caption=caption, width="stretch")

    subject_qc = metrics_by_subject.get(subject, {})
    metrics = subject_qc.get("quantitative_metrics", {})
    if metrics:
        images = metrics.get("images", {})
        metric_rows = []
        bold_tsnr = images.get("bold_tsnr", {}).get("median")
        dice = images.get("anat_bold_mask_overlap", {}).get("dice")
        bold_mask_fraction = images.get("bold_brain_mask", {}).get("mask_fraction")
        t1_mask_volume = images.get("t1w_brain_mask", {}).get("mask_volume_mm3")
        for name, value in (
            ("BOLD median tSNR", bold_tsnr),
            ("Anat/BOLD mask Dice", dice),
            ("BOLD mask fraction", bold_mask_fraction),
            ("T1w brain-mask volume (mL)", float(t1_mask_volume) / 1000 if t1_mask_volume else None),
        ):
            if value is not None:
                metric_rows.append({"指标": name, "数值": round(float(value), 4)})
        motion_check = next(
            (item for item in subject_qc.get("checks", []) if str(item.get("name", "")).endswith(":motion_review_thresholds")),
            None,
        )
        if motion_check:
            observed = motion_check.get("observed", {})
            for name, key in (
                ("Mean FD (mm)", "mean_fd_mm"),
                ("FD outlier fraction", "fd_outlier_fraction"),
                ("Mean standardized DVARS", "mean_std_dvars"),
            ):
                if observed.get(key) is not None:
                    metric_rows.append({"指标": name, "数值": round(float(observed[key]), 4)})
        if metric_rows:
            st.dataframe(metric_rows, hide_index=True, width="stretch")

    if not sample.get("synthetic"):
        with st.expander("影像文件与采集信息"):
            metadata_rows = []
            for name, metadata in sample.get("image_metadata", {}).items():
                metadata_rows.append({
                    "影像": name,
                    "维度": " x ".join(str(value) for value in metadata.get("shape", [])),
                    "体素大小 (mm)": " x ".join(str(value) for value in metadata.get("voxel_size_mm", [])),
                    "时间点": metadata.get("volumes"),
                    "非零占比": metadata.get("nonzero_fraction"),
                    "路径": metadata.get("path"),
                })
            if metadata_rows:
                st.dataframe(metadata_rows, hide_index=True, width="stretch")
            else:
                st.write(sample.get("sources", {}))

    prediction = predictions.get(sample_id) or predictions.get(subject)
    if reveal_vlm and prediction:
        with st.expander("VLM shadow 建议", expanded=True):
            st.write({
                "status": prediction.get("status"),
                "decision": prediction.get("decision"),
                "confidence": prediction.get("confidence"),
                "summary": prediction.get("summary"),
                "findings": prediction.get("findings", []),
            })

    prior_labels = [name for name, selected in (existing or {}).get("labels", {}).items() if selected]
    decision_options = ("pass", "uncertain", "fail")
    default_decision = existing.get("decision", "pass") if existing else "pass"
    if default_decision == "review":
        default_decision = "uncertain"
    with st.form(f"review_{sample_id}"):
        default_input_state = (existing or {}).get("anatomical_input_state", "unknown")
        if default_input_state not in ANATOMICAL_INPUT_STATES:
            default_input_state = "unknown"
        anatomical_input_state = st.selectbox(
            "原始结构像状态",
            ANATOMICAL_INPUT_STATES,
            index=ANATOMICAL_INPUT_STATES.index(default_input_state),
            format_func=lambda value: INPUT_STATE_TEXT[value],
            help="这是输入数据特征，不属于预处理异常标签。",
        )
        decision = st.radio(
            "总体结论",
            decision_options,
            index=decision_options.index(default_decision),
            horizontal=True,
        )
        selected_labels = st.multiselect(
            "异常标签",
            LABELS,
            default=prior_labels,
            format_func=lambda value: LABEL_TEXT[value],
        )
        severity_options = ("low", "medium", "high")
        default_severity = (existing or {}).get("severity") or "medium"
        severity = st.select_slider(
            "异常严重程度",
            options=severity_options,
            value=default_severity,
            disabled=not selected_labels,
        )
        exclude_subject = st.checkbox(
            "建议排除该被试",
            value=bool((existing or {}).get("exclude_subject", False)),
        )
        note = st.text_area("审核备注", value=(existing or {}).get("note", ""), height=100)
        submitted = st.form_submit_button("保存并进入下一例", type="primary", width="stretch")
    if submitted:
        if not annotator.strip():
            st.error("标注者不能为空。")
        elif decision != "pass" and not note.strip():
            st.error("不确定或不通过时必须填写审核备注。")
        else:
            try:
                record_visual_review(
                    labels_path,
                    sample,
                    decision,
                    list(selected_labels),
                    annotator.strip(),
                    note.strip(),
                    severity if selected_labels else None,
                    exclude_subject,
                    anatomical_input_state,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                if status_filter != "未标注":
                    st.session_state[index_key] = min(st.session_state[index_key] + 1, len(visible) - 1)
                st.rerun()

    sample_history = [item for item in review_history if str(item.get("sample_id")) == sample_id]
    if sample_history:
        with st.expander(f"历史记录（{len(sample_history)}）"):
            st.dataframe(sample_history, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
