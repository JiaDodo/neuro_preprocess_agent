from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from neuro_preprocess_agent.config import extract_absolute_path


def clean_terminal_text(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value).strip()


def parse_human_response(value: str) -> bool | dict[str, Any] | None:
    text = clean_terminal_text(value)
    lowered = text.lower()
    if text in {"继续", "同意", "是", "确认", "执行", "可以", "开始"} or lowered in {"yes", "y", "true", "ok", "go"}:
        return True
    if text in {"停止", "不同意", "否", "取消", "终止", "不执行"} or lowered in {"no", "n", "false", "stop", "cancel"}:
        return False

    update: dict[str, Any] = {}
    preprocess: dict[str, Any] = {}
    requested_path = extract_absolute_path(text)
    dataset_match = re.search(r"\b(ds\d{6})\b", text, flags=re.IGNORECASE)
    url_match = re.search(r"https?://[^\s，。；,;]+", text)
    if requested_path and not url_match:
        source_path = Path(requested_path).expanduser()
        update["source"] = {"name": source_path.name or "local_data", "type": "local_path", "path": str(source_path)}
    elif dataset_match:
        dataset_id = dataset_match.group(1).lower()
        update["source"] = {"name": dataset_id, "type": "openneuro", "dataset_id": dataset_id, "uri": "https://openneuro.org"}
    elif url_match:
        update["source"] = {"name": "URL dataset", "type": "url_file", "url": url_match.group(0)}

    workers = re.search(r"(?:并行|并发|worker|max_workers)[^\d]*(\d+)", text, flags=re.IGNORECASE)
    if workers:
        preprocess["max_workers"] = int(workers.group(1))
    derivatives = re.search(r"(?:输出目录|derivatives_dir|fmriprep目录)[：:=\s]*([^\s，。；,;]+)", text, flags=re.IGNORECASE)
    if derivatives:
        preprocess["derivatives_dir"] = derivatives.group(1)
    bids = re.search(r"(?:BIDS目录|bids_dir|bids目录)[：:=\s]*([^\s，。；,;]+)", text, flags=re.IGNORECASE)
    if bids:
        preprocess["bids_dir"] = bids.group(1)
    if preprocess:
        update["preprocess"] = preprocess
    if any(keyword in text for keyword in ("跳过入库", "不要入库", "不入库", "不写数据库", "跳过db", "跳过DB")):
        update["execution_plan"] = ["source", "fetch", "preprocess", "qc", "report"]
    if any(keyword in text for keyword in ("只做预处理", "只跑预处理")):
        update["execution_plan"] = ["source", "fetch", "preprocess", "report"]
    return update or None


def parse_qc_review_response(value: str) -> dict[str, str] | None:
    text = clean_terminal_text(value)
    lowered = text.lower()
    approve = any(token in text for token in ("人工通过", "确认通过", "同意入库", "批准入库")) or lowered.startswith("approve")
    reject = any(token in text for token in ("拒绝", "不通过", "不入库", "退回")) or lowered.startswith("reject")
    if approve == reject:
        return None
    note = text
    for prefix in ("人工通过", "确认通过", "同意入库", "批准入库", "approve", "拒绝", "不通过", "不入库", "退回", "reject"):
        if note.lower().startswith(prefix.lower()):
            note = note[len(prefix):].lstrip("：:，,。 ")
            break
    return {"decision": "approve" if approve else "reject", "note": note}


def parse_input_qc_response(value: str, subjects: list[str]) -> dict[str, Any] | None:
    text = clean_terminal_text(value)
    lowered = text.lower()
    if text in {"继续", "同意", "确认", "执行", "可以"} or lowered in {"yes", "y", "true", "ok", "approve"}:
        return {"approved": True, "subject_actions": {}}
    if text in {"停止", "取消", "拒绝", "不执行"} or lowered in {"no", "n", "false", "stop", "reject"}:
        return {"approved": False, "subject_actions": {}}

    action_aliases = {
        "skip": ("skip", "跳过去颅骨", "跳过脑提取", "不再去颅骨", "输入已去颅骨"),
        "force": ("force", "强制去颅骨", "强制脑提取"),
        "auto": ("auto", "自动判断", "自动处理"),
    }
    actions: dict[str, str] = {}
    for action, aliases in action_aliases.items():
        if any(re.search(rf"(?:全部|所有|all).*{re.escape(alias)}", lowered, flags=re.IGNORECASE) for alias in aliases):
            actions.update({subject: action for subject in subjects})
        for subject in subjects:
            subject_patterns = (rf"sub-{re.escape(subject)}", rf"被试\s*{re.escape(subject)}")
            if any(
                re.search(rf"{pattern}[^，。；;\n]*{re.escape(alias)}", lowered, flags=re.IGNORECASE)
                for pattern in subject_patterns for alias in aliases
            ):
                actions[subject] = action
    return {"approved": True, "subject_actions": actions} if actions else None
