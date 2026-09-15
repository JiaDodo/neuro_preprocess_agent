from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from neuro_preprocess_agent.config import PROJECT_ROOT, load_config, project_path
from neuro_preprocess_agent.db import check_database, init_mysql
from neuro_preprocess_agent.graph import WORKERS
from neuro_preprocess_agent.interaction import clean_terminal_text, parse_human_response
from neuro_preprocess_agent.runtime import AgentRuntime, interrupt_message


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(PROJECT_ROOT / ".env.mysql.local")
        if not os.environ.get("NEURO_AGENT_MYSQL_PASSWORD") and os.environ.get("MYSQL_PASSWORD"):
            os.environ["NEURO_AGENT_MYSQL_PASSWORD"] = os.environ["MYSQL_PASSWORD"]
    except ImportError:
        pass


def _read_approval() -> str | bool:
    try:
        return clean_terminal_text(input("\n你: "))
    except EOFError:
        return False


def _drive(runtime: AgentRuntime, result: dict, thread_id: str, interactive: bool, auto_approve: bool = False) -> dict:
    while (message := interrupt_message(result)) is not None:
        requires_explicit_review = any(
            marker in message for marker in ("运行前硬检查", "fMRIPrep 执行前", "自动 QC")
        )
        if not interactive and (not auto_approve or requires_explicit_review):
            return result
        print(f"\nAgent: 需要确认。\n{message}")
        response = True if auto_approve and not requires_explicit_review else _read_approval()
        if not interactive:
            result = runtime.resume(response, thread_id)
            continue

        log_paths = [Path(job["log_file"]) for job in result.get("preprocess_jobs", []) if job.get("log_file")]
        log_hint = f" 日志：{log_paths[0]}" if len(log_paths) == 1 else ""
        print(f"Agent: 已收到回复，正在执行后续步骤。长任务会每 30 秒报告一次状态。{log_hint}")
        stopped = threading.Event()

        def show_heartbeat(
            stop_event: threading.Event = stopped,
            paths: tuple[Path, ...] = tuple(log_paths),
        ) -> None:
            started = time.monotonic()
            if stop_event.wait(5):
                return
            while not stop_event.is_set():
                existing_logs = [path for path in paths if path.is_file()]
                latest_log = max(existing_logs, key=lambda path: path.stat().st_mtime) if existing_logs else None
                elapsed = int(time.monotonic() - started)
                suffix = f"，最新日志：{latest_log}" if latest_log else ""
                print(f"\nAgent: 任务仍在运行，已用时 {elapsed} 秒{suffix}", flush=True)
                stop_event.wait(30)

        heartbeat = threading.Thread(target=show_heartbeat, name="cli-progress", daemon=True)
        heartbeat.start()
        try:
            result = runtime.resume(response, thread_id)
        finally:
            stopped.set()
            heartbeat.join(timeout=1)
    return result


def _print_result(result: dict, as_json: bool = False) -> None:
    if as_json:
        redacted = copy.deepcopy(result)
        for section, key in (("supervisor", "api_key"), ("database", "password")):
            if redacted.get("config", {}).get(section, {}).get(key):
                redacted["config"][section][key] = "***"
        print(json.dumps(redacted, ensure_ascii=False, indent=2, default=str))
        return
    report = result.get("report", {})
    qc = result.get("qc_result", {})
    db = result.get("db_result", {})
    preflight = result.get("preflight_result", {})
    errors = result.get("errors", [])
    print(f"Agent: 运行状态：{report.get('status', result.get('stage', 'unknown'))}")
    if preflight:
        print(
            f"Agent: 运行前检查：{'通过' if preflight.get('passed') else '未通过'}；"
            f"警告={len(preflight.get('warnings', []))}，硬失败={len(preflight.get('hard_failures', []))}"
        )
    if qc:
        if qc.get("status") == "planned":
            print("Agent: QC：已生成检查计划，未执行真实影像检查")
        elif qc.get("manual_review", {}).get("decision") == "approve":
            print("Agent: QC：软阈值经人工复核后通过，审计记录已保存")
        elif qc.get("review_required"):
            print("Agent: QC：硬性完整性检查通过，但运动指标需要人工复核；数据库写入已阻断")
        else:
            print(f"Agent: QC：{'通过' if qc.get('passed') else '未通过'}")
        visual = qc.get("visual_model_qc", {})
        if visual.get("status") == "completed":
            statuses = sorted({item.get("status", item.get("decision")) for item in visual.get("subjects", [])})
            print(
                f"Agent: 视觉 QC：shadow 完成，{len(visual.get('subjects', []))} 名被试，"
                f"状态={','.join(str(item) for item in statuses)}"
            )
            decisions = {
                str(item.get("subject")): item.get("decision")
                for item in visual.get("subjects", [])
                if item.get("decision") not in {None, "unscored"}
            }
            if decisions:
                print("Agent: VLM shadow 决定：" + ", ".join(f"sub-{key}={value}" for key, value in decisions.items()))
        elif visual.get("status") == "unavailable":
            print(f"Agent: 视觉 QC：不可用，不影响规则门禁；{visual.get('error')}")
    if db:
        print(f"Agent: 入库：{db.get('status', 'unknown')} ({db.get('backend', 'unknown')})")
    if errors:
        for error in errors:
            print(f"Agent: 错误 [{error.get('node')}]: {error.get('message')}")
    if report.get("report_path"):
        print(f"Agent: 报告：{report['report_path']}")


def run_once(args: argparse.Namespace) -> None:
    runtime = AgentRuntime()
    thread_id = args.thread_id or f"run-{uuid4().hex[:10]}"
    if args.resume is not None:
        if not args.thread_id:
            raise SystemExit("使用 --resume 时必须提供原来的 --thread-id。")
        response = parse_human_response(args.resume)
        if response is None:
            response = args.resume
        result = runtime.resume(response, thread_id)
    else:
        if not args.request:
            raise SystemExit("请使用 --request 提供任务。")
        result = runtime.start(args.request, args.config, thread_id)
    result = _drive(runtime, result, thread_id, interactive=not args.no_interactive, auto_approve=args.yes)
    if interrupt_message(result):
        print(f"Agent: 任务已暂停，thread_id={thread_id}。使用 --thread-id {thread_id} --resume '继续' 恢复。")
        return
    _print_result(result, args.json)


def chat(args: argparse.Namespace) -> None:
    runtime = AgentRuntime()
    session_id = args.session_id or f"chat-{uuid4().hex[:8]}"
    task_index = 0
    last_result: dict = {}
    print(f"Agent: 神经影像预处理会话已启动。session_id={session_id}")
    print("Agent: 直接描述任务即可；输入 /help 查看命令，输入 exit 退出。")
    while True:
        try:
            user_input = clean_terminal_text(input("\n你: "))
        except EOFError:
            print("\nAgent: 已退出。")
            return
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q", "退出"}:
            print("Agent: 已退出。")
            return
        if user_input == "/help":
            print("Agent: 示例：帮我处理 OpenNeuro ds000001；处理本地 /path/to/data，正式运行，并行数为2。可用 /status、/tools、/runs。")
            continue
        if user_input == "/status":
            if last_result:
                _print_result(last_result)
            else:
                print("Agent: 当前会话还没有运行任务。")
            continue
        if user_input == "/tools":
            for item in WORKERS.catalog():
                print(f"Agent: {item['name']}: {item['description']} [risk={item['risk']}]")
            continue
        if user_input == "/runs":
            report_dir = PROJECT_ROOT / "runs/reports"
            reports = sorted(report_dir.glob("*_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:5]
            print("Agent: 最近运行：" + ("\n" + "\n".join(str(item) for item in reports) if reports else "暂无"))
            continue
        if user_input.lower() in {"你好", "您好", "hi", "hello"}:
            print("Agent: 你好。请告诉我数据在本地哪个路径，或给出 OpenNeuro 的 ds 编号。")
            continue

        task_index += 1
        thread_id = f"{session_id}-task-{task_index}"
        try:
            result = runtime.start(user_input, args.config, thread_id)
            last_result = _drive(runtime, result, thread_id, interactive=True)
            _print_result(last_result)
        except KeyboardInterrupt:
            print("\nAgent: 当前交互已中断；已完成状态仍保存在 checkpoint 中。")
        except Exception as exc:  # noqa: BLE001 - keep the interactive session alive after task failures
            print(f"Agent: 启动失败：{type(exc).__name__}: {exc}")


def doctor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    preprocess = config["preprocess"]
    checks = [
        ("Python", True, sys.executable),
        ("docker", shutil.which("docker") is not None, shutil.which("docker") or "not found"),
        ("openneuro-py", importlib.util.find_spec("openneuro") is not None, "python -m openneuro"),
        ("dcm2bids", project_path(preprocess["dcm2bids"], config["project_root"]).exists(), preprocess["dcm2bids"]),
        ("FreeSurfer license", project_path(preprocess["freesurfer_license"], config["project_root"]).exists(), preprocess["freesurfer_license"]),
    ]
    if shutil.which("docker"):
        image = preprocess["fmriprep_image"]
        inspected = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        checks.append(("fMRIPrep image", inspected.returncode == 0, image))
    visual = config["qc"]["visual"]
    if visual["enabled"]:
        visual_dependencies = all(
            importlib.util.find_spec(name) is not None
            for name in ("nibabel", "numpy", "PIL", "cairosvg")
        )
        checks.append(("visual QC dependencies", visual_dependencies, "nibabel, numpy, Pillow, CairoSVG"))
        if visual["model_backend"] in {"dinov2_embedding", "qwen2_5_vl"}:
            model_dependencies = all(
                importlib.util.find_spec(name) is not None
                for name in ("torch", "torchvision", "transformers")
            )
            checks.append(("visual model dependencies", model_dependencies, "torch, torchvision, transformers"))
            model_path = project_path(visual["model_id"], config["project_root"])
            if model_path.is_dir():
                model_files = list(model_path.glob("*.safetensors"))
                weights_complete = bool(model_files)
                for model_file in model_files:
                    try:
                        with model_file.open("rb") as handle:
                            header_size = int.from_bytes(handle.read(8), "little")
                            header = json.loads(handle.read(header_size))
                        data_size = max(
                            value["data_offsets"][1]
                            for key, value in header.items()
                            if key != "__metadata__"
                        )
                        weights_complete &= model_file.stat().st_size == 8 + header_size + data_size
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        weights_complete = False
                checks.append((
                    "visual model",
                    (model_path / "config.json").is_file() and weights_complete,
                    f"{model_path} ({len(model_files)} weight file(s))",
                ))
            if visual["device"].startswith("cuda") and importlib.util.find_spec("torch") is not None:
                import torch

                checks.append(("visual model CUDA", torch.cuda.is_available(), visual["device"]))
    database = check_database(config)
    checks.append(("database", database.get("status") == "ready", json.dumps(database, ensure_ascii=False)))
    for name, passed, detail in checks:
        print(f"{'OK' if passed else 'FAIL':4} {name}: {detail}")


def mysql_init(args: argparse.Namespace) -> None:
    raw = json.loads(Path(args.mysql_config).read_text(encoding="utf-8"))
    config = raw if isinstance(raw.get("database"), dict) else {"database": raw}
    result = init_mysql(config, create_database=args.create_database)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def show_tools(_: argparse.Namespace) -> None:
    print(json.dumps(WORKERS.catalog(), ensure_ascii=False, indent=2))


def show_status(args: argparse.Namespace) -> None:
    state = AgentRuntime().state(args.thread_id)
    if not state:
        raise SystemExit(f"没有找到 thread_id={args.thread_id} 的 checkpoint。")
    _print_result(state, args.json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph neuroimaging preprocessing agent")
    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="Start a natural-language interactive session")
    chat_parser.add_argument("--config", type=Path, default=Path("configs/llm_supervisor_example.json"))
    chat_parser.add_argument("--session-id")

    run_parser = subparsers.add_parser("run", help="Run or resume one task")
    run_parser.add_argument("--request")
    run_parser.add_argument("--config", type=Path, default=Path("configs/llm_supervisor_example.json"))
    run_parser.add_argument("--thread-id")
    run_parser.add_argument("--resume")
    run_parser.add_argument("--yes", action="store_true", help="Approve the generated plan automatically")
    run_parser.add_argument("--no-interactive", action="store_true")
    run_parser.add_argument("--json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime dependencies and configured services")
    doctor_parser.add_argument("--config", type=Path, default=Path("configs/example.json"))

    mysql_parser = subparsers.add_parser("mysql-init", help="Create the MySQL database schema")
    mysql_parser.add_argument("--mysql-config", type=Path, default=Path("configs/mysql.example.json"))
    mysql_parser.add_argument("--create-database", action="store_true")

    status_parser = subparsers.add_parser("status", help="Inspect a persisted LangGraph thread")
    status_parser.add_argument("--thread-id", required=True)
    status_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("tools", help="List registered workers and risk classes")
    return parser


def main() -> None:
    _load_env()
    args = build_parser().parse_args()
    if args.command == "run":
        run_once(args)
    elif args.command == "doctor":
        doctor(args)
    elif args.command == "mysql-init":
        mysql_init(args)
    elif args.command == "tools":
        show_tools(args)
    elif args.command == "status":
        show_status(args)
    else:
        if args.command is None:
            args = argparse.Namespace(config=Path("configs/llm_supervisor_example.json"), session_id=None)
        chat(args)


if __name__ == "__main__":
    main()
