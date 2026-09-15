from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from neuro_preprocess_agent.config import project_path
from neuro_preprocess_agent.io_utils import advisory_lock, atomic_write_text

MYSQL_RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS preprocessing_runs (
    run_id VARCHAR(80) PRIMARY KEY,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    status VARCHAR(32) NOT NULL,
    request_text TEXT,
    source_type VARCHAR(32),
    source_reference TEXT,
    qc_passed BOOLEAN,
    subject_count INT NOT NULL DEFAULT 0,
    report_path TEXT,
    payload_json JSON NOT NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

MYSQL_SUBJECT_QC_SCHEMA = """
CREATE TABLE IF NOT EXISTS subject_qc_results (
    run_id VARCHAR(80) NOT NULL,
    subject_id VARCHAR(80) NOT NULL,
    qc_passed BOOLEAN NOT NULL,
    hard_checks_passed BOOLEAN NOT NULL,
    review_required BOOLEAN NOT NULL,
    approved_for_database BOOLEAN NOT NULL,
    metrics_json JSON NOT NULL,
    issues_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (run_id, subject_id),
    CONSTRAINT fk_subject_qc_run FOREIGN KEY (run_id)
        REFERENCES preprocessing_runs(run_id) ON DELETE CASCADE
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

MYSQL_SCHEMAS = (MYSQL_RUN_SCHEMA, MYSQL_SUBJECT_QC_SCHEMA)


def _mysql_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("database", config)
    password = settings.get("password") or os.environ.get(settings.get("password_env", "NEURO_AGENT_MYSQL_PASSWORD"))
    return {
        "host": settings.get("host", "127.0.0.1"),
        "port": int(settings.get("port", 3306)),
        "user": settings.get("user", "neuro_agent"),
        "password": password or "",
        "database": settings.get("database", "neuro_preprocess"),
        "charset": settings.get("charset", "utf8mb4"),
        "connect_timeout": int(settings.get("connect_timeout", 5)),
        "autocommit": True,
    }


def init_mysql(config: dict[str, Any], create_database: bool = False) -> dict[str, Any]:
    import pymysql

    settings = _mysql_settings(config)
    database = settings.pop("database")
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError("MySQL database name may contain only letters, digits, and underscores")
    if create_database:
        with pymysql.connect(**settings) as connection, connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    settings["database"] = database
    with pymysql.connect(**settings) as connection, connection.cursor() as cursor:
        for statement in MYSQL_SCHEMAS:
            cursor.execute(statement)
    return {"status": "ready", "backend": "mysql", "host": settings["host"], "port": settings["port"], "database": database}


def check_database(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("database", config)
    if settings.get("backend", "jsonl") == "jsonl":
        path = project_path(settings.get("jsonl_path", "runs/db_records.jsonl"), config.get("project_root"))
        return {"status": "ready", "backend": "jsonl", "path": str(path), "parent_exists": path.parent.exists()}
    try:
        import pymysql

        mysql = _mysql_settings(config)
        with pymysql.connect(**mysql) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return {
            "status": "ready",
            "backend": "mysql",
            "connection": "ok",
            "host": mysql["host"],
            "port": mysql["port"],
            "database": mysql["database"],
        }
    except Exception as exc:  # noqa: BLE001 - health check reports import and connection failures uniformly
        return {"status": "unavailable", "backend": "mysql", "error": f"{type(exc).__name__}: {exc}"}


def record_run(config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("database", {})
    backend = settings.get("backend", "jsonl")
    report_path = project_path(config["storage"]["report_dir"], config.get("project_root")) / f"{state['run_id']}_summary.json"
    record = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": state.get("run_id"),
        "status": "qc_passed" if state.get("qc_result", {}).get("passed") else "qc_failed",
        "request": state.get("request"),
        "source": state.get("source"),
        "input_qc_result": state.get("input_qc_result"),
        "input_qc_review": state.get("input_qc_review"),
        "qc_result": state.get("qc_result"),
        "processed_data": state.get("processed_data"),
        "report_path": str(report_path),
    }
    if backend == "jsonl":
        output_path = project_path(settings.get("jsonl_path", "runs/db_records.jsonl"), config.get("project_root"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_lock(output_path.with_suffix(output_path.suffix + ".lock")):
            retained: list[str] = []
            replaced = False
            if output_path.exists():
                for line in output_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        retained.append(line)
                        continue
                    if existing.get("run_id") == record["run_id"]:
                        replaced = True
                        continue
                    retained.append(line)
            retained.append(json.dumps(record, ensure_ascii=False, default=str))
            atomic_write_text(output_path, "\n".join(retained) + "\n")
        return {
            "status": "updated" if replaced else "written",
            "backend": "jsonl",
            "path": str(output_path),
            "run_id": record["run_id"],
        }

    import pymysql

    mysql = _mysql_settings(config)
    payload = json.dumps(record, ensure_ascii=False, default=str)
    source = state.get("source", {})
    reference = source.get("path") or source.get("dataset_id") or source.get("url")
    subjects = state.get("processed_data", {}).get("subjects", [])
    mysql["autocommit"] = False
    with pymysql.connect(**mysql) as connection:
        with connection.cursor() as cursor:
            for statement in MYSQL_SCHEMAS:
                cursor.execute(statement)
            cursor.execute("SELECT 1 FROM preprocessing_runs WHERE run_id=%s", (state["run_id"],))
            replaced = cursor.fetchone() is not None
            cursor.execute(
                """
                INSERT INTO preprocessing_runs (
                    run_id, created_at, updated_at, status, request_text, source_type,
                    source_reference, qc_passed, subject_count, report_path, payload_json
                ) VALUES (%s, NOW(), NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    updated_at=NOW(), status=VALUES(status), request_text=VALUES(request_text),
                    source_type=VALUES(source_type), source_reference=VALUES(source_reference),
                    qc_passed=VALUES(qc_passed), subject_count=VALUES(subject_count),
                    report_path=VALUES(report_path), payload_json=VALUES(payload_json)
                """,
                (
                    state["run_id"], record["status"], state.get("request"), source.get("type"),
                    reference, bool(state.get("qc_result", {}).get("passed")), len(subjects),
                    str(report_path), payload,
                ),
            )
            qc_result = state.get("qc_result", {})
            visual_subjects = {
                str(item.get("subject")): item
                for item in qc_result.get("visual_model_qc", {}).get("subjects", [])
            }
            for subject_qc in qc_result.get("subject_qc", []):
                subject = str(subject_qc.get("subject", ""))
                checks = subject_qc.get("checks", [])
                motion_checks = [item for item in checks if str(item.get("name", "")).endswith(":motion_review_thresholds")]
                issues = [item.get("name") for item in checks if not item.get("passed")]
                cursor.execute(
                    """
                    INSERT INTO subject_qc_results (
                        run_id, subject_id, qc_passed, hard_checks_passed, review_required,
                        approved_for_database, metrics_json, issues_json, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                        qc_passed=VALUES(qc_passed), hard_checks_passed=VALUES(hard_checks_passed),
                        review_required=VALUES(review_required),
                        approved_for_database=VALUES(approved_for_database),
                        metrics_json=VALUES(metrics_json), issues_json=VALUES(issues_json), updated_at=NOW()
                    """,
                    (
                        state["run_id"], subject, bool(subject_qc.get("passed")),
                        bool(qc_result.get("hard_checks_passed")), bool(qc_result.get("review_required")),
                        bool(qc_result.get("approved_for_database")),
                        json.dumps(
                            {
                                "motion": motion_checks,
                                "quantitative": subject_qc.get("quantitative_metrics"),
                                "visual_model_qc": visual_subjects.get(subject),
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                        json.dumps(issues, ensure_ascii=False, default=str),
                    ),
                )
        connection.commit()
    return {
        "status": "updated" if replaced else "written",
        "backend": "mysql",
        "database": mysql["database"],
        "run_id": state["run_id"],
    }
