#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"$PYTHON_BIN" -m neuro_preprocess_agent.cli \
  run \
  --request "帮我从公开数据源下载一个示例数据集，完成预处理、QC、入库和报告生成" \
  --config configs/example.json \
  --yes
