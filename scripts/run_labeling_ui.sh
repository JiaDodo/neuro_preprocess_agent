#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/streamlit run src/neuro_preprocess_agent/labeling_app.py \
  --server.address 127.0.0.1 \
  --server.port 8502 \
  --server.headless true \
  -- "$@"
