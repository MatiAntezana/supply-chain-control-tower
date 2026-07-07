#!/usr/bin/env bash
# Launch API (background) + dashboard (foreground) in one container.
set -e
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 &
exec streamlit run app/dashboard.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true
