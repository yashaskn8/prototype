#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root regardless of the caller's directory.
cd "$(dirname "$0")/../.."
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check
echo "Setup complete. Run tools/launchers/run_linux.sh to start Servy RAG."
