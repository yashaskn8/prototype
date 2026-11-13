#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root regardless of the caller's directory.
cd "$(dirname "$0")/../.."
if [ ! -d .venv ]; then
  echo "Virtual environment not found. Run tools/launchers/setup_linux.sh first."
  exit 1
fi
source .venv/bin/activate
python manage.py migrate
python manage.py runserver
