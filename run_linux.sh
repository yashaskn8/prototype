#!/usr/bin/env bash
set -euo pipefail
if [ ! -d .venv ]; then
  echo "Virtual environment not found. Run ./setup_linux.sh first."
  exit 1
fi
source .venv/bin/activate
python manage.py migrate
python manage.py runserver
