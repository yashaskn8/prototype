# IDE test checklist (Windows, no Docker)

## First setup

1. Install Python 3.11+.
2. Open this folder in your IDE terminal.
3. Run:

```bat
setup_windows.bat
```

The script creates `.venv`, installs requirements, creates migrations, migrates the SQLite database, seeds the demo once and runs `manage.py check`.

## Security/regression tests

```bat
.venv\Scripts\activate
python manage.py test core -v 2
python red_team_static.py
python red_team_offline.py
```

Expected offline result from this package: **54 static + 23 behavioural checks passing**.

## Start the app

```bat
run_windows.bat
```

Open `http://127.0.0.1:8000/`.

## Demo credentials

- `admin / ServyAdmin!2026`
- `manager / ServyManager!2026`
- `engineer / ServyTech!2026`
- `freshdairy / FreshDairy!2026`
- `aero_admin / AeroAdmin!2026` (separate tenant)

## Reset demo data only when intentional

```bat
reset_demo_windows.bat
```

Normal startup does **not** reset or reseed the database.

## Optional local LLM

The default is deterministic local extractive mode. No Ollama or Docker is needed.

If Ollama is installed later, set `SERVY_LLM_PROVIDER=auto` or `ollama`, keep the endpoint on loopback, and run the local model separately.
