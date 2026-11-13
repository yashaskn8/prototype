# Tools directory guide

The `tools/` directory contains launchers, active demo services, regression tests, operational diagnostics and historical research/evaluation utilities. It is not a single runtime package; most scripts are invoked explicitly.

## Current customer and technician demo entry points

- `Start-ServyRealDemo.ps1` — starts the customer RAG demo and its browser-facing Vite proxy.
- `servy_real_demo.py` — customer real-data demo server.
- `Start-ServyTechnicianDemo.ps1` — starts the separate technician demo and its Vite proxy.
- `servy_technician_demo.py` — technician ticket-workspace server.

## Three-asset evaluation tools

Use each three-asset tool only for its documented read-only or evaluation purpose. Check its module docstring and the current README command before running it. A frozen fixture result is not a production accuracy certification.

## Research and historical experiment families

Files prefixed with `op21_`, `op22_`, `op23_`, `op25_`–`op28_`, `servy_v4_`, `servy_v5_` and `servy_v6_` include reproducible research, audits, preparation scripts and their tests. They are not all application runtime modules. They are retained until a full import, CI, manual-invocation and artifact-dependency audit proves which can safely be retired.

## Cleanup rule

Do not delete or move a tool based only on its age or filename. Check imports, CI workflow references, README/launcher invocations, tests, model/data artifact paths and whether the script is the only reproducible audit for a prior result. Keep active launchers, tests and serving/evaluation dependencies in place.
