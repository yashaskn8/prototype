# Operation 21 historical diagnostics

This folder contains ten standalone read-only diagnostic and ablation scripts from Operation 21:

- accuracy audit
- chunking ablations
- retrieval ablations
- final diagnostic aggregation
- dataset/source/Chroma integrity audit
- manual label review export
- retrieval latency benchmark
- training-negative similarity audit
- Operation 20 retrieval replay
- offline reranker audit

These are not production request handlers and are not included in the normal customer or technician runtime. They rely on local evaluation inputs/checkpoints, and write their private results under the ignored repository directory `private_training_data/op21_accuracy_audit/`.

Run from the repository root using the project's virtual environment. For example:

```powershell
.\.venv\Scripts\python.exe -B tools\op21\op21_accuracy_audit.py
```

Each script keeps the repository root stable using `Path(__file__).resolve().parents[2]`. They remain here for reproducibility; historical metrics must not be presented as current production accuracy.
