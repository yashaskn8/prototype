# Operation 28 — adversarial grounding evaluation

Historical standalone checks for numerical grounding, source-claim support, abstention behavior and evidence-reference accounting. These scripts are preserved for reproducibility and are not part of the customer request runtime.

Run explicitly from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -B tools\op28\op28_adversarial_grounding_tests.py
.\.venv\Scripts\python.exe -B tools\op28\op28_evaluate_zero_illusion.py
```

The evaluator needs the original local benchmark inputs and configured database/model environment. Its historical metric output is not a current production accuracy certification.
