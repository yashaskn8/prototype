# SERVY — Company Training Gate Report (`SERVY_COMPANY_TRAINING_GATE_REPORT.md`)

Two statuses must not be conflated:
- `READY_FOR_EXPLORATORY_FINE_TUNING` (preflight): certifies only that 111 selected train rows exactly match the recorded subset, each has an active negative, and train/dev families don't overlap. It certifies **no** permission, **no** SME approval, **no** independent holdout.
- `V6_PLAN_READY_TRAINING_NOT_AUTHORIZED`: the plan is valid; `check_permission()` still requires a signed organizational record bound to the train-subset SHA-256.

| Requirement | Status |
|---|---|
| Organizational permission to use company documents | **MISSING** |
| Verified source ownership/usage scope | Registry-metadata only; original bytes not verified |
| SME-approved labels | **No** (`expert_labels_approved=false`) |
| Train/dev separation | Yes (13/5 families) |
| Independent evaluation set | **No** |
| Secure compute | No GPU locally (`CUDA_REQUIRED_FOR_V6_FINETUNING`) |
| Checkpoint provenance | V5 hash verified |

**Decision: no company-data training was run.** Existing evidence also argues against it: V6 on the same data gave 0 net change, and in Django V5 currently regresses two correct first-stage hits. Pipeline, leakage checks and tests exist (synthetic) for use once authorization and data arrive.
