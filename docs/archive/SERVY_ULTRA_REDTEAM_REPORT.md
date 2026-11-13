# SERVY — Ultra Red-Team Report (`SERVY_ULTRA_REDTEAM_REPORT.md`)

Scope of THIS pass (honest): Phase 0 verification, Pipeline-C Django parity, the four remaining failures, a tokenizer attack with fixes, a margin-gating experiment, release-gate status. **Not executed:** the large adversarial query suite, tenant-isolation / prompt-injection / document-integrity attack suites, the full ablation sweep, concurrency/destruction tests, company-data fine-tuning (unauthorized). These are open work, not passes.

## Findings (severity)
| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Reported 90.48% not reproduced by Django (37/42 = 88.10%) | High | Documented, parity report |
| 2 | Python evaluator rewarded retrieving a confidential doc (`52-0`); authorized denominator is 41 | High | Documented |
| 3 | "Hybrid" mode silently equals sparse-only in Django (dense leg empty, backend `fallback`); no explicit failure signal | High | Open |
| 4 | Tokenizer false equivalences (`3 a`→`3A` etc.) | Medium | **Fixed + tests** |
| 5 | V5 reranker regresses 2 correct first-stage hits; net +1 only; 3.4–5.6 s/query CPU | Medium | Open; margin gate rejected |
| 6 | Dev benchmark never exercised production authorization scope (0/42 without asset) | Info | Open |

## Where "95%" stands
Best honest number: Django + V5 = 37/41 authorized = **90.24%** on a reused dev set. 95% on 41 needs 39; no change here is justified by generalizable evidence. Independent validation is impossible with current data (see `SERVY_INDEPENDENT_EVALUATION_PLAN.md`, `SERVY_COMPANY_TRAINING_GATE_REPORT.md`).
