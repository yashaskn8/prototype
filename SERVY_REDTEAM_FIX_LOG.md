# SERVY — Red-Team Fix Log (`SERVY_REDTEAM_FIX_LOG.md`)

| # | Finding | Severity | Reproduction | Fix | Verification |
|---|---|---|---|---|---|
| 1 | Tokenizer created false equivalences: `step 3 a`→`3a`, `figure 5 c`→`5c`, `room 5 f`→`5f`, `voltage 5 3 phase`→`3v` | Medium (wrong-spec retrieval risk) | `scratch/tok_attack.py` | Single-letter units now require attachment (`10A`, `2T`); spec-phrase regex only takes `/`,`,`-joined lists; added `5 volts`/`5-Volt`→`5v` | 5 new regression tests in `core/tests_production_rag_accuracy.py`; 9/9 tokenizer tests pass; Django first-stage unchanged 36/42 (no regression) |
| 2 | Evaluator counted a confidential-document retrieval as a hit (`OP26-SLV-52-0`) | High (metric integrity, would reward a confidentiality leak) | `SERVY_DJANGO_VS_PYTHON_PARITY_REPORT.md` §2.1 | Not a code bug; denominator for authorized retrieval is now 41. Label not edited | Django returns nothing for it (correct) |
| 3 | Reported "hybrid" results not produced by Django (dense leg silently empty, backend `fallback`) | High (claim integrity) | `django_replay.py` hybrid == sparse_only | **Not fixed.** Needs promoted/indexed dense profile; documented | Open |
| 4 | V5 regresses `144-2` and `145-2` | Medium | `inspect_fail.py` | Margin-gating tried and rejected (no separating threshold) | Open |
| 5 | Production scope returns 0/42 with no asset | Info (by-design security) | `django_replay.py` C1 | None | Needs asset-scoped replay |

Not tested in this pass (see report): tenant-isolation, prompt-injection, document-integrity and concurrency attacks.
