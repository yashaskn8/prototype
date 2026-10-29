# SERVY V5 Reranker Failure Analysis

> [!IMPORTANT]
> This report documents the empirical behavior of the V5 BGE cross-encoder reranker
> on the 3 remaining authorized failures in the 42-query C2 benchmark.

## Executive Summary

With true hybrid retrieval (dense BGE-small + sparse BM25 + RRF fusion) and V5 reranker
at pool_size=5, Servy achieves **38/42 Hit@1 (90.48%)** on the full 42-query benchmark,
and **38/41 = 92.68%** on the authorized 41-query set (excluding confidential OP26-SLV-52-0).

The 3 remaining authorized failures all have gold chunks in the top-5 pre-reranker.
The V5 reranker does NOT improve any of these 3 cases — and in 1 case **actively makes things worse**.

## Configuration Tested

| Parameter | Value |
|-----------|-------|
| Database | MySQL 8.0.41 |
| Embedding | `BAAI/bge-small-en-v1.5` (384d) |
| Retrieval | Hybrid (dense+sparse RRF, w_dense=0.55, w_sparse=0.45) |
| Reranker | V5 BGE cross-encoder (SHA: `b1cf23...`) |
| Evaluation | C2 variant (tenant-wide, no asset scoping) |

---

## Failure 1: OP26-SLV-144-2

**Question:** "Asus Celeron & Service Terminal Troubleshooting Guide operating v specification 5"

**Gold chunk:** 144 — "Diagnostic Port / USB Field Equipment Communication Lost"

| Stage | Gold Rank | Top-1 Chunk |
|-------|-----------|-------------|
| Pre-reranker (hybrid) | **3** (score=0.9134, HIGHEST) | 75 (black screen) |
| Post-reranker pool=5 | **4** | 75 |
| Post-reranker pool=10 | **7** | 75 |

**Root cause:** Semi-synthetic question. "Operating v specification 5" is a generated
query pattern that poorly describes the actual troubleshooting content (USB/COM port
communication loss). The V5 cross-encoder assigns higher logits to "black screen" and
"power" troubleshooting chunks, which are semantically more natural matches for the
vague query phrasing.

**V5 impact:** HARMFUL — demotes gold from rank 3 to rank 4 (pool=5) or rank 7 (pool=10).

**Classification:** Gold label quality issue + reranker semantic mismatch.

---

## Failure 2: OP26-SLV-143-0

**Question:** "Asus Celeron & Service Terminal Troubleshooting Guide terminal screen black or system unexpectedly shuts down"

**Gold chunk:** 143 — "Terminal Screen Black or System Unexpectedly Shuts Down" (Expected result / escalation criteria)

| Stage | Gold Rank | Top-1 Chunk |
|-------|-----------|-------------|
| Pre-reranker (hybrid) | **3** (score=1.1087, HIGHEST) | 142 (same heading, Step 1) |
| Post-reranker pool=5 | **2** | 142 |
| Post-reranker pool=10 | **2** | 142 |

**Root cause:** Gold label ambiguity. Chunks 142 and 143 are from the **same document**
and share the **same heading** ("Terminal Screen Black or System Unexpectedly Shuts Down").
Chunk 142 contains Step 1 (verify DC barrel connector), chunk 143 contains expected
result and escalation. Both are valid answers — the gold label favors chunk 143 but
chunk 142 is arguably more actionable for a field technician.

**V5 impact:** NEUTRAL — reranker consistently places gold at rank 2, same as the baseline.
The reranker correctly identifies both as relevant; it simply prefers the procedural
steps (142) over the expected-result summary (143).

**Classification:** Gold label ambiguity — chunk 142 is a valid alternative answer.

---

## Failure 3: OP26-SLV-76-2

**Question:** "Industrial Touch Terminal & Asus Celeron Controller Guide safety precautions and escalation criteria"

**Gold chunk:** 76 — "When touch inputs do not register or click in the wrong position"

| Stage | Gold Rank | Top-1 Chunk |
|-------|-----------|-------------|
| Pre-reranker (hybrid) | **2** (score=0.6613) | 79 (Stop and escalate if...) |
| Post-reranker pool=5 | **2** | 79 |
| Post-reranker pool=10 | **4** | 79 |

**Root cause:** **Gold label error.** The question explicitly asks for "safety precautions
and escalation criteria." Chunk 79 ("Stop and escalate if — Motherboard capacitor
bulging or liquid damage detected, Touch digitizer glass has physical spiderweb cracks,
Storage drive reports critical SMART failures...") is a BETTER match than chunk 76
(touch calibration steps). The system correctly retrieves chunk 79 as the most relevant
answer for escalation criteria.

**V5 impact:** NEUTRAL to CORRECT — the reranker agrees with the pre-reranker that
chunk 79 is the better match. Pool=10 makes things worse by promoting other irrelevant chunks.

**Classification:** Gold label error — chunk 79 is the correct answer.

---

## Pool Size Sweep Results

| Pool Size | Hit@1 (42q) | Hit@1 (41q auth) | MRR@10 | Notes |
|-----------|-------------|-------------------|--------|-------|
| **5 (default)** | **38/42 (90.48%)** | **38/41 (92.68%)** | **0.9345** | **Best** |
| 8 | 37/42 (88.10%) | 37/41 (90.24%) | 0.9167 | Loses OP26-SLV-145-2 |
| 10 | 36/42 (85.71%) | 36/41 (87.80%) | 0.8982 | Loses 74-1, 145-2 |
| **No reranker** | **33/42 (78.57%)** | **33/41 (80.49%)** | **0.873** | Loses 5 more queries |

The V5 reranker at pool=5 adds **+5 Hit@1** over no reranker (38 vs 33).
Removing the reranker loses: OP26-SLV-74-0, OP26-SLV-139-0, OP26-SLV-74-1,
OP26-SLV-78-0, OP26-SLV-73-0 — all drop from HIT@1 to rank=2.

> [!WARNING]
> **Increasing pool size HURTS accuracy.** Pool=8 loses OP26-SLV-145-2 (rank 2 instead of HIT@1).
> Pool=10 demotes OP26-SLV-144-2 to rank 7 and OP26-SLV-76-2 to rank 4.
> The V5 reranker introduces false positives when more candidates enter its pool.

---

## Recommendations

### 1. Keep Pool Size = 5
Pool=5 is the empirically optimal configuration. Do not increase.

### 2. Correct Gold Labels
- **OP26-SLV-76-2:** Update gold label from chunk 76 to chunk 79 (escalation criteria).
- **OP26-SLV-143-0:** Accept chunk 142 as alternative gold (same heading, same topic).

### 3. Review Semi-Synthetic Questions
- **OP26-SLV-144-2:** The question "operating v specification 5" is a poor proxy for
  the actual troubleshooting scenario. Consider revising the benchmark question to
  better reflect realistic technician queries.

### 4. Net Accuracy After Label Corrections
If OP26-SLV-76-2 gold is corrected (76→79, which the system already retrieves at rank 1)
and OP26-SLV-143-0 accepts 142 as valid (already at rank 1):

- **Corrected Hit@1: 40/42 (95.24%)** on full set
- **Corrected authorized Hit@1: 40/41 (97.56%)**

> [!CAUTION]
> These corrected scores require independent validation of the gold label changes.
> Do not claim 95%+ accuracy until an independent evaluator confirms the label corrections.

---

## Evidence Files

- Full benchmark: `outputs/42q_c2_full_mysql.json`
- Pool sweep: `outputs/42q_pool_sweep.json`
- V5 deep dive: `scratch/v5_deep_dive.py` (output in task logs)

---

*Generated: 2026-10-29 | DB: MySQL 8.0.41 | Commit: 18d0277*
