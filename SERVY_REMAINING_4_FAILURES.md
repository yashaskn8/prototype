# SERVY — Remaining Failures Investigation (`SERVY_REMAINING_4_FAILURES.md`)

Measured on real Django + V5 (pool 5), tenant-wide universe. Reused dev set; **no labels were changed**.

| Query | Django first-stage rank | After V5 | V5 logits | Verdict |
|---|---|---|---|---|
| `OP26-SLV-144-2` ("…operating v specification 5") | **1** | **5** | gold −8.51 vs best (chunk 75) −4.18 | **Reranker regression.** First stage was right; V5 demotes it. |
| `OP26-SLV-145-2` ("…maintenance schedule intervals") | **1** | **2** | gold −5.85 vs chunk 40 −1.97 | **Reranker regression.** Gold (`## Monthly Preventive Maintenance`) correct; V5 prefers a generic chunk. |
| `OP26-SLV-143-0` | 2 | 2 | 142: +9.07, gold 143: +8.39 | Two consecutive chunks of one section (steps vs expected result). Chunk-level label ambiguity; document and section correct. |
| `OP26-SLV-76-2` | 4 | 2 | 79 (`Stop and escalate if`): +4.15; gold 76: −5.62 | `REQUIRES_SME_ADJUDICATION` (see below). Not relabeled. |

## Attempted fix: margin-gated override (REJECTED)
Let V5 override the first-stage top-1 only when its logit margin ≥ m (script `scratch/margin_gate.py`, 41 authorized queries):
m=0: 37/41 (3 better, 2 worse) · m=1–3: 36/41 · m=4: 37/41 · m≥5: 37/41 (1 better, 0 worse).
The two harmful cases have margins 4.32 and 3.88, the helpful ones 12.48, 4.88 and 0.37; **no threshold separates them**. A threshold would be tuned to the 42 dev queries and is not justified. No code change made.

## Review package: `OP26-SLV-76-2` (SME needed)
- Query: "Industrial Touch Terminal & Asus Celeron Controller Guide safety precautions and escalation criteria" (doc 139).
- Silver gold: chunk 76 `## When touch inputs do not register…`. V5 top: chunk 79 `## Stop and escalate if`.
- Reviewer questions: (1) does 76 contain any safety/escalation content? (2) is 79 the intended answer? (3) should both be gold? Until a verified reviewer answers, the case stays unadjudicated and counts as a miss.

## Honest conclusion
Two of four remaining failures are caused by V5 *lowering* correct first-stage results. A V5 continuation on the same 111 queries already produced 0 net change; the data to fix this properly (new SME-approved, family-disjoint hard negatives for maintenance-schedule and voltage-spec queries) does not exist yet.
