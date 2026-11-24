# Acceptance round 2026-11-23/24 — summary

Real models (BGE-small, V5 reranker, local Qwen) on the `servy_dev` customer demo, three assets.
Full transcripts quote approved manual text and stay in the private evaluation folder; this page
holds aggregates only. Thresholds were fixed before the round and were not changed.

## Datasets

| Set | Turns | Purpose |
|---|---:|---|
| A | 89 | Fixed multi-turn conversations (known regressions) |
| B | 34 | Manual-coverage questions written from the section inventory |
| C | 35 | Sealed holdout, run once before its fixes; a known-regression set afterwards |
| D | 42 | Second sealed holdout, frozen before any C-driven fix, run once after all fixes |

## Results

| Set | Machine checks | Strict reading (strong + policy-correct / total) |
|---|---|---|
| A | rubric only | 71 / 89 |
| B | 31 / 34 | 29 / 34 |
| C | 35 / 35 (sealed first run: 25 / 35) | 33 / 35 |
| D | 34 / 42 | 23 / 42 |

"Policy-correct" means an honest refusal the policy requires: technician-only work or a value the
manuals do not give to customers.

Across all 200 final turns: 0 ungrounded claims (every displayed claim is verbatim in its cited
section), 0 cross-asset citations and 0 internal-jargon answers. One automatic failure: a
"periodic service" question in D was answered from a document's purpose section only.

## Gates

| Gate | Result |
|---|---|
| Known regressions >= 90% strong | Not met (A 80%, B 85%, C 94%) |
| Holdout >= 85% strong | Not met (D 55%) |
| Zero automatic failures | Not met (1) |

**Readiness: NOT READY.** Independent acceptance also needs the company VM, original manual files
for the checksum gate, and a second rater.

## Root causes fixed (with regression tests)

Each fix was traced to its earliest defective stage and tested with different wording and negative cases.

- Term matching: derivational roots, interval comparison by length, heading negation ("does start" is
  not "DOES NOT START").
- Sufficiency: symptom concepts for noise, airflow, capacity, compressor, fan, high/low pressure and
  does-not-start; generic inquiry words ("problem", "allowed") no longer required in a passage; fault
  qualifiers counted by family; interval sections support "how often" questions.
- Answer stage: escalation guidance, technician-only wording, front matter only for document and
  timing questions, value and role questions, whole-schedule answers, timing answers without fault causes.
- Conversation state: chat shorthand, corrections, retractions, meta requests, a question is never a
  complaint or a completed check, no filler or asset name added to self-contained queries, check
  requests anchored to the current issue.

## Known gaps found by holdout D (not fixed; D is spent)

Imperative requests without a question mark ("list the possible causes", "simpler please"), escalation
and note-taking phrasings, "stopped cooling", passive ticket status, "are these two problems
connected?", corrections prefaced "my mistake", installation acceptance and periodic-service
questions. Fixing them needs a new sealed holdout written before the fixes.
