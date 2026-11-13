# SERVY — Independent Evaluation Plan (`SERVY_INDEPENDENT_EVALUATION_PLAN.md`)

Why needed: the recovered corpus has 18 source families, all used (13 train, 5 dev, 0 unused); the 64-question holdout overlaps training in 10/10 families. No existing data can support an independent claim.

1. **Acquire** ≥6–10 new authorized manuals from *new* source families (not in `ai_source_family_recommendations.jsonl`), with written organizational permission.
2. **Write** ≥100 questions across: exact fault code, voltage/spec, maintenance schedule, safety/escalation, multi-step, equipment-confusion, unsupported/abstain, tenant-isolation probes, confidential-doc probes (expected: no retrieval).
3. **Label** by ≥2 independent SMEs; record reviewer IDs outside the repo; adjudicate disagreements; chunk, section and document gold recorded separately.
4. **Freeze** before any model run: files + SHA-256 in a manifest; run `servy_company_data_pipeline.verify_family_disjoint_splits` against all historical families/chunks/queries.
5. **Evaluate once** with the frozen configuration chosen on the dev set; report Hit@1, Recall@10, MRR, answer correctness, citation validity, abstention, with Wilson 95% intervals (n=100 → ±~4 points at 95%, so a point estimate of 95% cannot by itself prove ≥95%).
6. **No re-runs** to pick the best score; any change requires a fresh unseen set.
7. Evaluate through the real Django API with asset scoping (not tenant-wide), recording latency and isolation violations.
