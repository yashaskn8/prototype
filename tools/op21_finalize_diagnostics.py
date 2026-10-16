"""Join read-only Op21 measurements into private per-error diagnostic records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "private_training_data" / "op21_accuracy_audit"


def read_jsonl(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


fair = {r["query_id"]: r for r in read_jsonl("per_query.jsonl")}
raw = {r["query_id"]: r for r in read_jsonl("op20_replay.jsonl")}
rerank_path = OUT / "rerank_holdout.json"
reranked = (
    {r["query_id"]: r for r in json.loads(rerank_path.read_text(encoding="utf-8"))["per_query"]}
    if rerank_path.exists() else {}
)

# These are passage-text findings only. They do not amend the frozen labels.
unsupported_label_ids = {
    "EVAL-ERR-01": "The labeled communication passage does not define the queried E-04 fault code.",
    "EVAL-TRAP-02": "Belt deflection adjustment is absent from the labeled interval passage.",
    "EVAL-TRAP-03": "Crankcase heater inspection is absent from the labeled periodic-service passage.",
    "EVAL-TRAP-04": "Trap siphon priming is absent from the labeled drainage passage.",
    "EVAL-TRAP-06": "No flare nut torque specification appears in the labeled checklist.",
    "EVAL-AMB-03": "No fan capacitor replacement procedure appears in the labeled symptom passage.",
    "EVAL-AMB-04": "No low-pressure switch operating limit or test appears in the labeled fault passage.",
    "EVAL-TECH-01": "No DC bus capacitor discharge procedure appears in the labeled inverter passage.",
    "EVAL-TECH-02": "No gauge manifold attachment procedure or specific PPE appears in the labeled planning passage.",
    "EVAL-XTLK-01": "The split-wall AC passage describes noise and vibration, not a centrifugal belt check.",
    "EVAL-XTLK-02": "The ductable AC passage describes poor cooling, not a VRF branch-selector calibration.",
}
possible_equivalent_ids = {
    "EVAL-ERR-04": "Top chunk 129 shares the F-07 fault heading with labeled chunk 128.",
    "EVAL-SYM-04": "Top chunk 108 independently addresses milk chiller temperature above 4 C.",
    "EVAL-SYM-06": "Top chunk 117 continues the labeled pallet-lift hydraulic section.",
    "EVAL-MSP-02": "Top chunk 107 independently addresses milk chiller temperature above 4 C.",
    "EVAL-MLT-01": "Top chunk 154 is the first part of the labeled AC adapter section.",
    "EVAL-TECH-03": "Top chunk 220 explicitly discusses power isolation and lockout.",
}

incorrect = []
for qid, observed in raw.items():
    row = fair[qid]
    if row["role"] != "RELEVANCE_EVALUATION" or observed["positive_rank"] == 1:
        continue
    positives = set(row["positive_chunk_ids"])
    def rank(key):
        return next((i + 1 for i, cid in enumerate(row[key]) if cid in positives), None)
    incorrect.append({
        "query_id": qid,
        "query_text": row["query_text"],
        "tenant_id": row["tenant_id"],
        "equipment": row["equipment"],
        "asset_id": row["asset_id"],
        "category": row["category"],
        "expected_positive_chunk_ids": row["positive_chunk_ids"],
        "op20_top_10": observed["top_10"],
        "op20_positive_rank": observed["positive_rank"],
        "corpus_coverage": bool(row["positive_in_eligible_corpus"]),
        "authorization_and_applicability_survival": bool(row["positive_eligible_authorized"]),
        "authorized_candidate_count": row["eligible_authorized_candidate_count"],
        "fair_dense_top_10": row["fair_op20_bge_ft_top_10"],
        "fair_dense_positive_rank": rank("fair_op20_bge_ft"),
        "sparse_top_10": row["lexical_top_10"],
        "sparse_positive_rank": rank("lexical"),
        "hybrid_candidate_union_top_50": bool(positives & (set(row["fair_op20_bge_ft"][:50]) | set(row["lexical"][:50]))),
        "fixed_rrf_top_10": row["fixed_rrf_top_10"],
        "fixed_rrf_positive_rank": rank("fixed_rrf"),
        "op20_rank_fusion": "not_used_pure_dense",
        "op20_reranker": "not_used",
        "offline_cross_encoder_positive_rank_by_depth": {
            str(depth): next(
                (i + 1 for i, cid in enumerate(reranked[qid][f"reranked_{depth}"]) if cid in positives), None
            ) for depth in (8, 10, 20, 50)
        } if qid in reranked else None,
        "mechanism": row["failure_mechanism"],
        "labeled_passage_support_concern": unsupported_label_ids.get(qid),
        "plausible_equivalent_top_passage": possible_equivalent_ids.get(qid),
        "review_status": "research_diagnostic_not_human_adjudicated",
    })

with (OUT / "incorrect_query_diagnostics.jsonl").open("w", encoding="utf-8") as output:
    for row in incorrect:
        output.write(json.dumps(row) + "\n")
print(f"Wrote {len(incorrect)} original Operation 20 Hit@1 errors; {len(unsupported_label_ids)} label concerns and {len(possible_equivalent_ids)} plausible alternatives.")
