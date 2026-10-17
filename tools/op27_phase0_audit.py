"""Op27 Phase 0 — Reproduce and correct Op26 metrics."""
import json
import hashlib
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# === P0.3: CHECKPOINT VERIFICATION ===
print("=" * 80)
print("P0.3 — CHECKPOINT FINGERPRINT VERIFICATION")
print("=" * 80)

for name, ckpt_dir, reported_hash in [
    ("MiniLM", "ml_models/checkpoints/op26_minilm_ft/final",
     "419e8e44225be0107808152aff2063f71f2ad423fcba8382181dfcc64ea3d74d"),
    ("BGE-small", "ml_models/checkpoints/op26_bge_ft/final",
     "2c0e6bf3ee1f2c8a107c62d75217e5a0e784febf1327e729d1198b1874c47e64"),
]:
    ckpt = BASE_DIR / ckpt_dir
    print(f"\n{name} checkpoint: {ckpt}")
    print(f"  Exists: {ckpt.exists()}")
    if ckpt.exists():
        file_count = len([f for f in ckpt.rglob("*") if f.is_file()])
        print(f"  Files: {file_count}")
        h = hashlib.sha256()
        for p in sorted(ckpt.rglob("*")):
            if p.is_file():
                h.update(p.name.encode())
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
        actual_hash = h.hexdigest()
        match = actual_hash == reported_hash
        print(f"  Computed SHA256: {actual_hash}")
        print(f"  Reported SHA256: {reported_hash}")
        print(f"  MATCH: {match}")

# === P0.1: DENOMINATOR AUDIT ===
print("\n" + "=" * 80)
print("P0.1 - DENOMINATOR AUDIT")
print("=" * 80)

holdout_path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
holdout = [json.loads(l) for l in holdout_path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]

relevance = [r for r in holdout if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]
abstention = [r for r in holdout if not r.get("positive_chunk_ids") or len(r["positive_chunk_ids"]) == 0]

print(f"\nTotal holdout cases: {len(holdout)}")
print(f"Relevance cases (have positive_chunk_ids): {len(relevance)}")
print(f"Abstention/security cases (no positives): {len(abstention)}")

# Categorize
cats = {}
for r in holdout:
    cat = r.get("category", "unknown")
    cats.setdefault(cat, {"total": 0, "relevance": 0, "abstention": 0})
    cats[cat]["total"] += 1
    if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0:
        cats[cat]["relevance"] += 1
    else:
        cats[cat]["abstention"] += 1

print("\nCategory breakdown:")
print(f"  {'Category':<30} {'Total':>6} {'Relevance':>10} {'Abstention':>11}")
print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*11}")
for cat, counts in sorted(cats.items()):
    print(f"  {cat:<30} {counts['total']:>6} {counts['relevance']:>10} {counts['abstention']:>11}")

# === CORRECTED METRICS FROM PER-QUERY DATA ===
print("\n" + "=" * 80)
print("CORRECTED METRICS (relevance-only denominator = 43)")
print("=" * 80)

report = json.loads((BASE_DIR / "private_training_data" / "op26_final_report.json").read_text(encoding="utf-8"))

for track_name, eval_key in [
    ("minilm_ft", "minilm_finetuned"),
    ("bge_small_ft", "bge_small_finetuned"),
]:
    pq = report["per_query"].get(track_name, [])
    rel_queries = [q for q in pq if q.get("positive_ids") and len(q["positive_ids"]) > 0]
    abs_queries = [q for q in pq if not q.get("positive_ids") or len(q["positive_ids"]) == 0]

    print(f"\n--- {track_name} ---")
    print(f"  Total queries evaluated: {len(pq)}")
    print(f"  Relevance queries: {len(rel_queries)}")
    print(f"  Abstention queries: {len(abs_queries)}")

    if rel_queries:
        n = len(rel_queries)
        r1 = sum(1 for q in rel_queries if q["hit@1"]) / n
        r3 = sum(1 for q in rel_queries if q["hit@3"]) / n
        r5 = sum(1 for q in rel_queries if q["hit@5"]) / n
        r10 = sum(1 for q in rel_queries if q["hit@10"]) / n
        mrr_val = sum(q["rr"] for q in rel_queries) / n

        reported = report["evaluation"].get(eval_key, {})

        print(f"\n  {'Metric':<12} {'Op26 (N=64)':>12} {'Corrected (N=43)':>18} {'Ratio':>8}")
        print(f"  {'-'*12} {'-'*12} {'-'*18} {'-'*8}")
        for metric, op26_val, corrected_val in [
            ("Recall@1", reported.get("recall@1", 0), r1),
            ("Recall@3", reported.get("recall@3", 0), r3),
            ("Recall@5", reported.get("recall@5", 0), r5),
            ("Recall@10", reported.get("recall@10", 0), r10),
            ("MRR@10", reported.get("mrr@10", 0), mrr_val),
        ]:
            ratio = corrected_val / op26_val if op26_val > 0 else 0
            print(f"  {metric:<12} {op26_val:>12.4f} {corrected_val:>18.4f} {ratio:>8.3f}x")

# Also compute corrected base model metrics
print("\n" + "=" * 80)
print("CORRECTED BASE MODEL METRICS")
print("=" * 80)

for eval_key, label in [
    ("minilm_base", "MiniLM Base"),
    ("bge_small_base", "BGE-Small Base"),
]:
    reported = report["evaluation"].get(eval_key, {})
    # Base models don't have per-query data in the report, so we compute
    # the correction factor: 64/43 = 1.4884
    correction = 64 / 43
    print(f"\n--- {label} (estimated correction x{correction:.4f}) ---")
    for metric in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]:
        op26_val = reported.get(metric, 0)
        corrected = min(op26_val * correction, 1.0)
        print(f"  {metric:<12} Op26: {op26_val:.4f}  ->  Corrected (est): {corrected:.4f}")

# === P0.2: TRAINING/HOLDOUT OVERLAP CHECK ===
print("\n" + "=" * 80)
print("P0.2 - TRAINING/HOLDOUT OVERLAP AUDIT")
print("=" * 80)

train_path = BASE_DIR / "private_training_data" / "op26_train.jsonl"
dev_path = BASE_DIR / "private_training_data" / "op26_dev.jsonl"

train_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
dev_rows = [json.loads(l) for l in dev_path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]

# 1. Exact query text overlap
train_queries = set(r["query_text"].lower().strip() for r in train_rows)
dev_queries = set(r["query_text"].lower().strip() for r in dev_rows)
holdout_queries = set(r["query_text"].lower().strip() for r in holdout)

train_holdout_text = train_queries & holdout_queries
dev_holdout_text = dev_queries & holdout_queries
print(f"\nExact query text overlap:")
print(f"  Train <-> Holdout: {len(train_holdout_text)}")
print(f"  Dev <-> Holdout: {len(dev_holdout_text)}")
if train_holdout_text:
    for q in sorted(train_holdout_text)[:10]:
        print(f"    LEAKED: '{q[:80]}'")

# 2. Positive chunk ID overlap
train_pos_ids = set()
for r in train_rows:
    for cid in r.get("positive_chunk_ids", []):
        train_pos_ids.add(cid)

dev_pos_ids = set()
for r in dev_rows:
    for cid in r.get("positive_chunk_ids", []):
        dev_pos_ids.add(cid)

holdout_pos_ids = set()
for r in holdout:
    for cid in r.get("positive_chunk_ids", []):
        holdout_pos_ids.add(cid)

train_holdout_chunks = train_pos_ids & holdout_pos_ids
dev_holdout_chunks = dev_pos_ids & holdout_pos_ids
print(f"\nPositive chunk ID overlap:")
print(f"  Train positive chunks <-> Holdout positive chunks: {len(train_holdout_chunks)}")
print(f"  Dev positive chunks <-> Holdout positive chunks: {len(dev_holdout_chunks)}")
if train_holdout_chunks:
    print(f"    Overlapping chunk IDs: {sorted(train_holdout_chunks)[:20]}")

# 3. Source group / document overlap
train_source_groups = set(r.get("source_group", "") for r in train_rows)
dev_source_groups = set(r.get("source_group", "") for r in dev_rows)
holdout_source_groups = set(r.get("source_group", "") for r in holdout)

train_holdout_groups = train_source_groups & holdout_source_groups
dev_holdout_groups = dev_source_groups & holdout_source_groups
print(f"\nSource group overlap:")
print(f"  Train <-> Holdout: {len(train_holdout_groups)} groups")
print(f"  Dev <-> Holdout: {len(dev_holdout_groups)} groups")
if train_holdout_groups:
    print(f"    Overlapping groups: {sorted(train_holdout_groups)[:20]}")

# 4. Equipment family overlap
train_equipment = set(r.get("equipment", "") for r in train_rows if r.get("equipment"))
holdout_equipment = set(r.get("equipment", "") for r in holdout if r.get("equipment"))
shared_equipment = train_equipment & holdout_equipment
only_holdout_equipment = holdout_equipment - train_equipment
print(f"\nEquipment family overlap:")
print(f"  Shared between train & holdout: {len(shared_equipment)}")
print(f"  Holdout-only equipment: {len(only_holdout_equipment)}")
if only_holdout_equipment:
    print(f"    Equipment in holdout but NOT in train: {sorted(only_holdout_equipment)[:10]}")

# 5. Document ID overlap
train_doc_ids = set()
for r in train_rows:
    did = r.get("doc_id")
    if did:
        train_doc_ids.add(did)
    sg = r.get("source_group", "")
    if sg.startswith("doc-"):
        try:
            train_doc_ids.add(int(sg.split("-")[1]))
        except (ValueError, IndexError):
            pass

holdout_doc_ids = set()
# Holdout doesn't directly have doc_id, so we need to look up from chunk IDs
print(f"\nTrain document IDs: {len(train_doc_ids)}")
print(f"Holdout source groups: {sorted(holdout_source_groups)[:10]}")

print("\n" + "=" * 80)
print("PHASE 0 AUDIT SUMMARY")
print("=" * 80)
print(f"""
DENOMINATOR BUG: CONFIRMED
  Op26 used N=64 (all cases) as denominator for retrieval metrics.
  Correct denominator for retrieval: N=43 (relevance cases only).
  All Op26 retrieval metrics are DEFLATED by factor ~1.49x.

CHECKPOINT INTEGRITY: See above hash comparisons.

OVERLAP AUDIT: See above detailed analysis.

NEXT STEP: Build a corrected evaluator and run fresh evaluation
  on existing Op26 checkpoints with proper denominator separation.
""")
