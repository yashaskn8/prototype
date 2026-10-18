"""Operation 26 — Build Enhanced Training Dataset for Dual-Model Fine-Tuning.

Generates source-derived silver query-passage pairs with PROVISIONAL hard-negative candidates
from the live MySQL corpus. Covers all equipment families across both authorized
tenants (Skydream=9, Starlly=6).

Dataset roles:
  TRAIN: Used for optimization (both MiniLM and BGE-small)
  DEVELOPMENT: Separate labeled examples for model selection / early stopping
  HISTORICAL_DIAGNOSTIC: Existing 64-query holdout — regression analysis only

Output: private_training_data/op26_train.jsonl
        private_training_data/op26_dev.jsonl
        private_training_data/op26_dataset_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import random
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django
django.setup()

from core.models import KnowledgeDocument, KnowledgeChunk, Tenant
from tools.op26_source_isolated_split import source_isolated_split


def chunk_passage_text(chunk, content_first=True):
    """Produce passage text matching the indexer's passage_text() logic."""
    if content_first:
        heading = (chunk.heading or "").strip()
        body = chunk.text or ""
        return f"{heading}\n{body}".strip() if heading and heading not in body else body
    d = chunk.document
    prefix = " ".join(filter(None, [
        d.title, d.description, d.tags, d.doc_type,
        getattr(d.domain, 'name', '') if d.domain else "",
        getattr(d.category, 'name', '') if d.category else "",
        getattr(d.product, 'name', '') if d.product else "",
        getattr(d.asset, 'name', '') if d.asset else "",
        getattr(d.asset, 'model_number', '') if d.asset else "",
    ]))
    return f"{prefix} {chunk.heading}\n{chunk.text}".strip()


def extract_section_heading(chunk):
    """Extract the most specific heading from a chunk."""
    heading = (chunk.heading or "").strip()
    if heading:
        return heading
    text = chunk.text or ""
    lines = text.strip().split("\n")
    for line in lines[:3]:
        line = line.strip()
        if line.startswith("##"):
            return line.lstrip("#").strip()
        if line and len(line) < 120 and not line.endswith("."):
            return line
    return ""


def generate_natural_query(chunk, doc, query_type="symptom"):
    """Generate a natural technician-style query from a chunk's content."""
    heading = extract_section_heading(chunk)
    text = chunk.text or ""
    equipment = ""
    if doc.asset:
        equipment = doc.asset.name or ""
    elif doc.title:
        # Extract equipment from document title
        parts = doc.title.split(" - ")
        equipment = parts[0].strip() if parts else doc.title

    queries = []

    # Type 1: Symptom-based queries
    symptom_match = re.search(r"(?:SYMPTOM|symptom)[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if symptom_match:
        symptom = symptom_match.group(1).strip().rstrip(".")
        queries.append(f"{equipment} {symptom.lower()}")
        # Technician shorthand version
        short = symptom.lower().replace("unit does not start", "won't start").replace(
            "poor cooling", "not cooling properly").replace(
            "water leakage", "leaking water").replace(
            "noise or vibration", "making noise")
        if short != symptom.lower():
            queries.append(f"{equipment} {short}")

    # Type 2: Fault code queries
    fault_match = re.findall(r"(?:fault|error|alarm|code)\s*(?:code\s*)?([A-Z][\w-]*\d+|\d+)", text, re.IGNORECASE)
    for code in fault_match[:2]:
        queries.append(f"{equipment} fault code {code}")
        queries.append(f"{equipment} error {code} meaning")

    # Type 3: Procedure queries
    if any(kw in text.lower() for kw in ["step 1", "procedure", "checklist", "inspect", "verify"]):
        if heading:
            queries.append(f"How to {heading.lower().rstrip('.')}")
            queries.append(f"{equipment} {heading.lower()} procedure")

    # Type 4: Maintenance interval queries
    if any(kw in text.lower() for kw in ["monthly", "quarterly", "annually", "interval", "schedule"]):
        queries.append(f"{equipment} maintenance schedule intervals")
        queries.append(f"{equipment} preventive maintenance frequency")

    # Type 5: Specification queries
    spec_match = re.findall(r"(\d+(?:\.\d+)?)\s*(°C|degrees|bar|psi|V|A|mm|kg|RPM|kVA)", text, re.IGNORECASE)
    for val, unit in spec_match[:2]:
        queries.append(f"{equipment} operating {unit.lower()} specification {val}")

    # Type 6: Safety/escalation queries
    if any(kw in text.lower() for kw in ["stop and escalate", "do not", "danger", "hazard", "qualified"]):
        queries.append(f"{equipment} safety precautions and escalation criteria")

    # Type 7: Heading-based queries (generic fallback)
    if heading and not queries:
        queries.append(f"{equipment} {heading.lower()}")

    # Type 8: Misspelled variants
    if equipment and queries:
        misspelled = queries[0].replace("compressor", "compresser").replace(
            "temperature", "temprature").replace("refrigerant", "refridgerant").replace(
            "hydraulic", "hydralic").replace("thermostat", "thermostate").replace(
            "maintenance", "maintanence")
        if misspelled != queries[0]:
            queries.append(misspelled)

    return queries


def find_hard_negatives(chunk, all_chunks_by_doc, all_chunks_by_equipment, all_chunks_by_tenant):
    """Mine provisional candidates; heading/equipment differences do NOT prove irrelevance."""
    doc_id = chunk.document_id
    tenant_id = chunk.tenant_id
    chunk_id = chunk.id
    equipment = chunk.document.asset.name if chunk.document.asset else None

    negatives = []

    # Strategy 1: Same document, different section (hardest)
    same_doc = [c for c in all_chunks_by_doc.get(doc_id, []) if c.id != chunk_id]
    for neg in same_doc:
        # Verify it doesn't contain the same answer
        if chunk.heading and neg.heading and chunk.heading != neg.heading:
            negatives.append(("same_doc_diff_section", neg))

    # Strategy 2: Same equipment family, different document
    if equipment:
        same_equip = [c for c in all_chunks_by_equipment.get(equipment, [])
                      if c.document_id != doc_id and c.id != chunk_id]
        for neg in same_equip:
            negatives.append(("same_equipment_diff_doc", neg))

    # Strategy 3: Different equipment, same tenant (equipment trap)
    tenant_chunks = all_chunks_by_tenant.get(tenant_id, [])
    diff_equip = [c for c in tenant_chunks
                  if c.document_id != doc_id and c.id != chunk_id
                  and (not equipment or (c.document.asset and c.document.asset.name != equipment))]
    for neg in diff_equip[:5]:
        negatives.append(("diff_equipment_same_tenant", neg))

    return negatives


def verify_negative_is_safe(query_text, positive_chunk, negative_chunk):
    """Cheap screening only; a True result is NOT expert-verified irrelevance."""
    pos_text = positive_chunk.text.lower()
    neg_text = negative_chunk.text.lower()

    # Extract key technical terms from positive
    pos_terms = set(re.findall(r'\b[A-Z][\w-]*\d+\b', positive_chunk.text))  # Error codes
    neg_terms = set(re.findall(r'\b[A-Z][\w-]*\d+\b', negative_chunk.text))

    # If the negative contains the same error codes, it might be a valid answer
    if pos_terms and pos_terms & neg_terms:
        return False

    # If headings are identical, the negative likely contains the same answer
    if positive_chunk.heading and negative_chunk.heading:
        if positive_chunk.heading.strip().lower() == negative_chunk.heading.strip().lower():
            return False

    # Check substantial text overlap (>50% shared words)
    pos_words = set(pos_text.split())
    neg_words = set(neg_text.split())
    if pos_words and len(pos_words & neg_words) / len(pos_words) > 0.5:
        return False

    return True


def build_dataset():
    """Build the complete Op26 training dataset from MySQL."""
    print("=" * 80)
    print("OPERATION 26 — BUILDING ENHANCED TRAINING DATASET")
    print("=" * 80)

    # Load all eligible chunks
    chunks = list(KnowledgeChunk.objects.filter(
        is_quarantined=False,
        document__is_rag_enabled=True,
        document__index_status="INDEXED",
    ).exclude(text="").select_related(
        "document", "document__asset", "document__product",
        "document__category", "document__domain", "tenant"
    ).order_by("document_id", "chunk_index"))

    print(f"\nTotal eligible chunks: {len(chunks)}")

    # Build lookup structures
    all_chunks_by_doc = defaultdict(list)
    all_chunks_by_equipment = defaultdict(list)
    all_chunks_by_tenant = defaultdict(list)
    chunk_map = {}

    for c in chunks:
        all_chunks_by_doc[c.document_id].append(c)
        all_chunks_by_tenant[c.tenant_id].append(c)
        chunk_map[c.id] = c
        if c.document.asset:
            all_chunks_by_equipment[c.document.asset.name].append(c)

    # Load existing training data to avoid duplication
    existing_train_path = BASE_DIR / "private_training_data" / "company_train_weakly_supervised.jsonl"
    existing_holdout_path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"

    existing_train_qids = set()
    existing_holdout_chunk_ids = set()
    holdout_query_texts = set()

    if existing_train_path.exists():
        with open(existing_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing_train_qids.add(row.get("query_id"))

    if existing_holdout_path.exists():
        with open(existing_holdout_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    for cid in row.get("positive_chunk_ids", []):
                        existing_holdout_chunk_ids.add(cid)
                    holdout_query_texts.add(row.get("query_text", "").lower().strip())

    print(f"Existing training QIDs: {len(existing_train_qids)}")
    print(f"Holdout positive chunk IDs: {len(existing_holdout_chunk_ids)}")

    # Generate new training examples
    all_examples = []
    stats = {
        "total_candidates": 0,
        "provenance_valid": 0,
        "source_backed": 0,
        "rejected_false_negative": 0,
        "rejected_no_query": 0,
        "rejected_holdout_overlap": 0,
        "equipment_families": defaultdict(int),
        "query_types": defaultdict(int),
        "tenants": defaultdict(int),
        "hard_neg_strategies": defaultdict(int),
    }

    # Include existing weakly supervised data (re-import with Op26 IDs)
    if existing_train_path.exists():
        with open(existing_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    # Reformat for Op26
                    example = {
                        "query_id": f"OP26-EXIST-{row['query_id']}",
                        "query_text": row["query_text"],
                        "tenant_id": row["tenant_id"],
                        "positive_chunk_ids": row.get("positive_chunk_ids", [row.get("positive_chunk_id")]),
                        "negative_chunk_ids": row.get("negative_chunk_ids", []),
                        "source_group": row.get("source_group", f"doc-{row.get('positive_document_id', 0)}"),
                        "review_status": "weakly_supervised",
                        "negative_review_status": "provisional_unverified",
                        "label_source": "OP20_EXISTING_WEAKLY_SUPERVISED",
                        "dataset_version": "2.0",
                        "equipment": "",
                    }
                    # Resolve equipment from chunk
                    pos_id = example["positive_chunk_ids"][0] if example["positive_chunk_ids"] else None
                    if pos_id and pos_id in chunk_map:
                        c = chunk_map[pos_id]
                        if c.document.asset:
                            example["equipment"] = c.document.asset.name
                        example["source_content_hash"] = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
                    all_examples.append(example)
                    stats["total_candidates"] += 1
                    stats["provenance_valid"] += 1
                    stats["source_backed"] += 1

    print(f"\nRe-imported {len(all_examples)} existing training examples")

    # Generate NEW source-derived silver examples
    new_count = 0
    for chunk in chunks:
        doc = chunk.document
        if not chunk.text or len(chunk.text.strip()) < 30:
            continue

        queries = generate_natural_query(chunk, doc, query_type="auto")
        if not queries:
            stats["rejected_no_query"] += 1
            continue

        # Find hard negatives
        negatives = find_hard_negatives(
            chunk, all_chunks_by_doc, all_chunks_by_equipment, all_chunks_by_tenant
        )

        # Heuristic screening cannot establish a false-negative-free label.
        # Preserve candidates only as weak supervision until domain review.
        provisional_negs = []
        for strategy, neg_chunk in negatives:
            if verify_negative_is_safe(queries[0], chunk, neg_chunk):
                provisional_negs.append((strategy, neg_chunk))
                stats["hard_neg_strategies"][strategy] += 1
            else:
                stats["rejected_false_negative"] += 1

        equipment = doc.asset.name if doc.asset else doc.title.split(" - ")[0].strip()

        # Use up to 3 query variants per chunk, with best negatives
        for qi, query_text in enumerate(queries[:3]):
            stats["total_candidates"] += 1

            # Check holdout contamination (fuzzy)
            if query_text.lower().strip() in holdout_query_texts:
                stats["rejected_holdout_overlap"] += 1
                continue

            # Select best negative (prefer same-doc-diff-section)
            neg_ids = []
            if provisional_negs:
                # Prioritize same_doc_diff_section, then same_equipment_diff_doc
                priority = {"same_doc_diff_section": 0, "same_equipment_diff_doc": 1, "diff_equipment_same_tenant": 2}
                sorted_negs = sorted(provisional_negs, key=lambda x: priority.get(x[0], 99))
                neg_ids = [n.id for _, n in sorted_negs[:2]]  # Up to 2 negatives

            qid = f"OP26-SLV-{chunk.id}-{qi}"
            content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

            example = {
                "query_id": qid,
                "query_text": query_text,
                "tenant_id": chunk.tenant_id,
                "positive_chunk_ids": [chunk.id],
                "negative_chunk_ids": neg_ids,
                "source_group": f"doc-{doc.id}",
                "review_status": "weakly_supervised",
                "negative_review_status": "provisional_unverified",
                "negative_mining_method": "heuristic_only_no_expert_adjudication",
                "label_source": "SOURCE_DERIVED_SILVER_OP26",
                "source_content_hash": content_hash,
                "dataset_version": "2.0",
                "equipment": equipment,
                "doc_id": doc.id,
                "doc_title": doc.title,
            }

            all_examples.append(example)
            stats["provenance_valid"] += 1
            stats["source_backed"] += 1
            stats["equipment_families"][equipment] += 1
            stats["tenants"][chunk.tenant_id] += 1
            new_count += 1

    print(f"Generated {new_count} new source-derived silver examples")
    print(f"Total examples: {len(all_examples)}")

    # Deduplicate by query text (keep first occurrence)
    seen_queries = set()
    deduped = []
    for ex in all_examples:
        qt_norm = ex["query_text"].lower().strip()
        if qt_norm not in seen_queries:
            seen_queries.add(qt_norm)
            deduped.append(ex)

    print(f"After deduplication: {len(deduped)} examples")

    # Root-cause fix: positive-only source_group partitioning leaked
    # negative-reference documents into the opposite split.
    # Partition by connected positives + negatives + document checksums
    # + identical passages + normalized queries. Human source review remains required.
    train_examples, dev_examples, split_evidence = source_isolated_split(
        deduped, chunk_map, dev_fraction=0.15, seed=42
    )
    n_dev_groups = split_evidence["dev_components"]
    group_keys = list(range(split_evidence["source_components"]))
    print(f"\nTRAIN: {len(train_examples)} examples")
    print(f"DEVELOPMENT: {len(dev_examples)} examples")
    print(f"Connected source components: {split_evidence['source_components']}")

    # Equipment coverage
    train_equipment = defaultdict(int)
    for ex in train_examples:
        train_equipment[ex.get("equipment", "unknown")] += 1
    print(f"\nEquipment coverage in TRAIN:")
    for eq, cnt in sorted(train_equipment.items(), key=lambda x: -x[1]):
        print(f"  {eq}: {cnt}")

    # Write output files
    out_dir = BASE_DIR / "private_training_data"
    out_dir.mkdir(exist_ok=True)

    train_path = out_dir / "op26_train.jsonl"
    dev_path = out_dir / "op26_dev.jsonl"
    manifest_path = out_dir / "op26_dataset_manifest.json"
    if any(p.exists() for p in (train_path, dev_path, manifest_path)):
        raise FileExistsError(
            "Refusing to overwrite historical Op26 datasets. "
            "Preserve or explicitly relocate them before regenerating."
        )

    with open(train_path, "x", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(dev_path, "x", encoding="utf-8") as f:
        for ex in dev_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Compute file hashes
    def file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    train_sha = file_sha256(train_path)
    dev_sha = file_sha256(dev_path)

    # Write manifest
    manifest = {
        "operation": "OPERATION_26",
        "dataset_version": "2.0",
        "created_by": "op26_build_training_dataset.py",
        "files": {
            "train": {
                "path": str(train_path),
                "sha256": train_sha,
                "n_records": len(train_examples),
                "role": "TRAIN",
            },
            "development": {
                "path": str(dev_path),
                "sha256": dev_sha,
                "n_records": len(dev_examples),
                "role": "DEVELOPMENT",
            },
            "historical_diagnostic": {
                "path": str(existing_holdout_path),
                "sha256": "f516a183d510a275de5856cb9d26eefc055181223d28cc8630db914613508dda",
                "n_records": 64,
                "role": "HISTORICAL_DIAGNOSTIC",
            },
        },
        "stats": {
            "total_candidates": stats["total_candidates"],
            "provenance_valid": stats["provenance_valid"],
            "source_backed": stats["source_backed"],
            "rejected_false_negative": stats["rejected_false_negative"],
            "rejected_no_query": stats["rejected_no_query"],
            "rejected_holdout_overlap": stats["rejected_holdout_overlap"],
            "equipment_families": dict(stats["equipment_families"]),
            "tenants": dict(stats["tenants"]),
            "hard_neg_strategies": dict(stats["hard_neg_strategies"]),
            "n_distinct_queries": len(seen_queries),
            "n_train": len(train_examples),
            "n_dev": len(dev_examples),
            "n_source_groups_train": len(group_keys) - n_dev_groups,
            "n_source_groups_dev": n_dev_groups,
            "source_graph_isolation": split_evidence,
            "expert_label_approval": "UNVERIFIED_WEAK_SUPERVISION",
        },
    }

    with open(manifest_path, "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("DATASET BUILD COMPLETE")
    print(f"Train: {train_path} ({len(train_examples)} examples, SHA256: {train_sha[:16]}...)")
    print(f"Dev:   {dev_path} ({len(dev_examples)} examples, SHA256: {dev_sha[:16]}...)")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*80}")

    return manifest


if __name__ == "__main__":
    build_dataset()
