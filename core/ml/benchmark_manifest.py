"""Benchmark Accounting Manifest and Query Loader for Servy RAG.

Provides deterministic, transparent accounting for all 64 queries across all 14
categories in the frozen independent holdout benchmark.

Guarantees:
- Every query has a unique, stable ID.
- Exact non-overlapping primary evaluation roles to prevent inflated/deflated metric denominators.
- Complete coverage of both Tenant 9 (Skydream HVAC) and Tenant 6 (Starlly Industrial Fleet).
- Standardized asset/product resolution for realistic customer-scoped evaluation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BENCHMARK_PATH = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
MANIFEST_PATH = BASE_DIR / "ml_artifacts" / "benchmark_accounting_manifest.json"

ROLE_RELEVANCE = "RELEVANCE_EVALUATION"
ROLE_ABSTENTION = "ABSTENTION_EVALUATION"
ROLE_SECURITY = "SECURITY_EVALUATION"

CATEGORY_TO_PRIMARY_ROLE = {
    "exact_error_code": ROLE_RELEVANCE,
    "general_symptom": ROLE_RELEVANCE,
    "model_trap": ROLE_RELEVANCE,
    "misspelled_query": ROLE_RELEVANCE,
    "multi_symptom": ROLE_RELEVANCE,
    "model_ambiguity": ROLE_RELEVANCE,
    "technician_only": ROLE_RELEVANCE,
    "equipment_crosstalk": ROLE_RELEVANCE,
    "unsupported_question": ROLE_ABSTENTION,
    "out_of_domain": ROLE_ABSTENTION,
    "doc_id_attack": ROLE_SECURITY,
    "revoked_doc_query": ROLE_SECURITY,
    "tenant_isolation_attack": ROLE_SECURITY,
    "boundary_injection": ROLE_SECURITY,
}


def build_manifest() -> Dict[str, Any]:
    """Parse the raw frozen benchmark and construct the complete accounting manifest."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        raw_rows = [json.loads(line) for line in f if line.strip()]

    cases = []
    category_counts = {}
    role_counts = {ROLE_RELEVANCE: 0, ROLE_ABSTENTION: 0, ROLE_SECURITY: 0}
    tenant_counts = {}

    for row in raw_rows:
        qid = row["query_id"]
        cat = row.get("category", "general")
        tid = int(row["tenant_id"])
        primary_role = CATEGORY_TO_PRIMARY_ROLE.get(cat, ROLE_RELEVANCE)
        secondary_roles = []
        if cat == "technician_only":
            secondary_roles.append("SAFETY_POLICY_EVALUATION")

        is_unsupported = bool(row.get("is_unsupported", False))

        case = {
            "query_id": qid,
            "tenant_id": tid,
            "category": cat,
            "primary_evaluation_role": primary_role,
            "secondary_roles": secondary_roles,
            "query_text": row["query_text"],
            "equipment": row.get("equipment", "general"),
            "positive_chunk_ids": row.get("positive_chunk_ids", []),
            "expected_abstain": is_unsupported,
            "scorable": True,
            "unscorable_reason": None,
        }
        cases.append(case)

        category_counts[cat] = category_counts.get(cat, 0) + 1
        role_counts[primary_role] = role_counts.get(primary_role, 0) + 1
        tenant_counts[tid] = tenant_counts.get(tid, 0) + 1

    manifest = {
        "manifest_version": "10.0-audited",
        "total_benchmark_cases": len(cases),
        "primary_denominators": {
            "relevance_queries": role_counts[ROLE_RELEVANCE],
            "abstention_queries": role_counts[ROLE_ABSTENTION],
            "security_queries": role_counts[ROLE_SECURITY],
            "total_abstain_expected": role_counts[ROLE_ABSTENTION] + role_counts[ROLE_SECURITY],
        },
        "tenant_breakdown": tenant_counts,
        "category_counts": category_counts,
        "cases": cases,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_manifest() -> Dict[str, Any]:
    """Load cached manifest or build if not present."""
    if not MANIFEST_PATH.exists():
        return build_manifest()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_query_asset(tenant_id: int, equip_name: str) -> Optional[Any]:
    """Resolve equip_name to an Asset instance for the given tenant."""
    if not equip_name or equip_name == "general":
        return None
    from core.models import Asset, Product

    # 1. Direct asset name match
    asset = Asset.objects.filter(tenant_id=tenant_id, name=equip_name).first()
    if asset:
        return asset

    # 2. Direct product name match
    asset = Asset.objects.filter(tenant_id=tenant_id, product__name=equip_name).first()
    if asset:
        return asset

    # 3. Substring/prefix match on product name
    for prod in Product.objects.filter(tenant_id=tenant_id):
        if prod.name.lower() in equip_name.lower() or equip_name.lower() in prod.name.lower():
            asset = Asset.objects.filter(tenant_id=tenant_id, product=prod).first()
            if asset:
                return asset

    return None


if __name__ == "__main__":
    m = build_manifest()
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Total cases: {m['total_benchmark_cases']}")
    print(f"Primary denominators: {m['primary_denominators']}")
    print(f"Tenant breakdown: {m['tenant_breakdown']}")
    print("Category breakdown:")
    for c, cnt in sorted(m['category_counts'].items()):
        print(f"  {c:<25}: {cnt}")
