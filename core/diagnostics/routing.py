"""
Deterministic Playbook Routing and Selection based on Asset Hierarchy and Complaint.

Order of Preference:
  1. Product-specific published playbook
  2. Category-specific published playbook
  3. Domain-specific published playbook
  4. Tenant-general published playbook
"""

from typing import Optional
from django.db.models import Q
from core.models import Asset, DiagnosticPlaybook


def select_playbook_for_asset(
    asset: Asset,
    complaint: str,
    tenant
) -> Optional[DiagnosticPlaybook]:
    """Select the most specific published diagnostic playbook for an asset and complaint."""
    base_qs = DiagnosticPlaybook.objects.filter(tenant=tenant, status="PUBLISHED")

    product = asset.product
    category = product.category if product else None
    domain = category.domain if category else None

    # Step 1: Product-specific playbooks
    if product:
        prod_qs = base_qs.filter(product=product)
        match = _find_best_tag_match(prod_qs, complaint)
        if match:
            return match

    # Step 2: Category-specific playbooks
    if category:
        cat_qs = base_qs.filter(category=category, product__isnull=True)
        match = _find_best_tag_match(cat_qs, complaint)
        if match:
            return match

    # Step 3: Domain-specific playbooks
    if domain:
        dom_qs = base_qs.filter(domain=domain, category__isnull=True, product__isnull=True)
        match = _find_best_tag_match(dom_qs, complaint)
        if match:
            return match

    # Step 4: Tenant-general playbooks
    gen_qs = base_qs.filter(product__isnull=True, category__isnull=True, domain__isnull=True)
    return _find_best_tag_match(gen_qs, complaint)


def _find_best_tag_match(qs, complaint: str) -> Optional[DiagnosticPlaybook]:
    """Score matching playbooks by applicability tags matching complaint words."""
    playbooks = list(qs)
    if not playbooks:
        return None

    if len(playbooks) == 1:
        return playbooks[0]

    complaint_words = set((complaint or "").lower().split())
    scored = []
    for pb in playbooks:
        score = 0
        tags = [t.strip().lower() for t in pb.applicability_tags.split(",") if t.strip()]
        for tag in tags:
            if tag in complaint.lower():
                score += 2
            elif any(w in tag for w in complaint_words):
                score += 1
        scored.append((score, pb))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else playbooks[0]
