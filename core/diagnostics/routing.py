"""
Deterministic Playbook Routing and Selection based on Asset Hierarchy, Complaint, and Hybrid RAG.

Non-negotiable Invariants:
  - Hierarchy Applicability: Product > Category > Domain > Tenant-General.
  - Positive match required: No candidate selected on 0 relevance or without positive evidence.
  - No single-candidate bypass: Sole candidate must still meet the positive match threshold.
  - Ambiguity margin: If two top candidates are nearly tied, reject automatic selection.
  - Hybrid RAG is a ranking/evidence signal only; RAG never creates authority or widens scope.
"""

from typing import List, Optional, Tuple
from core.models import Asset, DiagnosticPlaybook
from core.services.identifiers import extract_error_codes, extract_identifiers

MIN_SCORE_THRESHOLD = 1.0
AMBIGUITY_MARGIN = 0.5


def diagnostic_routing_evidence(tenant, complaint: str, asset: Optional[Asset] = None, customer=None) -> List[str]:
    """Safely retrieve authorized knowledge titles and headings to assist candidate ranking."""
    try:
        from core.services.retriever import retrieve
        results = retrieve(
            tenant=tenant,
            question=complaint,
            asset=asset,
            customer=customer,
            include_confidential=False,
            top_k=5,
        )
        evidence_terms = []
        for r in results:
            if hasattr(r, "title") and r.title:
                evidence_terms.append(r.title.lower())
            if hasattr(r, "heading") and r.heading:
                evidence_terms.append(r.heading.lower())
        return evidence_terms
    except Exception:
        # Degraded mode: fallback gracefully to deterministic tags/identifiers only
        return []


def select_playbook_for_asset(
    asset: Asset,
    complaint: str,
    tenant,
    customer=None
) -> Optional[DiagnosticPlaybook]:
    """Select the most specific, confidently matched published diagnostic playbook for an asset and complaint."""
    base_qs = DiagnosticPlaybook.objects.filter(tenant=tenant, status="PUBLISHED")
    complaint_clean = (complaint or "").strip()
    if not complaint_clean:
        return None

    product = asset.product
    category = product.category if product else None
    domain = category.domain if category else None

    # Retrieve authorized RAG evidence for ranking assist
    rag_terms = diagnostic_routing_evidence(tenant, complaint_clean, asset=asset, customer=customer)

    # Step 1: Product-specific playbooks
    if product:
        prod_qs = base_qs.filter(product=product)
        match = _score_and_select_candidates(prod_qs, complaint_clean, rag_terms, hierarchy_weight=4.0)
        if match:
            return match

    # Step 2: Category-specific playbooks
    if category:
        cat_qs = base_qs.filter(category=category, product__isnull=True)
        match = _score_and_select_candidates(cat_qs, complaint_clean, rag_terms, hierarchy_weight=3.0)
        if match:
            return match

    # Step 3: Domain-specific playbooks
    if domain:
        dom_qs = base_qs.filter(domain=domain, category__isnull=True, product__isnull=True)
        match = _score_and_select_candidates(dom_qs, complaint_clean, rag_terms, hierarchy_weight=2.0)
        if match:
            return match

    # Step 4: Tenant-general playbooks
    gen_qs = base_qs.filter(product__isnull=True, category__isnull=True, domain__isnull=True)
    return _score_and_select_candidates(gen_qs, complaint_clean, rag_terms, hierarchy_weight=1.0)


def _score_and_select_candidates(
    qs,
    complaint: str,
    rag_terms: List[str],
    hierarchy_weight: float = 1.0
) -> Optional[DiagnosticPlaybook]:
    """Score candidate playbooks with threshold, ambiguity margin, and RAG assist."""
    playbooks = list(qs)
    if not playbooks:
        return None

    complaint_lower = complaint.lower()
    complaint_words = set(complaint_lower.split())
    error_codes = set(extract_error_codes(complaint)) | set(extract_identifiers(complaint))
    error_codes_lower = [str(e).lower() for e in error_codes]

    scored: List[Tuple[float, DiagnosticPlaybook]] = []
    for pb in playbooks:
        score = 0.0
        tags = [t.strip().lower() for t in pb.applicability_tags.split(",") if t.strip()]

        # 1. Exact tag phrase match in complaint
        for tag in tags:
            if tag in complaint_lower:
                score += 3.0
            elif any(w in tag for w in complaint_words if len(w) > 2):
                score += 1.5

        # 2. Technical identifiers / Error codes match in tags or name
        for code in error_codes_lower:
            if any(code in tag for tag in tags) or code in pb.name.lower():
                score += 4.0

        # 3. Hybrid RAG retrieved evidence alignment (document title / section match in definition)
        if rag_terms and pb.definition:
            def_str = str(pb.definition).lower()
            for term in rag_terms:
                if term in def_str or any(term in tag for tag in tags):
                    score += 1.0

        scored.append((score, pb))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    top_score, best_playbook = scored[0]

    # CRITICAL FIX 2: Require positive score above threshold (no single-candidate bypass)
    if top_score < MIN_SCORE_THRESHOLD:
        return None

    # CRITICAL FIX 11: Ambiguity margin check if multiple candidates exist
    if len(scored) > 1:
        second_score = scored[1][0]
        if (top_score - second_score) < AMBIGUITY_MARGIN:
            # Ambiguous near-tie: fail safe and return None to trigger clarification or escalation
            return None

    return best_playbook
