"""
Canonical Knowledge Scope Policy for Servy (Operation 12.0).

Unified Equipment Hierarchy:
Level 1: Exact Asset
Level 2: Product
Level 3: Category
Level 4: Domain
Level 5: Fleet General (unscoped to any asset, product, category, or domain)

Enforces strict tenant scoping, customer isolation, confidentiality, and
fail-closed applicability across:
- Knowledge search
- Dense candidate filtering
- Sparse candidate filtering
- Customer evidence retrieval
- Source grounding
- Diagnostic playbooks
- Citation validation
- Evaluation
- Answer generation
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from django.db.models import Q, QuerySet
from core.models import KnowledgeDocument, KnowledgeChunk


def get_applicable_knowledge_queryset(asset: Optional[Any], ctx: Dict[str, Any]) -> QuerySet:
    """Return KnowledgeDocument queryset applicable to an asset obeying canonical 5-level hierarchy.

    Tenant isolation: document.tenant == ctx["tenant"]
    Confidentiality: if role is 'customer' or ctx['can_view_confidential'] is False, is_confidential=False
    Customer isolation: if role is 'customer', document.customer is NULL or document.customer == ctx['customer']
    """
    tenant = ctx["tenant"]
    role = ctx.get("role", "customer")
    customer = ctx.get("customer")
    can_view_confidential = ctx.get("can_view_confidential", False)

    qs = KnowledgeDocument.objects.filter(tenant=tenant).select_related(
        "customer", "product", "asset", "category", "domain",
        "product__category", "product__category__domain"
    )

    if role == "customer":
        qs = qs.filter(is_confidential=False)
        if customer:
            qs = qs.filter(Q(customer__isnull=True) | Q(customer=customer))
        else:
            return KnowledgeDocument.objects.none()
    elif not can_view_confidential and role != "superuser":
        qs = qs.filter(is_confidential=False)

    if asset:
        product = getattr(asset, "product", None)
        category = getattr(product, "category", None) if product else None
        domain = getattr(category, "domain", None) if category else None

        scope = Q(asset=asset)
        if product:
            scope |= Q(asset__isnull=True, product=product)
        if category:
            scope |= Q(asset__isnull=True, product__isnull=True, category=category)
        if domain:
            scope |= Q(
                asset__isnull=True,
                product__isnull=True,
            ) & (Q(domain=domain) | Q(category__domain=domain))
        # Fleet general: unscoped to any asset/product/category/domain
        scope |= Q(
            asset__isnull=True,
            product__isnull=True,
            category__isnull=True,
            domain__isnull=True,
        )
        qs = qs.filter(scope)
    else:
        # Fleet general only: unscoped query may NEVER match equipment-specific documentation
        qs = qs.filter(
            asset__isnull=True,
            product__isnull=True,
            category__isnull=True,
            domain__isnull=True,
        )

    return qs.order_by("-updated_at", "title")


def get_applicable_chunks_queryset(asset: Optional[Any], ctx: Dict[str, Any]) -> QuerySet:
    """Return KnowledgeChunk queryset obeying canonical 5-level hierarchy and tenant/customer isolation."""
    tenant = ctx["tenant"]
    role = ctx.get("role", "customer")
    customer = ctx.get("customer")
    can_view_confidential = ctx.get("can_view_confidential", False)

    qs = KnowledgeChunk.objects.filter(
        tenant=tenant,
        document__is_rag_enabled=True,
        is_quarantined=False,
    ).select_related(
        "document", "document__product", "document__asset",
        "document__category", "document__domain", "document__customer",
        "document__product__category", "document__product__category__domain",
    )

    if role == "customer":
        qs = qs.filter(document__is_confidential=False)
        if customer:
            qs = qs.filter(Q(document__customer__isnull=True) | Q(document__customer=customer))
        else:
            return KnowledgeChunk.objects.none()
    elif not can_view_confidential and role != "superuser":
        qs = qs.filter(document__is_confidential=False)

    if asset:
        product = getattr(asset, "product", None)
        category = getattr(product, "category", None) if product else None
        domain = getattr(category, "domain", None) if category else None

        scope = Q(document__asset=asset)
        if product:
            scope |= Q(document__asset__isnull=True, document__product=product)
        if category:
            scope |= Q(document__asset__isnull=True, document__product__isnull=True, document__category=category)
        if domain:
            scope |= Q(
                document__asset__isnull=True,
                document__product__isnull=True,
            ) & (Q(document__domain=domain) | Q(document__category__domain=domain))
        scope |= Q(
            document__asset__isnull=True,
            document__product__isnull=True,
            document__category__isnull=True,
            document__domain__isnull=True,
        )
        qs = qs.filter(scope)
    else:
        scope = Q(
            document__asset__isnull=True,
            document__product__isnull=True,
            document__category__isnull=True,
            document__domain__isnull=True,
        )
        qs = qs.filter(scope)

    return qs


def is_document_applicable_to_asset(document: Optional[Any], asset: Optional[Any]) -> bool:
    """Canonical in-memory predicate to determine if a KnowledgeDocument applies to an asset.

    Exact Hierarchy:
    - Level 1: doc.asset_id == asset.id
    - Level 2: doc.asset_id is None and doc.product_id == asset.product_id
    - Level 3: doc.asset_id is None and doc.product_id is None and doc.category_id == asset.product.category_id
    - Level 4: doc.asset_id is None and doc.product_id is None and doc.domain_id == asset.product.category.domain_id
               (including documents categorized within the same domain)
    - Level 5: doc.asset_id is None and doc.product_id is None and doc.category_id is None and doc.domain_id is None (Fleet General)

    Fails closed on any mismatch, cross-equipment assignment, or foreign hierarchy level.
    """
    if not document:
        return False

    doc_asset_id = getattr(document, "asset_id", None)
    doc_product_id = getattr(document, "product_id", None)
    doc_category_id = getattr(document, "category_id", None)
    doc_domain_id = getattr(document, "domain_id", None)
    if doc_domain_id is None and hasattr(document, "category") and document.category:
        doc_domain_id = getattr(document.category, "domain_id", None)

    # If query has NO asset context: only Level 5 (Fleet General) documents are applicable
    if asset is None:
        return bool(
            doc_asset_id is None
            and doc_product_id is None
            and doc_category_id is None
            and doc_domain_id is None
        )

    # Query HAS asset context: check hierarchy levels
    asset_id = getattr(asset, "id", None)
    product = getattr(asset, "product", None)
    product_id = getattr(asset, "product_id", None) or (product.id if product else None)
    category = getattr(product, "category", None) if product else None
    category_id = getattr(product, "category_id", None) if product else (category.id if category else None)
    domain = getattr(category, "domain", None) if category else None
    domain_id = getattr(category, "domain_id", None) if category else (domain.id if domain else None)

    # Level 1: Exact Asset
    if doc_asset_id is not None:
        return bool(asset_id is not None and doc_asset_id == asset_id)

    # Level 2: Product match
    if doc_product_id is not None:
        return bool(product_id is not None and doc_product_id == product_id)

    # Level 3: Category match
    if doc_category_id is not None:
        if category_id is not None and doc_category_id == category_id:
            return True
        # Level 4 fallback: check if document belongs to the same domain
        if domain_id is not None and doc_domain_id == domain_id:
            return True
        return False

    # Level 4: Domain match
    if doc_domain_id is not None:
        return bool(domain_id is not None and doc_domain_id == domain_id)

    # Level 5: Fleet General
    return True


def is_chunk_applicable_to_asset(chunk: Optional[Any], asset: Optional[Any]) -> bool:
    """Canonical predicate for KnowledgeChunk equipment applicability."""
    if not chunk:
        return False
    doc = getattr(chunk, "document", None)
    return is_document_applicable_to_asset(doc, asset)
