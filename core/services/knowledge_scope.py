from django.db.models import Q
from core.models import KnowledgeDocument


def get_applicable_knowledge_queryset(asset, ctx):
    """
    Return KnowledgeDocument queryset applicable to an asset obeying hierarchy:
    Exact Asset -> Product -> Category -> Domain -> Fleet General.

    Enforces strict tenant scoping, customer isolation, and confidentiality rules:
    - Tenant isolation: document.tenant == ctx["tenant"]
    - Confidentiality: if role is 'customer' or ctx['can_view_confidential'] is False, is_confidential=False
    - Customer isolation: if role is 'customer', document.customer is NULL or document.customer == ctx['customer']
    """
    tenant = ctx["tenant"]
    role = ctx["role"]
    customer = ctx.get("customer")
    can_view_confidential = ctx.get("can_view_confidential", False)

    qs = KnowledgeDocument.objects.filter(tenant=tenant).select_related(
        "customer", "product", "asset", "category", "domain"
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
        product = asset.product
        category = product.category if product else None
        domain = category.domain if category else None

        scope = Q(asset=asset)
        if product:
            scope |= Q(asset__isnull=True, product=product)
        if category:
            scope |= Q(asset__isnull=True, product__isnull=True, category=category)
        if domain:
            scope |= Q(
                asset__isnull=True,
                product__isnull=True,
                category__isnull=True,
                domain=domain,
            )
        # Fleet general: unscoped to any asset/product/category/domain
        scope |= Q(
            asset__isnull=True,
            product__isnull=True,
            category__isnull=True,
            domain__isnull=True,
        )
        qs = qs.filter(scope)
    else:
        # Fleet general only
        qs = qs.filter(
            asset__isnull=True,
            product__isnull=True,
            category__isnull=True,
            domain__isnull=True,
        )

    return qs.order_by("-updated_at", "title")
