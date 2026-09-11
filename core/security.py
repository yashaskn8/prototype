from functools import wraps
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import redirect
from django.core.cache import cache
from django.conf import settings
import time

from .models import Tenant, TenantMembership

STAFF_ROLES = {"admin", "manager", "technician", "store_admin", "store_operator"}


def get_memberships(request):
    if not request.user.is_authenticated:
        return TenantMembership.objects.none()
    qs = TenantMembership.objects.filter(user=request.user, is_active=True, tenant__is_active=True).select_related("tenant", "customer")
    if request.user.is_superuser:
        # Superusers still need explicit demo memberships for tenant context; this
        # prevents an accidental global tenant selector in normal screens.
        return qs
    return qs


def get_current_membership(request):
    memberships = get_memberships(request)
    tenant_id = request.session.get("tenant_id")
    membership = memberships.filter(tenant_id=tenant_id).first() if tenant_id else None
    if membership is None:
        membership = memberships.order_by("tenant_id").first()
        if membership:
            request.session["tenant_id"] = membership.tenant_id
    return membership


def require_membership(request):
    membership = get_current_membership(request)
    if membership is None:
        raise PermissionDenied("Your account is not assigned to an active Servy tenant.")
    return membership


def roles_required(*roles):
    allowed = set(roles)
    def decorator(view_func):
        @login_required
        @wraps(view_func)
        def wrapped(request, *args, **kwargs):
            membership = require_membership(request)
            if membership.role not in allowed:
                raise PermissionDenied("You do not have permission to access this module.")
            request.servy_membership = membership
            request.servy_tenant = membership.tenant
            return view_func(request, *args, **kwargs)
        return wrapped
    return decorator


def any_member(view_func):
    @login_required
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        membership = require_membership(request)
        request.servy_membership = membership
        request.servy_tenant = membership.tenant
        return view_func(request, *args, **kwargs)
    return wrapped


def rate_limit_exceeded(request, bucket="rag", limit=None):
    """Simple per-user, per-tenant minute bucket using Django cache.

    This is intentionally dependency-free for the local prototype. Production
    deployments should use a shared cache/reverse-proxy limiter.
    """
    if not request.user.is_authenticated:
        return True
    membership = get_current_membership(request)
    tenant_id = membership.tenant_id if membership else 0
    limit = int(limit or getattr(settings, "SERVY_RAG_RATE_LIMIT_PER_MINUTE", 20))
    minute = int(time.time() // 60)
    key = f"servy:rl:{bucket}:{tenant_id}:{request.user.id}:{minute}"
    if cache.add(key, 1, timeout=70):
        return False
    try:
        count = cache.incr(key)
    except ValueError:
        cache.set(key, 1, timeout=70)
        count = 1
    return count > limit
