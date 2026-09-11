from .security import get_current_membership, get_memberships


def tenant_context(request):
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        return {"current_tenant": None, "current_membership": None, "available_memberships": []}
    membership = get_current_membership(request)
    memberships = list(get_memberships(request))
    return {
        "current_tenant": membership.tenant if membership else None,
        "current_membership": membership,
        "available_memberships": memberships,
    }
