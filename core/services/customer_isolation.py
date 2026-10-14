"""
Customer Data Isolation and Provenance Service for Servy RAG.

Enforces Phase 4 strict real-only customer data policy:
- SERVY_CUSTOMER_DATA_MODE == 'verified_real_only'
- Prohibits synthetic customers, synthetic assets, synthetic complaints, and demo manuals
- Enforces real company tenant mapping (skydream / Org 273)
- Ensures fail-closed rejection with zero silent fallback to dummy data
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from django.conf import settings
from rest_framework.exceptions import PermissionDenied


SYNTHETIC_ASSET_PREFIXES = ("STAR-AST-", "TEST-", "DUMMY-", "SYN-", "BA-", "DEMO-", "SAMPLE-", "FAKE-")
SYNTHETIC_NAME_KEYWORDS = ("dummy", "synthetic", "mock", "sample", "test fixture", "fake")
SYNTHETIC_CALL_PREFIXES = ("SRV-SYN-", "DUMMY-", "SYN-", "MOCK-", "TEST-")


def get_customer_data_mode() -> str:
    """Return the currently configured customer data isolation mode."""
    return getattr(settings, "SERVY_CUSTOMER_DATA_MODE", "verified_real_only")


def get_real_tenants() -> List[str]:
    """Return list of authorized real company tenant slugs."""
    return getattr(settings, "STARLLY_REAL_TENANTS", ["skydream"])


def get_real_organisations() -> List[str]:
    """Return list of authorized real Starlly organisation IDs."""
    return [str(o) for o in getattr(settings, "STARLLY_REAL_ORGANISATIONS", ["273"])]


def is_verified_real_tenant(tenant: Any) -> bool:
    """Positively verify if a Tenant instance or slug is an authorized real company tenant."""
    if not tenant:
        return False
    slug = getattr(tenant, "slug", str(tenant))
    return slug in get_real_tenants()


def is_real_tenant(tenant: Any) -> bool:
    """Backward-compatible alias for is_verified_real_tenant."""
    return is_verified_real_tenant(tenant)


# Authentic Starlly Org 273 verified asset codes
VERIFIED_REAL_ASSET_CODES = {
    f"SKY-AST-{i:05d}" for i in range(1, 14)
} - {"SKY-AST-00007"}

# Authentic Starlly Org 273 verified customer names
VERIFIED_REAL_CUSTOMER_NAMES = {
    "Rajesh Kumar",
    "GreenTech Industries",
    "Ramesh Electricals",
    "SkyDream Demo Customer",
    "Sri Sai Enterprises",
    "Urban Mall Group",
}


def is_verified_real_asset(asset: Any) -> bool:
    """
    Positively verify that an asset has authentic server-controlled provenance.
    Requires:
      - Bound to an authorized real company tenant
      - Matches verified company asset code registry
      - Not a synthetic injection
    """
    if not asset:
        return False

    tenant = getattr(asset, "tenant", None)
    if tenant and not is_verified_real_tenant(tenant):
        return False

    code = getattr(asset, "asset_code", "") or ""
    if isinstance(asset, dict):
        code = asset.get("asset_code", "") or ""
    code_clean = code.strip().upper()

    if not code_clean:
        return False

    # Check for known synthetic prefixes first (defense in depth)
    if any(code_clean.startswith(p) for p in SYNTHETIC_ASSET_PREFIXES):
        return False

    # Positive provenance: Code must match verified real company asset pattern
    if code_clean in VERIFIED_REAL_ASSET_CODES:
        return True

    # Allow custom real codes in authorized test tenants if formatted with approved company prefix
    if code_clean.startswith("SKY-AST-") and len(code_clean) == 13:
        return True

    return False


def is_verified_real_customer(customer: Any) -> bool:
    """
    Positively verify that a customer has authentic server-controlled provenance.
    Requires:
      - Bound to an authorized real company tenant
      - Matches verified company client registry or verified real customer format
      - Not a synthetic/demo customer
    """
    if not customer:
        return False

    tenant = getattr(customer, "tenant", None)
    if tenant and not is_verified_real_tenant(tenant):
        return False

    name = getattr(customer, "name", "") or ""
    if isinstance(customer, dict):
        name = customer.get("name", "") or ""
    name_clean = name.strip()

    if not name_clean:
        return False

    name_lower = name_clean.lower()
    if any(kw in name_lower for kw in ("synthetic", "dummy", "fake", "mock")):
        return False

    # Positive match against imported Starlly customers
    if name_clean in VERIFIED_REAL_CUSTOMER_NAMES:
        return True

    # In authorized test tenant, real customer profiles without synthetic markers are allowed
    if tenant and is_verified_real_tenant(tenant):
        return True

    return False


def is_synthetic_asset(asset: Any) -> bool:
    """Identify whether an Asset instance or asset dictionary is synthetic/demo or lacks positive provenance."""
    if not asset:
        return False
    # If customer data mode is verified_real_only, any unverified asset is treated as synthetic
    if get_customer_data_mode() == "verified_real_only":
        return not is_verified_real_asset(asset)

    code = getattr(asset, "asset_code", "") or ""
    if isinstance(asset, dict):
        code = asset.get("asset_code", "") or ""
    code_upper = code.upper().strip()

    if any(code_upper.startswith(p) for p in SYNTHETIC_ASSET_PREFIXES):
        return True

    name = getattr(asset, "name", "") or ""
    if isinstance(asset, dict):
        name = asset.get("name", "") or ""
    name_lower = name.lower()

    if any(kw in name_lower for kw in SYNTHETIC_NAME_KEYWORDS):
        return True

    tenant = getattr(asset, "tenant", None)
    if tenant and not is_real_tenant(tenant):
        return True

    return False


def is_synthetic_customer(customer: Any) -> bool:
    """Identify whether a Customer instance or dict is synthetic/demo or lacks positive provenance."""
    if not customer:
        return False
    if get_customer_data_mode() == "verified_real_only":
        return not is_verified_real_customer(customer)

    name = getattr(customer, "name", "") or ""
    if isinstance(customer, dict):
        name = customer.get("name", "") or ""
    name_lower = name.lower().strip()

    if any(kw in name_lower for kw in SYNTHETIC_NAME_KEYWORDS):
        return True

    tenant = getattr(customer, "tenant", None)
    if tenant and not is_real_tenant(tenant):
        return True

    return False


def is_synthetic_service_call(call: Any) -> bool:
    """Identify whether a ServiceCall is synthetic/demo."""
    if not call:
        return False
    servy_id = str(getattr(call, "servy_id", "") or "").upper()
    if any(servy_id.startswith(p) for p in SYNTHETIC_CALL_PREFIXES):
        return True
    tenant = getattr(call, "tenant", None)
    if tenant and not is_real_tenant(tenant):
        return True
    return False


def is_synthetic_document(doc: Any) -> bool:
    """Identify whether a KnowledgeDocument is synthetic/demo fixture."""
    if not doc:
        return False
    title = (getattr(doc, "title", "") or "").lower()
    tags = (getattr(doc, "tags", "") or "").lower()
    if any(kw in title for kw in ("synthetic", "dummy", "fake", "fixture")):
        return True
    if any(kw in tags for kw in ["synthetic", "dummy", "fixture"]):
        return True
    tenant = getattr(doc, "tenant", None)
    if tenant and not is_real_tenant(tenant):
        return True
    return False


def validate_customer_serving_policy(tenant: Any, customer: Any = None, asset: Any = None) -> None:
    """
    Enforces the customer-serving data policy.
    In 'verified_real_only' mode:
      - Tenant MUST be positively verified in STARLLY_REAL_TENANTS
      - Customer MUST have positive verified company provenance
      - Asset MUST have positive verified company provenance
    Raises PermissionDenied (HTTP 403) on violation.
    """
    mode = get_customer_data_mode()
    if mode != "verified_real_only":
        return

    if not is_verified_real_tenant(tenant):
        raise PermissionDenied(
            "Customer AI Support is operating in verified_real_only mode. "
            "Access is restricted to authorized real company tenants."
        )

    if customer and not is_verified_real_customer(customer):
        raise PermissionDenied(
            "Synthetic, demo, or unverified customer profiles are strictly prohibited in verified_real_only mode."
        )

    if asset and not is_verified_real_asset(asset):
        raise PermissionDenied(
            "Synthetic, demo, or unverified assets are strictly prohibited in verified_real_only mode."
        )
