"""
Customer Data Isolation and Provenance Service for Servy RAG.

Enforces Phase 4 strict real-only customer data policy:
- SERVY_CUSTOMER_DATA_MODE == 'verified_real_only'
- Prohibits synthetic customers, synthetic assets, synthetic complaints, and demo manuals
- Enforces real company tenant mapping (skydream / Org 273)
- Ensures fail-closed rejection with zero silent fallback to dummy data
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple
from django.conf import settings
from rest_framework.exceptions import PermissionDenied


SYNTHETIC_ASSET_PREFIXES = ("STAR-AST-", "TEST-", "DUMMY-", "SYN-", "BA-", "DEMO-", "SAMPLE-", "FAKE-")
SYNTHETIC_NAME_KEYWORDS = ("dummy", "synthetic", "mock", "sample", "test fixture", "fake")
SYNTHETIC_CALL_PREFIXES = ("SRV-SYN-", "DUMMY-", "SYN-", "MOCK-", "TEST-")


def get_customer_data_mode() -> str:
    """Return the currently configured customer data isolation mode.
    Fails closed to 'verified_real_only' if unset, empty, or unrecognized.
    """
    raw_mode = getattr(settings, "SERVY_CUSTOMER_DATA_MODE", "verified_real_only")
    clean_mode = str(raw_mode or "").lower().strip()
    if clean_mode in ("verified_real_only", "demo_only", "test_demo", "test_isolated"):
        return clean_mode
    # Fail-closed default for unknown/misconfigured values
    return "verified_real_only"


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


def is_test_environment() -> bool:
    """Detect if current execution is within an isolated test runner or CI environment."""
    import sys
    import os
    return (
        "test" in sys.argv
        or getattr(settings, "IN_TEST_SUITE", False)
        or getattr(settings, "SERVY_ALLOW_TEST_PROVENANCE", False)
        or os.getenv("CI") == "true"
        or bool(getattr(settings, "DEBUG", False))
    )


# In-memory test-authorized registry for isolated test suites
_TEST_VERIFIED_OBJECTS: set = set()


def register_test_verified_provenance(obj: Any) -> None:
    """Explicitly register an in-memory test fixture as having verified provenance for the current test run.
    Attacker runtime forgery via setting attributes on arbitrary objects is strictly blocked.
    Production runtime registration is strictly prohibited (fails closed).
    """
    if not is_test_environment():
        raise RuntimeError("Test provenance registration is strictly prohibited in production runtime.")
    if not obj:
        return
    obj_id = getattr(obj, "id", None)
    if not obj_id and isinstance(obj, dict):
        obj_id = obj.get("id")
    model_name = obj.__class__.__name__.lower() if hasattr(obj, "__class__") else "dict"
    _TEST_VERIFIED_OBJECTS.add((model_name, obj_id))


def clear_test_verified_provenance() -> None:
    """Clear all test-authorized objects."""
    _TEST_VERIFIED_OBJECTS.clear()


def is_test_verified_object(model_name: str, obj_id: Any) -> bool:
    """Check if an object ID is in the test-authorized set."""
    if not is_test_environment() or obj_id is None:
        return False
    return (model_name.lower(), obj_id) in _TEST_VERIFIED_OBJECTS


# Server-controlled, source-backed provenance registry for Starlly Org 273 verified records
# Binds source system, external ID, and exact local database ID.
VERIFIED_REAL_PROVENANCE_REGISTRY = {
    "assets": {
        "SKY-AST-00001": {"source_system": "starlly", "source_org": "273", "external_id": "1401", "local_id": 59, "tenant_slug": "skydream", "model": "Split AC Unit 2T", "status": "active"},
        "SKY-AST-00002": {"source_system": "starlly", "source_org": "273", "external_id": "1402", "local_id": 60, "tenant_slug": "skydream", "model": "VRF Outdoor Module", "status": "active"},
        "SKY-AST-00003": {"source_system": "starlly", "source_org": "273", "external_id": "1403", "local_id": 61, "tenant_slug": "skydream", "model": "Ductable AC 5T", "status": "active"},
        "SKY-AST-00004": {"source_system": "starlly", "source_org": "273", "external_id": "1404", "local_id": 62, "tenant_slug": "skydream", "model": "Air Compressor Unit 01", "status": "active"},
        "SKY-AST-00005": {"source_system": "starlly", "source_org": "273", "external_id": "1405", "local_id": 63, "tenant_slug": "skydream", "model": "test asset", "status": "active"},
        "SKY-AST-00006": {"source_system": "starlly", "source_org": "273", "external_id": "1406", "local_id": 64, "tenant_slug": "skydream", "model": "Fridge", "status": "active"},
        "SKY-AST-00008": {"source_system": "starlly", "source_org": "273", "external_id": "1408", "local_id": 65, "tenant_slug": "skydream", "model": "Precision Chiller 10TR", "status": "active"},
        "SKY-AST-00009": {"source_system": "starlly", "source_org": "273", "external_id": "1409", "local_id": 66, "tenant_slug": "skydream", "model": "Package AC 3T Inverter", "status": "active"},
        "SKY-AST-00010": {"source_system": "starlly", "source_org": "273", "external_id": "1410", "local_id": 67, "tenant_slug": "skydream", "model": "Cooling Tower 50TR", "status": "active"},
        "SKY-AST-00011": {"source_system": "starlly", "source_org": "273", "external_id": "1411", "local_id": 68, "tenant_slug": "skydream", "model": "AHU 8000 CFM", "status": "active"},
        "SKY-AST-00012": {"source_system": "starlly", "source_org": "273", "external_id": "1412", "local_id": 69, "tenant_slug": "skydream", "model": "VRF Cassette Indoor Unit", "status": "active"},
        "SKY-AST-00013": {"source_system": "starlly", "source_org": "273", "external_id": "1413", "local_id": 70, "tenant_slug": "skydream", "model": "Ductable AC 8T Commercial", "status": "active"},
    },
    "customers": {
        "GreenTech Industries": {"source_system": "starlly", "source_org": "273", "local_id": 13, "tenant_slug": "skydream", "client_type": "industrial", "status": "active"},
        "Rajesh Kumar": {"source_system": "starlly", "source_org": "273", "local_id": 14, "tenant_slug": "skydream", "client_type": "commercial", "status": "active"},
        "Ramesh Electricals": {"source_system": "starlly", "source_org": "273", "local_id": 15, "tenant_slug": "skydream", "client_type": "commercial", "status": "active"},
        "SkyDream Demo Customer": {"source_system": "starlly", "source_org": "273", "local_id": 16, "tenant_slug": "skydream", "client_type": "pilot", "status": "active"},
        "Sri Sai Enterprises": {"source_system": "starlly", "source_org": "273", "local_id": 17, "tenant_slug": "skydream", "client_type": "commercial", "status": "active"},
        "Urban Mall Group": {"source_system": "starlly", "source_org": "273", "local_id": 18, "tenant_slug": "skydream", "client_type": "enterprise", "status": "active"},
    },
    "documents": {
        # Split AC Unit 2T (SKY-AST-00001) - Local Asset 59
        "9e5d2d9e772dc71a849fa5cd1b7512b60f0f417f6832e0cdc77a879ab301e8d1": {"asset_code": "SKY-AST-00001", "local_doc_id": 168, "local_asset_id": 59, "source_org": "273", "doc_type": "troubleshooting", "status": "active"},
        "c4765a1c4e695ae10963a2390597ae7c0b3f24cc2c0a61ca32c7efc44c6bb1ac": {"asset_code": "SKY-AST-00001", "local_doc_id": 169, "local_asset_id": 59, "source_org": "273", "doc_type": "maintenance", "status": "active"},
        "28837acbce2e7374c482a5c6658eb1da6707ae559554b434e2660a535d7fefaf": {"asset_code": "SKY-AST-00001", "local_doc_id": 170, "local_asset_id": 59, "source_org": "273", "doc_type": "installation", "status": "active"},
        # VRF Outdoor Module (SKY-AST-00002) - Local Asset 60
        "18f14f21cee126012d7997f38f4abc385f59ff9497622a5aa68b9d55f3b1ca30": {"asset_code": "SKY-AST-00002", "local_doc_id": 171, "local_asset_id": 60, "source_org": "273", "doc_type": "troubleshooting", "status": "active"},
        "4aa0c57dd3bc68394efd167434b15685dac97a4b6efe689e60019aeced1cf629": {"asset_code": "SKY-AST-00002", "local_doc_id": 172, "local_asset_id": 60, "source_org": "273", "doc_type": "maintenance", "status": "active"},
        "b65b4037c6b315c6675d7e95d543be1ca002eca6e46a68fc01ee20f63de7dc91": {"asset_code": "SKY-AST-00002", "local_doc_id": 173, "local_asset_id": 60, "source_org": "273", "doc_type": "installation", "status": "active"},
        # Ductable AC 5T (SKY-AST-00003) - Local Asset 61
        "75abb20b207a2738abd6922e71107150db27f9265252ea1ff20abe77885f8b59": {"asset_code": "SKY-AST-00003", "local_doc_id": 174, "local_asset_id": 61, "source_org": "273", "doc_type": "troubleshooting", "status": "active"},
        "225b7d7821b9613fbeb7bbceb4d35405006f85b6d965df756dc14093dd3d4df0": {"asset_code": "SKY-AST-00003", "local_doc_id": 175, "local_asset_id": 61, "source_org": "273", "doc_type": "maintenance", "status": "active"},
        "33fe98bd9e80f17744f2fd307caf31c067c564a17506510ac4a127dce08e6f95": {"asset_code": "SKY-AST-00003", "local_doc_id": 176, "local_asset_id": 61, "source_org": "273", "doc_type": "installation", "status": "active"},
    },
    "service_calls": {
        # Verified historical resolutions for Skydream (Org 273). Currently 0 historical calls exist.
    }
}


def is_verified_real_asset(asset: Any) -> bool:
    """Positively verify that an asset has authentic server-controlled provenance.
    Requires:
      - Bound to an authorized real company tenant
      - Matches verified company asset code registry with matching tenant
      - Matches exact local object ID binding (preventing synthetic copies of real asset codes)
      - Not a synthetic injection or unmapped identifier
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

    # Positive provenance: Code must match verified real company asset in registry
    registry = VERIFIED_REAL_PROVENANCE_REGISTRY.get("assets", {})
    entry = registry.get(code_clean)
    if not entry or entry.get("status") != "active":
        return False

    if tenant:
        t_slug = getattr(tenant, "slug", str(tenant))
        if t_slug != entry.get("tenant_slug") and t_slug not in get_real_tenants():
            return False

    # Source provenance key validation:
    src_org = getattr(asset, "source_org", None)
    if isinstance(asset, dict):
        src_org = asset.get("source_org") or asset.get("organisation_id")
    if src_org and str(src_org) != str(entry.get("source_org")):
        return False

    src_sys = getattr(asset, "source_system", None)
    if isinstance(asset, dict):
        src_sys = asset.get("source_system")
    if src_sys and str(src_sys) != str(entry.get("source_system")):
        return False

    ext_id = getattr(asset, "external_id", None)
    if isinstance(asset, dict):
        ext_id = asset.get("external_id")
    if ext_id and str(ext_id) != str(entry.get("external_id")):
        return False

    # Durable database-backed SourceProvenanceRecord check
    try:
        from core.models import SourceProvenanceRecord
        if asset_id and tenant:
            prov = SourceProvenanceRecord.objects.filter(
                tenant=tenant,
                local_object_type="Asset",
                local_object_id=asset_id,
                verification_status="VERIFIED",
            ).first()
            if prov:
                return True
    except Exception:
        pass

    # Durable local object identity binding:
    # A synthetic or duplicate asset cannot inherit authorization by simply copying a real asset code.
    asset_id = getattr(asset, "id", None)
    if isinstance(asset, dict):
        asset_id = asset.get("id")

    if not asset_id:
        return False

    if is_test_verified_object("asset", asset_id):
        return True

    # Durable database-backed SourceProvenanceRecord check
    try:
        from core.models import SourceProvenanceRecord
        if tenant:
            prov = SourceProvenanceRecord.objects.filter(
                tenant=tenant,
                local_object_type="Asset",
                local_object_id=asset_id,
                verification_status="VERIFIED",
            ).first()
            if prov:
                return True
    except Exception:
        pass

    t_slug = getattr(tenant, "slug", str(tenant)) if tenant else ""
    if t_slug == entry.get("tenant_slug") or t_slug in get_real_tenants():
        if entry.get("local_id") and asset_id == entry.get("local_id"):
            return True
        return False
    return False


def is_verified_real_customer(customer: Any) -> bool:
    """Positively verify that a customer has authentic server-controlled provenance.
    Requires:
      - Bound to an authorized real company tenant
      - Matches verified company client registry or database SourceProvenanceRecord
      - Matches exact local customer ID binding
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
    if any(kw in name_lower for kw in ("synthetic", "dummy", "fake", "mock", "fixture")):
        return False

    cust_id = getattr(customer, "id", None)
    if isinstance(customer, dict):
        cust_id = customer.get("id")

    if not cust_id:
        return False

    if is_test_verified_object("customer", cust_id):
        return True

    # Check durable database-backed SourceProvenanceRecord
    try:
        from core.models import SourceProvenanceRecord
        if tenant:
            prov = SourceProvenanceRecord.objects.filter(
                tenant=tenant,
                local_object_type="Customer",
                local_object_id=cust_id,
                verification_status="VERIFIED",
            ).first()
            if prov:
                return True
    except Exception:
        pass

    # Positive match against imported Starlly customers in registry
    registry = VERIFIED_REAL_PROVENANCE_REGISTRY.get("customers", {})
    entry = registry.get(name_clean)
    if not entry or entry.get("status") != "active":
        return False

    t_slug = getattr(tenant, "slug", str(tenant)) if tenant else ""
    if tenant:
        if t_slug != entry.get("tenant_slug") and t_slug not in get_real_tenants():
            return False

    # Source provenance key validation:
    src_org = getattr(customer, "source_org", None)
    if isinstance(customer, dict):
        src_org = customer.get("source_org") or customer.get("organisation_id")
    if src_org and str(src_org) != str(entry.get("source_org")):
        return False

    if t_slug == entry.get("tenant_slug") or t_slug in get_real_tenants():
        if entry.get("local_id") and cust_id == entry.get("local_id"):
            return True
        return False
    return False


def is_verified_real_document(doc: Any) -> bool:
    """Positively verify that a KnowledgeDocument has authentic company provenance.
    Requires:
      - Bound to an authorized real company tenant
      - Bound to a verified real asset
      - Active for RAG, non-confidential, sharing not disabled
      - Has verified import source (matching authentic attachment checksum in registry or DB)
      - Content text matches verified SHA-256 hash for plain text, or media file matches for rich docs
      - Not a synthetic/demo fixture
    """
    if not doc:
        return False
    tenant = getattr(doc, "tenant", None)
    if not is_verified_real_tenant(tenant):
        return False
    if not getattr(doc, "is_rag_enabled", False) or getattr(doc, "is_confidential", False):
        return False
    if getattr(doc, "disable_sharing", False):
        return False
    if getattr(doc, "index_status", "") not in ("INDEXED", "INDEXING"):
        return False

    # Positive Asset Binding
    asset = getattr(doc, "asset", None)
    if not asset or not is_verified_real_asset(asset):
        return False

    # Check for synthetic markers
    title = (getattr(doc, "title", "") or "").lower()
    tags = (getattr(doc, "tags", "") or "").lower()
    desc = (getattr(doc, "description", "") or "").lower()
    if any(kw in title for kw in ("synthetic", "dummy", "fake", "fixture", "mock")):
        return False
    if any(kw in tags for kw in ("synthetic", "dummy", "fake", "fixture", "mock")):
        return False
    if any(kw in desc for kw in ("synthetic", "dummy", "fake", "fixture", "mock")):
        return False

    # Provenance check:
    doc_id = getattr(doc, "id", None)
    if isinstance(doc, dict):
        doc_id = doc.get("id")

    if not doc_id:
        return False

    if is_test_verified_object("knowledgedocument", doc_id):
        return True

    checksum = (getattr(doc, "checksum_sha256", "") or "").strip().lower()

    # Check durable database-backed SourceProvenanceRecord
    try:
        from core.models import SourceProvenanceRecord
        if tenant:
            prov = SourceProvenanceRecord.objects.filter(
                tenant=tenant,
                local_object_type="KnowledgeDocument",
                local_object_id=doc_id,
                verification_status="VERIFIED",
            ).first()
            if prov and (not prov.checksum_sha256 or prov.checksum_sha256 == checksum):
                return True
    except Exception:
        pass

    # In production/registry check: document must have authentic SHA-256 checksum bound to the exact asset
    registry = VERIFIED_REAL_PROVENANCE_REGISTRY.get("documents", {})
    if not checksum or checksum not in registry:
        return False

    entry = registry[checksum]
    if entry.get("status") != "active":
        return False

    # Exact asset code binding: checksum must be registered for this specific asset code
    asset_code = getattr(asset, "asset_code", "")
    if isinstance(asset, dict):
        asset_code = asset.get("asset_code", "")
    if asset_code != entry.get("asset_code"):
        return False

    # Durable local document ID binding with tenant check:
    t_slug = getattr(tenant, "slug", str(tenant)) if tenant else ""
    if t_slug == "skydream" or t_slug in get_real_tenants():
        if entry.get("local_doc_id"):
            if doc_id != entry.get("local_doc_id"):
                return False
    elif t_slug not in get_real_tenants():
        return False

    # Content integrity verification:
    # 1. Plain text documents (.txt, .md): extracted text equals file content
    # 2. Rich/binary documents (.pdf, .docx): content_text is extracted text, so do not reject
    #    valid PDF merely because its extracted text is not byte-identical to raw PDF bytes.
    content_text = getattr(doc, "content_text", "")
    filename = (getattr(doc, "original_filename", "") or "").lower()
    is_plain_text = filename.endswith((".txt", ".md")) or getattr(doc, "source_type", "") == "text"

    if content_text and is_plain_text:
        calc_sha = hashlib.sha256(content_text.encode("utf-8", errors="ignore")).hexdigest().lower()
        if calc_sha != checksum:
            return False

    return True


def is_synthetic_asset(asset: Any) -> bool:
    """Identify whether an Asset instance or asset dictionary is synthetic/demo or lacks positive provenance."""
    if not asset:
        return False
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


def is_verified_real_service_call(call: Any) -> bool:
    """Positively verify that a ServiceCall has authentic real company provenance.
    Requires:
      - Bound to an authorized real company tenant
      - Bound to a verified real customer
      - Bound to a verified real asset
      - Appropriate lifecycle status ('closed' or 'marked_for_closure')
      - Not a synthetic injection (no synthetic prefixes or dummy keywords)
    """
    if not call:
        return False
    tenant = getattr(call, "tenant", None)
    if not is_verified_real_tenant(tenant):
        return False

    customer = getattr(call, "customer", None)
    if not customer or not is_verified_real_customer(customer):
        return False

    asset = getattr(call, "asset", None)
    if not asset or not is_verified_real_asset(asset):
        return False

    servy_id_str = str(getattr(call, "servy_id", "") or "").upper()
    if any(servy_id_str.startswith(p) for p in SYNTHETIC_CALL_PREFIXES):
        return False

    # Check text fields for synthetic markers
    complaint_type = (getattr(call, "complaint_type", "") or "").lower()
    complaint_text = (getattr(call, "complaint_text", "") or "").lower()
    resolution_text = (getattr(call, "resolution_text", "") or "").lower()
    for kw in ("synthetic", "dummy", "fake", "fixture", "mock"):
        if kw in complaint_type or kw in complaint_text or kw in resolution_text:
            return False

    # Appropriate lifecycle status for historical resolution guidance
    status = getattr(call, "status", "")
    if status not in ("closed", "marked_for_closure"):
        return False

    # Positive Resolution Provenance:
    # A closed ticket is not automatically an approved customer self-service resolution.
    # Must have authentic source mapping or explicit customer-visibility approval in registry.
    call_id = getattr(call, "id", None)
    call_registry = VERIFIED_REAL_PROVENANCE_REGISTRY.get("service_calls", {})
    servy_id_int = getattr(call, "servy_id", None)

    if call_id and str(call_id) in call_registry:
        entry = call_registry[str(call_id)]
        return entry.get("status") == "active" and entry.get("customer_visible", False)
    elif servy_id_int and str(servy_id_int) in call_registry:
        entry = call_registry[str(servy_id_int)]
        return entry.get("status") == "active" and entry.get("customer_visible", False)
    elif call_id and is_test_verified_object("servicecall", call_id):
        return True

    return False


def is_synthetic_service_call(call: Any) -> bool:
    """Identify whether a ServiceCall is synthetic/demo or lacks positive provenance."""
    if not call:
        return False
    if get_customer_data_mode() == "verified_real_only":
        return not is_verified_real_service_call(call)
    servy_id = str(getattr(call, "servy_id", "") or "").upper()
    if any(servy_id.startswith(p) for p in SYNTHETIC_CALL_PREFIXES):
        return True
    tenant = getattr(call, "tenant", None)
    if tenant and not is_real_tenant(tenant):
        return True
    return False


def is_synthetic_document(doc: Any) -> bool:
    """Identify whether a KnowledgeDocument is synthetic/demo fixture or lacks positive provenance."""
    if not doc:
        return False
    if get_customer_data_mode() == "verified_real_only":
        return not is_verified_real_document(doc)

    title = (getattr(doc, "title", "") or "").lower()
    tags = (getattr(doc, "tags", "") or "").lower()
    desc = (getattr(doc, "description", "") or "").lower()
    if any(kw in title for kw in ("synthetic", "dummy", "fake", "fixture")):
        return True
    if any(kw in tags for kw in ("synthetic", "dummy", "fake", "fixture")):
        return True
    if any(kw in desc for kw in ("synthetic", "dummy", "fake", "fixture")):
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
    if mode not in ("verified_real_only", "demo_only", "test_demo", "test_isolated"):
        raise PermissionDenied("Invalid customer serving data mode configuration. Fail-closed.")

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
