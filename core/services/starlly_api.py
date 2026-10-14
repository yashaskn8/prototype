"""Starlly Solutions external API client service.

Provides a secure, reusable interface for communicating with Starlly's
external endpoints (specifically the Assets API) without exposing credentials
to clients, templates, logs, or exceptions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from django.conf import settings
import requests

logger = logging.getLogger("servy.starlly_api")

DEFAULT_TIMEOUT_SECONDS = 15


class StarllyError(Exception):
    """Base exception for all Starlly API client interactions."""
    pass


class StarllyConfigurationError(StarllyError):
    """Raised when configuration is incomplete, invalid, or blocked."""
    def __init__(self, message: str, is_blocker: bool = True):
        super().__init__(message)
        self.is_blocker = is_blocker


class StarllyAuthenticationError(StarllyError):
    """Raised when Starlly rejects API authentication (HTTP 401 or 403)."""
    pass


class StarllyNotFoundError(StarllyError):
    """Raised when an asset or resource is not found (HTTP 404)."""
    pass


class StarllyTimeoutError(StarllyError):
    """Raised when a request to Starlly API times out."""
    pass


class StarllyConnectionError(StarllyError):
    """Raised when connection to Starlly API fails (DNS, network, TLS)."""
    pass


class StarllyTenantIsolationError(StarllyError):
    """Raised when a Servy tenant is not authorized for external Starlly access."""
    pass


class StarllyApiError(StarllyError):
    """Raised when Starlly returns an unexpected response or 5xx error."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def get_authorized_org_for_tenant(tenant) -> Optional[str]:
    """Resolve and verify the authorized Starlly organisation ID for a given Servy tenant.

    Returns the authorized organisation ID string, or None if the tenant is unmapped.
    """
    if tenant is None:
        return None
    mapping = getattr(settings, "STARLLY_TENANT_ORGANISATION_MAP", {})
    tenant_slug = getattr(tenant, "slug", str(tenant) if isinstance(tenant, str) else "")
    tenant_id = str(getattr(tenant, "id", ""))

    if tenant_slug and tenant_slug in mapping:
        return str(mapping[tenant_slug]).strip()
    if tenant_id and tenant_id in mapping:
        return str(mapping[tenant_id]).strip()
    return None


def mask_secret(secret: Optional[str]) -> str:
    """Safely mask secret strings so credentials never leak into logs or UI."""
    if not secret:
        return "<none>"
    s = str(secret).strip()
    if len(s) <= 8:
        return "sk_***"
    return f"{s[:4]}...{s[-4:]}"


def validate_base_url(url: Optional[str]) -> str:
    """Validate Starlly API base URL and detect incomplete / blocker configurations.

    Raises StarllyConfigurationError if:
    - URL is empty or missing
    - Scheme is not http or https
    - Hostname contains placeholder text (e.g. YOUR-ACTUAL-SUBDOMAIN)
    - Subdomain is missing (e.g. https://.starlly.in or https://starlly.in)
    - Hostname starts with a period or has invalid syntax
    """
    if not url or not str(url).strip():
        raise StarllyConfigurationError(
            "STARLLY_API_BASE_URL is not configured in environment.",
            is_blocker=True,
        )

    clean_url = str(url).strip().rstrip("/")
    parsed = urlparse(clean_url)

    if not parsed.scheme or parsed.scheme.lower() not in ("http", "https"):
        raise StarllyConfigurationError(
            f"Invalid URL scheme '{parsed.scheme}'. Starlly API base URL must use HTTPS (e.g. https://<tenant>.starlly.in).",
            is_blocker=True,
        )

    # In production, require HTTPS strictly. Allow http only for localhost test mocks in DEBUG
    if parsed.scheme.lower() == "http":
        is_debug_localhost = getattr(settings, "DEBUG", False) and ("localhost" in parsed.netloc or "127.0.0.1" in parsed.netloc)
        if not is_debug_localhost:
            raise StarllyConfigurationError(
                f"Insecure URL scheme '{parsed.scheme}'. Starlly API base URL must use HTTPS.",
                is_blocker=True,
            )

    netloc = parsed.netloc.lower()
    # Check for empty hostname or malformed hostname like '.starlly.in'
    if not netloc or netloc.startswith(".") or netloc == "starlly.in" or netloc == ".starlly.in":
        raise StarllyConfigurationError(
            "Configuration Blocker: The provided base URL is incomplete (e.g. 'https://.starlly.in'). "
            "A valid tenant subdomain is required before connecting to Starlly (e.g. https://servybackdev.starlly.in).",
            is_blocker=True,
        )

    if "your-actual-subdomain" in netloc or "placeholder" in netloc:
        raise StarllyConfigurationError(
            f"Configuration Blocker: STARLLY_API_BASE_URL contains placeholder host '{netloc}'. "
            "Replace with your actual registered Starlly hostname.",
            is_blocker=True,
        )

    return clean_url


def validate_api_key(key: Optional[str]) -> str:
    """Validate Starlly API key.

    Raises StarllyConfigurationError if missing or placeholder.
    """
    if not key or not str(key).strip():
        raise StarllyConfigurationError(
            "STARLLY_API_KEY is not configured in environment.",
            is_blocker=True,
        )
    clean_key = str(key).strip()
    if "paste_your_secret_key" in clean_key.lower():
        raise StarllyConfigurationError(
            "Configuration Blocker: STARLLY_API_KEY contains placeholder text. "
            "Provide a valid Starlly API secret key.",
            is_blocker=True,
        )
    return clean_key


class StarllyApiClient:
    """Production-hardened client for Starlly External Assets API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        org_id: Optional[Union[str, int]] = None,
        tenant: Any = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self._tenant = tenant
        # Resolve config from arguments -> Django settings -> os.environ
        resolved_url = base_url if base_url is not None else (getattr(settings, "STARLLY_API_BASE_URL", "") or os.getenv("STARLLY_API_BASE_URL", ""))
        resolved_key = api_key if api_key is not None else (getattr(settings, "STARLLY_API_KEY", "") or os.getenv("STARLLY_API_KEY", ""))

        if tenant is not None:
            authorized_org = get_authorized_org_for_tenant(tenant)
            if not authorized_org:
                raise StarllyTenantIsolationError(
                    f"Tenant '{getattr(tenant, 'slug', tenant)}' is not authorized for external Starlly integrations."
                )
            if org_id is not None and str(org_id) != str(authorized_org):
                raise StarllyTenantIsolationError(
                    f"Tenant '{getattr(tenant, 'slug', tenant)}' cannot access organisation '{org_id}'."
                )
            resolved_org = authorized_org
        else:
            resolved_org = org_id if org_id is not None else (getattr(settings, "STARLLY_ORGANISATION_ID", "") or os.getenv("STARLLY_ORGANISATION_ID", ""))

        self._base_url = str(resolved_url).strip()
        self._api_key = str(resolved_key).strip()
        self._org_id = str(resolved_org).strip() if resolved_org else None
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return validate_base_url(self._base_url)

    @property
    def api_key(self) -> str:
        return validate_api_key(self._api_key)

    @property
    def org_id(self) -> Optional[str]:
        return self._org_id

    def get_auth_headers(self) -> Dict[str, str]:
        """Returns the documented authentication headers for Starlly API.

        Documented & verified format:
        Authorization: Api-Key <STARLLY_API_KEY>
        """
        key = self.api_key
        return {
            "Authorization": f"Api-Key {key}",
            "Accept": "application/json",
            "User-Agent": "Servy-RAG-Backend/1.0",
        }

    def fetch_assets(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch asset records from Starlly External Assets API.

        Endpoint: GET /api/external/assets/
        Authentication: Authorization: Api-Key <STARLLY_API_KEY>
        """
        # Validates base URL and API key
        endpoint_url = f"{self.base_url}/api/external/assets/"
        headers = self.get_auth_headers()

        params: Dict[str, Any] = {}
        if limit is not None and limit > 0:
            params["limit"] = int(limit)
        if offset is not None and offset >= 0:
            params["offset"] = int(offset)
        if search and str(search).strip():
            params["search"] = str(search).strip()

        try:
            logger.info("Calling Starlly assets API at %s (params=%s)", endpoint_url, params)
            response = requests.get(
                endpoint_url,
                headers=headers,
                params=params,
                timeout=self._timeout,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            logger.warning("Starlly Assets API timed out after %ds", self._timeout)
            raise StarllyTimeoutError(
                f"Connection to Starlly Assets API timed out after {self._timeout} seconds."
            ) from exc
        except requests.ConnectionError as exc:
            logger.error("Failed to connect to Starlly Assets API at %s", self._base_url)
            raise StarllyConnectionError(
                f"Failed to connect to Starlly API server ({self._base_url}). Service may be unreachable."
            ) from exc
        except requests.RequestException as exc:
            logger.error("Starlly Assets API request exception: %s", exc)
            raise StarllyApiError(f"HTTP request to Starlly failed: {exc}") from exc

        # Handle Redirects (Prevent credential leakage to untrusted hosts)
        is_redirect_code = getattr(response, "status_code", None) in (301, 302, 303, 307, 308)
        if is_redirect_code:
            target_location = str(getattr(response, "headers", {}).get("Location", "") or "").strip()
            if target_location:
                parsed_target = urlparse(target_location)
                base_parsed = urlparse(endpoint_url)
                if parsed_target.netloc and parsed_target.netloc.lower() != base_parsed.netloc.lower():
                    logger.error("Redirect to untrusted host rejected: %s", parsed_target.netloc)
                    raise StarllyApiError("Redirect to untrusted destination rejected to prevent credential leakage.")
                if parsed_target.scheme and parsed_target.scheme.lower() != "https":
                    raise StarllyApiError("Insecure redirect downgraded to HTTP rejected.")

        # Handle HTTP Status
        if response.status_code in (401, 403):
            logger.warning("Starlly API rejected credentials: status=%d", response.status_code)
            raise StarllyAuthenticationError(
                "Starlly API rejected credentials. Verify that STARLLY_API_KEY is valid and authorized for organization."
            )
        elif response.status_code == 404:
            raise StarllyNotFoundError("Starlly Assets endpoint returned 404 Not Found.")
        elif response.status_code >= 500:
            logger.error("Starlly API server error %d", response.status_code)
            raise StarllyApiError(
                f"Starlly API server error (HTTP {response.status_code}). Please try again later.",
                status_code=response.status_code,
            )
        elif response.status_code >= 400:
            logger.warning("Starlly API returned HTTP %d", response.status_code)
            raise StarllyApiError(
                f"Starlly API returned client error (HTTP {response.status_code}).",
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except Exception as exc:
            logger.error("Invalid JSON response from Starlly Assets API: %s", exc)
            raise StarllyApiError("Starlly Assets API returned invalid JSON response.") from exc

        # Process payload (handles both direct list and paginated dict formats)
        raw_assets: List[Dict[str, Any]] = []
        total_count = 0

        if isinstance(payload, list):
            raw_assets = payload
            total_count = len(payload)
        elif isinstance(payload, dict):
            # DRF pagination standard
            if "results" in payload and isinstance(payload["results"], list):
                raw_assets = payload["results"]
                total_count = payload.get("count", len(raw_assets))
            elif "data" in payload and isinstance(payload["data"], list):
                raw_assets = payload["data"]
                total_count = payload.get("count", len(raw_assets))
            else:
                raw_assets = [payload]
                total_count = 1
        else:
            raise StarllyApiError(f"Unexpected payload type '{type(payload).__name__}' from Starlly Assets API.")

        # Filter and validate by org_id
        filtered_assets = []
        for asset in raw_assets:
            if not isinstance(asset, dict):
                continue
            rec_org = asset.get("org_id") or asset.get("organisation_id") or asset.get("organization_id")
            if self._org_id:
                if rec_org is not None and str(rec_org) != str(self._org_id):
                    logger.warning(
                        "Rejecting asset record %s: unexpected organisation ID %s (expected %s)",
                        asset.get("id"), rec_org, self._org_id
                    )
                    continue
            filtered_assets.append(asset)

        return {
            "assets": filtered_assets,
            "count": total_count,
            "total_returned": len(filtered_assets),
            "organisation_id": int(self._org_id) if (self._org_id and self._org_id.isdigit()) else self._org_id,
            "limit": limit,
            "offset": offset,
        }


def get_starlly_client(tenant: Any = None) -> StarllyApiClient:
    """Convenience factory for obtaining a StarllyApiClient instance bound to an authorized tenant."""
    return StarllyApiClient(tenant=tenant)
