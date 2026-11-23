"""Server-to-server API keys for external portal integrations.

Request header::

    Authorization: Bearer servy_<prefix>_<secret>

The key authenticates as the Servy user it was issued for; authorization is then
the normal per-request tenant/customer/asset scoping (``require_api_context``).
Token-authenticated requests are not subject to CSRF (no cookies are involved).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional, Tuple

from django.utils import timezone
from rest_framework import authentication, exceptions

KEY_PREFIX = "servy_"


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def issue_key(*, tenant, user, name: str, expires_at=None) -> Tuple[object, str]:
    """Create a key and return (record, plaintext). The plaintext is never stored."""
    from core.models import ApiClientKey
    prefix = secrets.token_hex(6)
    key = f"{KEY_PREFIX}{prefix}_{secrets.token_urlsafe(32)}"
    record = ApiClientKey.objects.create(tenant=tenant, user=user, name=name[:120], key_prefix=prefix,
                                         key_sha256=_digest(key), expires_at=expires_at)
    return record, key


def revoke_key(prefix: str) -> int:
    from core.models import ApiClientKey
    return ApiClientKey.objects.filter(key_prefix=prefix, revoked_at__isnull=True).update(revoked_at=timezone.now())


def lookup_key(key: str) -> Optional[object]:
    from core.models import ApiClientKey
    if not key.startswith(KEY_PREFIX) or key.count("_") < 2:
        return None
    prefix = key[len(KEY_PREFIX):].split("_", 1)[0]
    record = ApiClientKey.objects.select_related("user", "tenant").filter(key_prefix=prefix).first()
    if record is None or not hmac.compare_digest(record.key_sha256, _digest(key)):
        return None
    return record


class ApiKeyAuthentication(authentication.BaseAuthentication):
    keyword = "Bearer"

    def authenticate(self, request):
        header = authentication.get_authorization_header(request).decode("latin-1", "replace").split()
        if not header or header[0] != self.keyword:
            return None
        if len(header) != 2:
            raise exceptions.AuthenticationFailed("Invalid authorization header.")
        record = lookup_key(header[1])
        # One uniform message: never reveal whether a key exists, expired or was revoked.
        if record is None or not record.is_usable or not record.user.is_active:
            raise exceptions.AuthenticationFailed("Invalid or inactive API key.")
        # A key acts for exactly one customer: the account must still be an active
        # customer-role member of the key's tenant, linked to a customer profile.
        # Staff/admin accounts (which can see many customers) never use keys.
        if (record.user.is_superuser or not record.tenant.is_active
                or not record.user.tenant_memberships.filter(
                    tenant=record.tenant, is_active=True, role="customer", customer__isnull=False).exists()):
            raise exceptions.AuthenticationFailed("Invalid or inactive API key.")
        now = timezone.now()
        if record.last_used_at is None or (now - record.last_used_at).total_seconds() > 60:
            type(record).objects.filter(pk=record.pk).update(last_used_at=now)
        # Bind the request to the key's tenant (require_api_context reads this).
        request._servy_api_key = record
        return record.user, record

    def authenticate_header(self, request):
        # 401 + WWW-Authenticate only when a Bearer key was actually presented;
        # anonymous requests keep the existing 403 JSON contract.
        header = authentication.get_authorization_header(request).decode("latin-1", "replace")
        return self.keyword if header.split(" ", 1)[0] == self.keyword else None
