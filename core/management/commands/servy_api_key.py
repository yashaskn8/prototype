"""Issue, list and revoke portal API keys.

    manage.py servy_api_key issue --username portal_acme --name "Acme portal" [--days 365]
    manage.py servy_api_key list [--username portal_acme]
    manage.py servy_api_key revoke --prefix 1a2b3c4d5e6f

The plaintext key is printed once by ``issue`` and never stored.
"""
from datetime import timedelta

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.api_keys import issue_key, revoke_key
from core.models import ApiClientKey, TenantMembership


class Command(BaseCommand):
    help = "Manage server-to-server API keys for external portal integrations."

    def add_arguments(self, parser):
        parser.add_argument("action", choices=["issue", "list", "revoke"])
        parser.add_argument("--username")
        parser.add_argument("--name", default="portal")
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--prefix")

    def handle(self, *args, action, username=None, name="portal", days=365, prefix=None, **kwargs):
        if action == "issue":
            user = User.objects.filter(username=username).first() if username else None
            if user is None or user.is_superuser:
                raise CommandError("Give --username of an existing non-superuser account.")
            memberships = list(TenantMembership.objects.filter(user=user, is_active=True))
            if len(memberships) != 1:
                raise CommandError("The account must have exactly one active tenant membership.")
            membership = memberships[0]
            if membership.role == "customer" and not membership.customer_id:
                raise CommandError("The customer account is not linked to a customer profile.")
            expires = timezone.now() + timedelta(days=days) if days > 0 else None
            record, key = issue_key(tenant=membership.tenant, user=user, name=name, expires_at=expires)
            self.stdout.write(f"Issued key {record.key_prefix} for {username} (role {membership.role}), "
                              f"expires {expires or 'never'}.")
            self.stdout.write("Store it in the portal's secret store now; it cannot be shown again:")
            self.stdout.write(key)
        elif action == "list":
            qs = ApiClientKey.objects.select_related("user").order_by("-created_at")
            if username:
                qs = qs.filter(user__username=username)
            for k in qs:
                state = "active" if k.is_usable else ("revoked" if k.revoked_at else "expired")
                self.stdout.write(f"{k.key_prefix}  {k.user.username}  {k.name}  {state}  "
                                  f"created {k.created_at:%Y-%m-%d}  last used {k.last_used_at or '-'}")
        else:
            if not prefix:
                raise CommandError("Give --prefix of the key to revoke.")
            count = revoke_key(prefix)
            self.stdout.write(f"Revoked {count} key(s).")
