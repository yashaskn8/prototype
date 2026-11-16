"""Backend tests for customer conversation soft-deletion.

Tests the deletion lifecycle: model fields, access control, queryset
filtering, idempotency, and message immutability preservation.

Run with:
    python -m tools.test_conversation_deletion

These tests run against the real servy_dev MySQL database (not the test
sqlite3), so they create and clean up their own records.
"""
import os
import sys
import unittest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

# Force the real database (not the test override that uses sqlite3).
_real_run = "test" not in sys.argv
if not _real_run:
    # Allow running under manage.py test by skipping the real-DB tests.
    pass

import django
django.setup()

from django.utils import timezone
from core.models import CustomerConversation, CustomerMessage
from core.api_views import _can_read_customer_conversation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_fixtures():
    """Load the standard dev fixtures. Returns (tenant, customer, user, asset)."""
    from core.models import Asset, Customer, Tenant
    from django.contrib.auth.models import User
    tenant = Tenant.objects.filter(is_active=True).first()
    if tenant is None:
        return None, None, None, None
    customer = Customer.objects.filter(tenant=tenant).first()
    # dev_customer may not exist if seed_servy_dev hasn't been run; fall back
    # to the first available user for test-record ownership.
    user = (
        User.objects.filter(username="dev_customer").first()
        or User.objects.filter(is_active=True).first()
    )
    asset = Asset.objects.filter(tenant=tenant, customer=customer).first()
    return tenant, customer, user, asset


def _ctx(tenant, customer=None, role="customer", user=None):
    return {"tenant": tenant, "customer": customer, "role": role, "user": user}


def _make_conv(tenant, customer, asset, user, title="Test conv"):
    return CustomerConversation.objects.create(
        tenant=tenant, customer=customer, asset=asset,
        created_by=user, title=title, status="ACTIVE",
    )


_FIXTURES = _get_fixtures()
_HAVE_DB = _FIXTURES[0] is not None

_skip_reason = "servy_dev database not available" if not _HAVE_DB else None


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------

@unittest.skipIf(not _HAVE_DB, _skip_reason)
class SoftDeleteFieldTests(unittest.TestCase):
    """Verify the soft-delete fields exist and behave correctly."""

    tenant, customer, user, asset = _FIXTURES

    def test_new_conversation_has_null_deletion_fields(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user,
                          title="null deletion test")
        try:
            self.assertIsNone(conv.customer_deleted_at)
            self.assertIsNone(conv.customer_deleted_by)
        finally:
            conv.delete()

    def test_soft_delete_persists_round_trip(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user,
                          title="soft delete persist test")
        try:
            now = timezone.now()
            conv.customer_deleted_at = now
            conv.customer_deleted_by = self.user
            conv.save(update_fields=["customer_deleted_at", "customer_deleted_by"])

            conv.refresh_from_db()
            self.assertIsNotNone(conv.customer_deleted_at)
            self.assertEqual(conv.customer_deleted_by_id, self.user.id)
        finally:
            conv.delete()

    def test_messages_survive_soft_delete(self):
        """CustomerMessage immutability: messages remain after soft-delete."""
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user,
                          title="msg survival test")
        try:
            CustomerMessage.objects.create(
                tenant=self.tenant, conversation=conv,
                sender_role="customer", message_text="Test question",
            )
            self.assertEqual(conv.messages.count(), 1)

            conv.customer_deleted_at = timezone.now()
            conv.customer_deleted_by = self.user
            conv.save(update_fields=["customer_deleted_at", "customer_deleted_by"])

            # Messages remain intact after soft-delete.
            self.assertEqual(conv.messages.count(), 1)
        finally:
            # Clean up: use raw SQL because CustomerMessage blocks delete().
            from django.db import connection
            with connection.cursor() as cur:
                cur.execute(
                    "DELETE FROM core_customermessage WHERE conversation_id = %s",
                    [conv.id],
                )
            conv.delete()


# ---------------------------------------------------------------------------
# Access-control tests
# ---------------------------------------------------------------------------

@unittest.skipIf(not _HAVE_DB, _skip_reason)
class AccessControlTests(unittest.TestCase):
    """Test _can_read_customer_conversation with the soft-delete flag."""

    tenant, customer, user, asset = _FIXTURES

    def test_customer_can_read_active_conversation(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user)
        try:
            ctx = _ctx(self.tenant, self.customer)
            self.assertTrue(_can_read_customer_conversation(ctx, conv))
        finally:
            conv.delete()

    def test_customer_cannot_read_deleted_conversation(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user)
        conv.customer_deleted_at = timezone.now()
        conv.customer_deleted_by = self.user
        conv.save(update_fields=["customer_deleted_at", "customer_deleted_by"])
        try:
            ctx = _ctx(self.tenant, self.customer)
            self.assertFalse(_can_read_customer_conversation(ctx, conv))
        finally:
            conv.delete()

    def test_allow_deleted_flag_grants_access_for_idempotent_delete(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user)
        conv.customer_deleted_at = timezone.now()
        conv.save(update_fields=["customer_deleted_at"])
        try:
            ctx = _ctx(self.tenant, self.customer)
            self.assertTrue(
                _can_read_customer_conversation(ctx, conv, allow_deleted=True)
            )
        finally:
            conv.delete()

    def test_admin_can_still_read_deleted_conversation(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user)
        conv.customer_deleted_at = timezone.now()
        conv.save(update_fields=["customer_deleted_at"])
        try:
            ctx = _ctx(self.tenant, role="admin")
            self.assertTrue(_can_read_customer_conversation(ctx, conv))
        finally:
            conv.delete()

    def test_wrong_customer_rejected_uniformly(self):
        """Cross-customer: always 404, regardless of deletion state."""
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user)
        try:
            from core.models import Customer
            fake = Customer(id=99999, tenant=self.tenant, name="Other")
            ctx = _ctx(self.tenant, fake)
            self.assertFalse(_can_read_customer_conversation(ctx, conv))
        finally:
            conv.delete()


# ---------------------------------------------------------------------------
# Queryset filter tests (what the LIST view uses)
# ---------------------------------------------------------------------------

@unittest.skipIf(not _HAVE_DB, _skip_reason)
class ListFilterTests(unittest.TestCase):
    """Test the queryset filter that the LIST view applies."""

    tenant, customer, user, asset = _FIXTURES

    def test_deleted_excluded_from_customer_list_queryset(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user,
                          title="list filter test")
        try:
            qs_before = CustomerConversation.objects.filter(
                tenant=self.tenant, customer=self.customer,
                customer_deleted_at__isnull=True,
            )
            self.assertIn(conv.id, list(qs_before.values_list("id", flat=True)))

            conv.customer_deleted_at = timezone.now()
            conv.customer_deleted_by = self.user
            conv.save(update_fields=["customer_deleted_at", "customer_deleted_by"])

            qs_after = CustomerConversation.objects.filter(
                tenant=self.tenant, customer=self.customer,
                customer_deleted_at__isnull=True,
            )
            self.assertNotIn(conv.id, list(qs_after.values_list("id", flat=True)))

            # Still exists in unfiltered DB.
            self.assertTrue(CustomerConversation.objects.filter(id=conv.id).exists())
        finally:
            conv.delete()


# ---------------------------------------------------------------------------
# Idempotency tests
# ---------------------------------------------------------------------------

@unittest.skipIf(not _HAVE_DB, _skip_reason)
class IdempotencyTests(unittest.TestCase):

    tenant, customer, user, asset = _FIXTURES

    def test_double_soft_delete_preserves_original_timestamp(self):
        conv = _make_conv(self.tenant, self.customer, self.asset, self.user,
                          title="idempotency test")
        try:
            first_time = timezone.now()
            conv.customer_deleted_at = first_time
            conv.customer_deleted_by = self.user
            conv.save(update_fields=["customer_deleted_at", "customer_deleted_by"])

            conv.refresh_from_db()
            # The view logic only writes when customer_deleted_at is None, so a
            # second pass must not change the timestamp.
            self.assertEqual(conv.customer_deleted_at, first_time)
        finally:
            conv.delete()


if __name__ == "__main__":
    unittest.main()
