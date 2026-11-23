"""Customer / store / technician data boundaries on direct API calls, and the
original-document integrity gate on every customer answer path. Synthetic data."""
import json
from unittest import mock

from django.contrib.auth.models import User
from django.test import override_settings
from rest_framework.test import APIClient

from core.models import (
    CustomerConversation, DiagnosticSession, RecoveryPassport, StaffProfile, TenantMembership)
from core.services.ticketing import create_service_call_record
from core.tests_conversational_gateway import VRFConversationGatewayTests
from core.tests_demo_gateway import GatewayTestBase

REPORT_MARKER = "INTERNAL-HANDOVER-ANALYSIS"
PASSPORT_MARKER = "PASSPORT-INTERNAL-HISTORY"


@override_settings(ROOT_URLCONF="servy_rag.urls")
class RoleDataBoundaryTests(GatewayTestBase):
    def setUp(self):
        super().setUp()
        self.admin = User.objects.create(username="synthetic_admin")
        TenantMembership.objects.create(user=self.admin, tenant=self.tenant, role="admin")
        self.store = User.objects.create(username="synthetic_store")
        TenantMembership.objects.create(user=self.store, tenant=self.tenant, role="store_operator")
        self.other_user = User.objects.create(username="synthetic_other_customer")
        TenantMembership.objects.create(user=self.other_user, tenant=self.tenant, role="customer",
                                        customer=self.other_customer)
        self.call = create_service_call_record(
            self.tenant, "Cooling", "Customer says the unit is not cooling.", self.customer,
            site=self.site, asset=self.asset, technician=self.tech, contact_name="Synthetic Contact",
            contact_phone="+1 555 0100", technician_notes=f"{REPORT_MARKER}: possible refrigerant fault.")
        session = DiagnosticSession.objects.create(tenant=self.tenant, customer=self.customer, asset=self.asset)
        RecoveryPassport.objects.create(tenant=self.tenant, session=session, service_call=self.call,
                                        asset=self.asset, complaint="not cooling",
                                        structured_data={"history": PASSPORT_MARKER})

    def client_for(self, user):
        client = APIClient()
        client.force_login(user)
        return client

    def test_call_detail_by_role(self):
        url = f"/api/calls/{self.call.id}/"
        customer = self.client_for(self.customer_user).get(url)
        self.assertEqual(customer.status_code, 200)
        body = json.dumps(customer.json())
        self.assertNotIn(REPORT_MARKER, body)
        self.assertNotIn(PASSPORT_MARKER, body)
        self.assertIsNone(customer.json()["recovery_passport"])
        self.assertEqual(customer.json()["call"]["status"], self.call.status)   # status still visible

        store = self.client_for(self.store).get(url)
        self.assertEqual(store.status_code, 200)
        body = json.dumps(store.json())
        for secret in (REPORT_MARKER, PASSPORT_MARKER, "not cooling", "+1 555 0100", "Synthetic Contact"):
            self.assertNotIn(secret, body)
        self.assertEqual(store.json()["call"]["servy_id"], self.call.servy_id)

        admin = self.client_for(self.admin).get(url)
        self.assertIn(REPORT_MARKER, json.dumps(admin.json()))
        self.assertIn(PASSPORT_MARKER, json.dumps(admin.json()))

        self.assertEqual(self.client_for(self.other_user).get(url).status_code, 404)

    def test_call_register_for_store_roles_omits_customer_details(self):
        rows = self.client_for(self.store).get("/api/calls/").json()["calls"]
        self.assertTrue(rows)
        for row in rows:
            self.assertFalse({"complaint_text", "contact_name", "customer__name", "technician__full_name"} & set(row))
        asset_calls = self.client_for(self.store).get(f"/api/assets/{self.asset.id}/calls/").json()["calls"]
        self.assertNotIn(REPORT_MARKER, json.dumps(asset_calls))
        self.assertNotIn("Synthetic Contact", json.dumps(asset_calls))

    def test_ticket_preview_never_gives_customers_the_technician_report(self):
        conv = CustomerConversation.objects.create(tenant=self.tenant, customer=self.customer, asset=self.asset,
                                                   site=self.site, status="ACTIVE")
        from core.models import CustomerMessage
        CustomerMessage.objects.create(tenant=self.tenant, conversation=conv, sender_role="customer",
                                       message_text="The unit is not cooling.")
        url = f"/api/customer-support/conversations/{conv.conversation_uuid}/ticket/"
        preview = self.client_for(self.customer_user).get(url).json()["preview"]
        self.assertNotIn("description", preview)
        self.assertNotIn("sections", preview)
        self.assertEqual(preview["complaint"], "The unit is not cooling.")
        self.assertEqual(self.client_for(self.other_user).get(url).status_code, 404)
        staff = self.client_for(self.admin).get(url).json()["preview"]
        self.assertIn("description", staff)
        self.assertIn("sections", staff)
        # The demo customer gateway serves the same restricted preview.
        demo = self.as_customer("get", f"/demo-api/customer-support/conversations/{conv.conversation_uuid}/ticket/")
        self.assertNotIn("sections", demo.json()["preview"])


class OriginalDocumentGateTests(VRFConversationGatewayTests):
    """A refused original-document gate must not be bypassed by later answer paths."""

    def _blocked_rag(self, **kwargs):
        from core.services.rag import NO_AUTHORIZED_ORIGINAL_DOCUMENTATION
        return {"answer": "", "references": [], "reason_code": NO_AUTHORIZED_ORIGINAL_DOCUMENTATION,
                "retrieval_backend": "none", "reranker_applied": False}

    def test_generic_cause_scan_cannot_reintroduce_claims_after_a_refusal(self):
        uuid = self.create()
        with mock.patch("core.services.rag.ask_rag", side_effect=self._blocked_rag):
            answer, _ = self.ask(uuid, "what could be the possible causes for it to not work")
        self.assertNotEqual(answer["response_type"], "DIAGNOSTIC_GUIDANCE", answer.get("answer"))
        presentation = answer.get("presentation") or {}
        self.assertFalse(presentation.get("citations"))
        self.assertFalse([c for c in presentation.get("supporting_claims") or [] if c.get("evidence_ids")])

    def test_explaining_previously_listed_causes_is_regated(self):
        uuid = self.create()
        overview, _ = self.ask(uuid, "What are the possible causes of a VRF outdoor module not working?")
        self.assertEqual(overview["response_type"], "DIAGNOSTIC_GUIDANCE")
        with mock.patch("core.services.rag.original_documentation_permits", return_value=False), \
                mock.patch("core.services.rag.ask_rag", side_effect=self._blocked_rag):
            explained, _ = self.ask(uuid, "Explain each of these causes.")
        self.assertFalse((explained.get("presentation") or {}).get("citations"), explained.get("answer"))
