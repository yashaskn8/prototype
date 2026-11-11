from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient

from core.models import (
    Asset,
    Branch,
    Customer,
    CustomerInteraction,
    CustomerConversation,
    CustomerMessage,
    KnowledgeChunk,
    KnowledgeDocument,
    OperationalZone,
    RecoveryPassport,
    DiagnosticSession,
    ServiceCall,
    Site,
    StaffProfile,
    Tenant,
    TenantMembership,
)


@override_settings(SERVY_CUSTOMER_DATA_MODE="test_isolated")
class TechnicianHandoffFlowTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="Handoff Tenant", slug="handoff-tenant")
        self.other_tenant = Tenant.objects.create(name="Other Tenant", slug="other-handoff-tenant")
        self.branch = Branch.objects.create(tenant=self.tenant, name="North Branch")
        self.zone = OperationalZone.objects.create(tenant=self.tenant, branch=self.branch, name="North Zone")
        self.other_zone = OperationalZone.objects.create(tenant=self.tenant, name="South Zone")
        self.customer = Customer.objects.create(tenant=self.tenant, name="North Customer")
        self.other_customer = Customer.objects.create(tenant=self.tenant, name="Other Customer")
        self.site = Site.objects.create(tenant=self.tenant, customer=self.customer, zone=self.zone, name="Plant North")
        self.asset = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=self.site,
            name="Pump 1", asset_code="P-1", model_number="Model-P",
        )
        self.customer_user = User.objects.create_user("handoff_customer", password="test-pass")
        TenantMembership.objects.create(
            user=self.customer_user, tenant=self.tenant, role="customer", customer=self.customer,
        )
        self.technician_user = User.objects.create_user("handoff_technician", password="test-pass")
        TenantMembership.objects.create(user=self.technician_user, tenant=self.tenant, role="technician")
        self.technician = StaffProfile.objects.create(
            tenant=self.tenant, user=self.technician_user, role="technician", full_name="North Technician",
            branch=self.branch, zone=self.zone,
        )
        self.out_of_scope_user = User.objects.create_user("south_technician", password="test-pass")
        TenantMembership.objects.create(user=self.out_of_scope_user, tenant=self.tenant, role="technician")
        self.out_of_scope_tech = StaffProfile.objects.create(
            tenant=self.tenant, user=self.out_of_scope_user, role="technician", full_name="South Technician",
            zone=self.other_zone,
        )
        self.manager_user = User.objects.create_user("handoff_manager", password="test-pass")
        TenantMembership.objects.create(user=self.manager_user, tenant=self.tenant, role="manager")
        self.other_customer_user = User.objects.create_user("other_customer", password="test-pass")
        TenantMembership.objects.create(
            user=self.other_customer_user, tenant=self.tenant, role="customer", customer=self.other_customer,
        )

        self.source_doc = KnowledgeDocument.objects.create(
            tenant=self.tenant,
            customer=self.customer,
            asset=self.asset,
            title="Approved Customer Reference",
            content_text="Authorized source body is not returned in ticket detail.",
            index_status="INDEXED",
            is_rag_enabled=True,
            is_confidential=False,
        )
        self.source_chunk = KnowledgeChunk.objects.create(
            tenant=self.tenant,
            document=self.source_doc,
            chunk_index=0,
            heading="Indicator notes",
            text="Authorized source body is not returned in ticket detail.",
        )
        self.confidential_doc = KnowledgeDocument.objects.create(
            tenant=self.tenant,
            customer=self.customer,
            asset=self.asset,
            title="Confidential Technician Notes",
            content_text="Confidential body must not be exposed.",
            index_status="INDEXED",
            is_rag_enabled=True,
            is_confidential=True,
        )
        self.confidential_chunk = KnowledgeChunk.objects.create(
            tenant=self.tenant,
            document=self.confidential_doc,
            chunk_index=0,
            text="Confidential body must not be exposed.",
        )
        foreign_doc = KnowledgeDocument.objects.create(
            tenant=self.other_tenant,
            title="Foreign Tenant Document",
            content_text="Foreign tenant body must not be exposed.",
            index_status="INDEXED",
            is_rag_enabled=True,
        )
        self.foreign_chunk = KnowledgeChunk.objects.create(
            tenant=self.other_tenant,
            document=foreign_doc,
            chunk_index=0,
            text="Foreign tenant body must not be exposed.",
        )
        self.conversation = CustomerConversation.objects.create(
            tenant=self.tenant,
            customer=self.customer,
            site=self.site,
            asset=self.asset,
            created_by=self.customer_user,
            title="Pump does not start",
        )
        CustomerMessage.objects.create(
            tenant=self.tenant,
            conversation=self.conversation,
            sender_role="customer",
            message_text="The pump stops after a few seconds. I saw indicator E12.",
        )
        CustomerMessage.objects.create(
            tenant=self.tenant,
            conversation=self.conversation,
            sender_role="assistant",
            message_text="Please note the exact indicator state for the technician.",
            structured_contract={"response_type": "CLARIFICATION", "clarification_question": "What does the indicator show now?"},
            source_refs=[
                {
                    "document_id": self.source_doc.id,
                    "chunk_id": self.source_chunk.id,
                    "title": "Customer-supplied false title",
                    "excerpt": "This excerpt must not be returned.",
                },
                {"document_id": self.confidential_doc.id, "chunk_id": self.confidential_chunk.id, "title": "Confidential"},
                {"document_id": self.foreign_chunk.document_id, "chunk_id": self.foreign_chunk.id, "title": "Foreign"},
            ],
        )

    def login(self, user):
        self.client = APIClient()
        self.client.force_authenticate(user=user)

    def create_ticket(self):
        self.login(self.customer_user)
        response = self.client.post(
            reverse("api_customer_conversation_ticket", args=[self.conversation.conversation_uuid]),
            {"confirm_ticket_creation": True},
            format="json",
        )
        return response

    def test_customer_confirmation_creates_one_authoritative_ticket_and_technician_can_acknowledge(self):
        self.login(self.customer_user)
        url = reverse("api_customer_conversation_ticket", args=[self.conversation.conversation_uuid])
        unconfirmed = self.client.post(url, {}, format="json")
        self.assertEqual(unconfirmed.status_code, 400)

        created = self.client.post(url, {"confirm_ticket_creation": True}, format="json")
        self.assertEqual(created.status_code, 200)
        ticket_data = created.data["ticket"]
        call = ServiceCall.objects.get(pk=ticket_data["id"])
        self.assertEqual(call.tenant_id, self.tenant.id)
        self.assertEqual(call.customer_id, self.customer.id)
        self.assertEqual(call.asset_id, self.asset.id)
        self.assertEqual(call.complaint_text, "The pump stops after a few seconds. I saw indicator E12.")
        self.assertEqual(call.technician_id, self.technician.id)
        self.assertEqual(ticket_data["handoff_state"], "assigned")
        self.assertFalse(ticket_data["technician_acknowledged"])
        self.assertEqual(ServiceCall.objects.filter(tenant=self.tenant).count(), 1)
        session = call.diagnostic_session
        RecoveryPassport.objects.create(
            tenant=self.tenant,
            session=session,
            service_call=call,
            asset=self.asset,
            complaint=call.complaint_text,
            do_not_repeat_items=["Customer confirmed the exterior indicator was observed."],
        )

        duplicate = self.client.post(url, {"confirm_ticket_creation": True}, format="json")
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(duplicate.data["ticket"]["id"], call.id)
        self.assertEqual(ServiceCall.objects.filter(tenant=self.tenant).count(), 1)

        self.login(self.technician_user)
        inbox = self.client.get(reverse("api_technician_inbox"))
        self.assertEqual(inbox.status_code, 200)
        self.assertEqual([item["id"] for item in inbox.data["tickets"]], [call.id])

        detail = self.client.get(reverse("api_technician_inbox_detail", args=[call.id]))
        self.assertEqual(detail.status_code, 200)
        ticket = detail.data["ticket"]
        self.assertEqual(ticket["conversation"][0]["message_text"], "The pump stops after a few seconds. I saw indicator E12.")
        self.assertEqual(ticket["conversation"][0]["confirmation"], "reported by customer")
        self.assertIn("not confirmed", ticket["conversation"][1]["confirmation"])
        self.assertEqual(ticket["outstanding_information"], ["What does the indicator show now?"])
        self.assertEqual(ticket["confirmed_troubleshooting"], ["Customer confirmed the exterior indicator was observed."])
        self.assertEqual(ticket["supporting_documents"], [{
            "document_id": self.source_doc.id,
            "chunk_id": self.source_chunk.id,
            "title": "Approved Customer Reference",
            "heading": "Indicator notes",
        }])
        serialized = str(detail.data)
        self.assertNotIn("This excerpt must not be returned", serialized)
        self.assertNotIn("Authorized source body", serialized)
        self.assertNotIn("Customer-supplied false title", serialized)
        self.assertNotIn("Confidential body", serialized)
        self.assertNotIn("Foreign tenant body", serialized)

        ack_url = reverse("api_technician_ticket_acknowledge", args=[call.id])
        acknowledged = self.client.post(ack_url, {}, format="json")
        self.assertEqual(acknowledged.status_code, 200)
        call.refresh_from_db()
        self.assertEqual(call.status, "assigned")
        self.assertEqual(call.updates.filter(status="acknowledged", author=self.technician).count(), 1)
        self.assertEqual(acknowledged.data["ticket"]["handoff_state"], "acknowledged")

        duplicate_ack = self.client.post(ack_url, {}, format="json")
        self.assertEqual(duplicate_ack.status_code, 200)
        self.assertEqual(call.updates.filter(status="acknowledged", author=self.technician).count(), 1)

        self.login(self.customer_user)
        customer_ticket = self.client.get(url)
        self.assertEqual(customer_ticket.status_code, 200)
        self.assertTrue(customer_ticket.data["ticket"]["technician_acknowledged"])
        self.assertEqual(customer_ticket.data["ticket"]["handoff_state"], "acknowledged")
        list_response = self.client.get(reverse("api_customer_tickets"))
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.data["tickets"][0]["handoff_state"], "acknowledged")

    def test_customer_and_out_of_scope_technician_cannot_access_ticket_inbox(self):
        created = self.create_ticket()
        call_id = created.data["ticket"]["id"]
        assigned_call = ServiceCall.objects.get(pk=call_id)
        self.assertEqual(assigned_call.technician_id, self.technician.id)

        self.login(self.technician_user)
        self.assertEqual(self.client.get(reverse("api_call_detail", args=[call_id])).status_code, 200)

        self.login(self.customer_user)
        self.assertEqual(self.client.get(reverse("api_technician_inbox")).status_code, 403)

        self.login(self.out_of_scope_user)
        self.assertEqual(self.client.get(reverse("api_technician_inbox")).data["tickets"], [])
        detail = self.client.get(reverse("api_technician_inbox_detail", args=[call_id]))
        self.assertEqual(detail.status_code, 404)
        ack = self.client.post(reverse("api_technician_ticket_acknowledge", args=[call_id]), {}, format="json")
        self.assertEqual(ack.status_code, 404)
        conversation_url = reverse("api_customer_conversation_detail", args=[self.conversation.conversation_uuid])
        self.assertEqual(self.client.get(conversation_url).status_code, 404)
        self.assertEqual(self.client.get(reverse("api_customer_conversation_messages", args=[self.conversation.conversation_uuid])).status_code, 404)
        self.assertEqual(self.client.get(reverse("api_customer_conversation_ticket", args=[self.conversation.conversation_uuid])).status_code, 404)
        self.assertEqual(
            self.client.post(
                reverse("api_customer_conversation_ticket", args=[self.conversation.conversation_uuid]),
                {"confirm_ticket_creation": True}, format="json",
            ).status_code,
            403,
        )
        self.assertEqual(self.client.get(reverse("api_calls")).data["calls"], [])
        assigned_to_other = self.client.get(reverse("api_call_detail", args=[call_id]))
        self.assertEqual(assigned_to_other.status_code, 404)
        self.assertEqual(assigned_to_other.data, {"detail": "Call not found."})
        self.assertEqual(self.client.get(reverse("api_asset_calls", args=[self.asset.id])).data["calls"], [])
        copilot = self.client.post(reverse("api_engineer_copilot_query"), {
            "call_id": call_id,
            "question": "What should I inspect?",
        }, format="json")
        self.assertEqual(copilot.status_code, 404)
        self.assertEqual(copilot.data, {"detail": "Service call not found in this tenant."})

        # An open, unassigned ticket is visible only inside its dispatch zone.
        south_site = Site.objects.create(
            tenant=self.tenant, customer=self.other_customer, zone=self.other_zone, name="Plant South",
        )
        south_call = ServiceCall.objects.create(
            tenant=self.tenant, servy_id=902, complaint_type="South zone issue",
            complaint_text="Reported issue in the south zone.", customer=self.other_customer,
            site=south_site, zone=self.other_zone, status="open",
        )
        self.login(self.technician_user)
        north_inbox = self.client.get(reverse("api_technician_inbox"))
        self.assertNotIn(south_call.id, [item["id"] for item in north_inbox.data["tickets"]])
        out_of_zone = self.client.get(reverse("api_call_detail", args=[south_call.id]))
        self.assertEqual(out_of_zone.status_code, 404)
        self.assertEqual(out_of_zone.data, {"detail": "Call not found."})

        self.login(self.out_of_scope_user)
        south_inbox = self.client.get(reverse("api_technician_inbox"))
        self.assertIn(south_call.id, [item["id"] for item in south_inbox.data["tickets"]])
        self.assertEqual(self.client.get(reverse("api_call_detail", args=[south_call.id])).status_code, 200)

    def test_technician_cannot_act_as_customer_in_pre_ticket_support_workflow(self):
        self.login(self.technician_user)
        context = self.client.get(reverse("api_customer_support_context"), {"customer_id": self.customer.id})
        self.assertEqual(context.status_code, 403)
        query = self.client.post(reverse("api_customer_support_query"), {
            "customer_id": self.customer.id,
            "site_id": self.site.id,
            "asset_id": self.asset.id,
            "question": "The pump will not start.",
        }, format="json")
        self.assertEqual(query.status_code, 403)

        interaction = CustomerInteraction.objects.create(
            tenant=self.tenant,
            created_by=self.customer_user,
            customer=self.customer,
            site=self.site,
            asset=self.asset,
            question="The pump will not start.",
            answer="Please contact service.",
        )
        resolve = self.client.post(
            reverse("api_customer_support_resolve", args=[interaction.id]),
            {"customer_reports_resolved": True}, format="json",
        )
        escalate = self.client.post(
            reverse("api_customer_support_escalate", args=[interaction.id]),
            {"confirm_ticket_creation": True}, format="json",
        )
        self.assertEqual(resolve.status_code, 403)
        self.assertEqual(escalate.status_code, 403)
        interaction.refresh_from_db()
        self.assertFalse(interaction.resolved)
        self.assertIsNone(interaction.escalated_call_id)
        self.assertEqual(ServiceCall.objects.filter(tenant=self.tenant).count(), 0)
        self.assertEqual(DiagnosticSession.objects.filter(tenant=self.tenant).count(), 0)

    def test_customer_cannot_read_or_ticket_a_conversation_with_another_customers_asset(self):
        other_customer_asset = Asset.objects.create(
            tenant=self.tenant, customer=self.other_customer,
            name="Other customer pump", asset_code="OTHER-CUSTOMER-P-1", model_number="Other customer",
        )
        malformed = CustomerConversation(
            tenant=self.tenant,
            customer=self.customer,
            site=self.site,
            asset=other_customer_asset,
            created_by=self.customer_user,
            title="Mismatched asset conversation",
        )
        # Bypass model.clean() only to represent a legacy/corrupted row and
        # verify API reads still fail closed if that invariant is violated.
        CustomerConversation.objects.bulk_create([malformed])
        self.login(self.customer_user)
        detail = self.client.get(reverse("api_customer_conversation_detail", args=[malformed.conversation_uuid]))
        ticket = self.client.post(
            reverse("api_customer_conversation_ticket", args=[malformed.conversation_uuid]),
            {"confirm_ticket_creation": True}, format="json",
        )
        listing = self.client.get(reverse("api_customer_conversations"))
        self.assertEqual(detail.status_code, 404)
        self.assertEqual(ticket.status_code, 404)
        self.assertNotIn(malformed.id, [item["id"] for item in listing.data["conversations"]])
        self.assertEqual(ServiceCall.objects.filter(tenant=self.tenant).count(), 0)

    def test_ticket_requires_a_customer_report_in_the_saved_conversation(self):
        empty = CustomerConversation.objects.create(
            tenant=self.tenant,
            customer=self.customer,
            site=self.site,
            asset=self.asset,
            created_by=self.customer_user,
            title="Unreported issue",
        )
        self.login(self.customer_user)
        response = self.client.post(
            reverse("api_customer_conversation_ticket", args=[empty.conversation_uuid]),
            {"confirm_ticket_creation": True}, format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(ServiceCall.objects.filter(tenant=self.tenant).count(), 0)
        empty.refresh_from_db()
        self.assertIsNone(empty.linked_service_call_id)

    def test_cross_tenant_ticket_is_not_visible_or_acknowledgeable(self):
        other_call = ServiceCall.objects.create(
            tenant=self.other_tenant,
            servy_id=1,
            complaint_type="Other tenant issue",
            complaint_text="Private other-tenant report",
            customer=Customer.objects.create(tenant=self.other_tenant, name="Other tenant customer"),
            status="open",
        )
        self.login(self.technician_user)
        response = self.client.get(reverse("api_technician_inbox_detail", args=[other_call.id]))
        self.assertEqual(response.status_code, 404)
        ack = self.client.post(reverse("api_technician_ticket_acknowledge", args=[other_call.id]), {}, format="json")
        self.assertEqual(ack.status_code, 404)
        self.assertIsNone(other_call.technician_id)

    def test_missing_queue_recipient_keeps_ticket_created_without_claiming_dispatch(self):
        self.technician.is_active = False
        self.technician.save(update_fields=["is_active"])
        self.out_of_scope_tech.is_active = False
        self.out_of_scope_tech.save(update_fields=["is_active"])

        response = self.create_ticket()
        self.assertEqual(response.status_code, 200)
        call = ServiceCall.objects.get(pk=response.data["ticket"]["id"])
        self.assertEqual(call.status, "open")
        self.assertIsNone(call.technician_id)
        self.assertEqual(response.data["ticket"]["handoff_state"], "awaiting_assignment")
        self.assertEqual(ServiceCall.objects.filter(tenant=self.tenant).count(), 1)

        self.login(self.manager_user)
        inbox = self.client.get(reverse("api_technician_inbox"))
        self.assertEqual(inbox.status_code, 200)
        self.assertEqual(inbox.data["tickets"][0]["id"], call.id)

    def test_missing_conversation_data_is_returned_as_empty_and_not_fabricated(self):
        call = ServiceCall.objects.create(
            tenant=self.tenant,
            servy_id=901,
            complaint_type="Manual service call",
            complaint_text="Customer reported a vibration.",
            customer=self.customer,
            site=self.site,
            asset=self.asset,
            zone=self.zone,
            technician=self.technician,
            status="assigned",
        )
        self.login(self.technician_user)
        response = self.client.get(reverse("api_technician_inbox_detail", args=[call.id]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["ticket"]["conversation"], [])
        self.assertEqual(response.data["ticket"]["supporting_documents"], [])
        self.assertEqual(response.data["ticket"]["confirmed_troubleshooting"], [])
