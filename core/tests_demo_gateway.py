"""Access-code demo gateways over the official APIs (synthetic data, throwaway test DB)."""
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth.models import User
from django.core.cache import cache
from django.db import connection
from django.test import SimpleTestCase, TestCase, override_settings

from core.models import (
    Asset, Brand, CallUpdate, Customer, CustomerConversation, KnowledgeDocument, Product, ProductCategory,
    ProductDomain, ServiceCall, ServiceCallAttachment, Site, StaffProfile, Tenant, TenantMembership)
from core.services.indexing import index_document

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
CUSTOMER_CODE = "c" * 48
TECH_CODE = "t" * 48


def _upload(name, content, content_type):
    from django.core.files.uploadedfile import SimpleUploadedFile
    return SimpleUploadedFile(name, content, content_type=content_type)


class GatewayTestBase(TestCase):
    def setUp(self):
        cache.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "customer.txt").write_text(CUSTOMER_CODE, encoding="ascii")
        (root / "tech.txt").write_text(TECH_CODE, encoding="ascii")
        overrides = override_settings(
            MEDIA_ROOT=str(root / "media"), SERVY_DEMO_ACCESS_FILE=str(root / "customer.txt"),
            SERVY_TECHNICIAN_ACCESS_FILE=str(root / "tech.txt"), SERVY_CUSTOMER_DATA_MODE="demo_only",
            SERVY_LLM_PROVIDER="extractive", SERVY_RAG_MIN_SCORE=0.0)
        overrides.enable()
        self.addCleanup(overrides.disable)
        patcher = mock.patch("core.demo_gateway.ensure_isolation", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.tenant = Tenant.objects.create(name="Servy Local Development", slug="servy-local-dev-test")
        self.other_tenant = Tenant.objects.create(name="Other", slug="other-dev-test")
        domain = ProductDomain.objects.create(tenant=self.tenant, name="HVAC")
        category = ProductCategory.objects.create(tenant=self.tenant, domain=domain, name="AC")
        brand = Brand.objects.create(tenant=self.tenant, name="B")
        product = Product.objects.create(tenant=self.tenant, category=category, brand=brand, name="Split AC Unit 2T")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Synthetic Development Customer",
                                                contact_name="Synthetic Contact", phone="+1 555 0100",
                                                email="dev-customer@example.invalid")
        self.site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Site")
        self.asset = Asset.objects.create(tenant=self.tenant, customer=self.customer, site=self.site,
                                          product=product, name="Split AC Unit 2T", asset_code="DEV-AST-059")
        self.other_customer = Customer.objects.create(tenant=self.tenant, name="Other customer", contact_name="O",
                                                      phone="+1 555 0101")
        other_site = Site.objects.create(tenant=self.tenant, customer=self.other_customer, name="Other site")
        self.other_asset = Asset.objects.create(tenant=self.tenant, customer=self.other_customer, site=other_site,
                                                product=product, name="Other AC", asset_code="DEV-OTHER")
        doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=product, asset=self.asset, title="Split AC Unit 2T - Troubleshooting",
            source_type="text", is_rag_enabled=True,
            content_text="3. SYMPTOM: POOR COOLING\nPossible causes:\n- Dirty filter or coil.\n- Blocked outdoor unit.")
        index_document(doc)
        self.customer_user = User.objects.create(username="dev_customer")
        TenantMembership.objects.create(user=self.customer_user, tenant=self.tenant, role="customer",
                                        customer=self.customer)
        self.tech_user = User.objects.create(username="dev_technician")
        TenantMembership.objects.create(user=self.tech_user, tenant=self.tenant, role="technician")
        self.tech = StaffProfile.objects.create(tenant=self.tenant, user=self.tech_user,
                                                full_name="Synthetic Technician", role="technician")

    def as_customer(self, method, path, code=CUSTOMER_CODE, **kwargs):
        with override_settings(ROOT_URLCONF="servy_rag.demo_customer_urls"):
            return getattr(self.client, method)(path, HTTP_HOST="127.0.0.1",
                                                HTTP_AUTHORIZATION=f"Demo {code}" if code else "", **kwargs)

    def as_technician(self, method, path, code=TECH_CODE, **kwargs):
        with override_settings(ROOT_URLCONF="servy_rag.demo_technician_urls"):
            return getattr(self.client, method)(path, HTTP_HOST="127.0.0.1",
                                                HTTP_AUTHORIZATION=f"Technician {code}" if code else "", **kwargs)

    def start_conversation(self, message="What can cause poor cooling?"):
        r = self.as_customer("post", "/demo-api/customer-support/conversations/",
                          data={"asset_id": self.asset.id, "message": message}, content_type="application/json")
        self.assertEqual(r.status_code, 201, r.content[:300])
        return r.json()["conversation_uuid"]

    def create_ticket(self, uuid, files=(), **fields):
        data = {"confirm_ticket_creation": "true", "call_type": "Service", "priority": "normal", **fields}
        if files:
            data["attachments"] = list(files)
        return self.as_customer("post", f"/demo-api/customer-support/conversations/{uuid}/ticket/", data=data)


class CustomerGatewayTests(GatewayTestBase):
    def test_access_code_is_required_and_checked(self):
        self.assertEqual(self.as_customer("get", "/demo-api/inventory/", code="").status_code, 401)
        self.assertEqual(self.as_customer("get", "/demo-api/inventory/", code="x" * 48).status_code, 401)
        self.assertEqual(self.as_customer("get", "/demo-api/inventory/", code=TECH_CODE).status_code, 401)
        with override_settings(ROOT_URLCONF="servy_rag.demo_customer_urls"):
            remote = self.client.get("/demo-api/inventory/", HTTP_HOST="example.com",
                                     HTTP_AUTHORIZATION=f"Demo {CUSTOMER_CODE}")
        self.assertIn(remote.status_code, (400, 401, 403))

    def test_inventory_lists_only_the_synthetic_customers_assets(self):
        r = self.as_customer("get", "/demo-api/inventory/")
        self.assertEqual(r.status_code, 200)
        assets = r.json()["assets"]
        self.assertEqual([a["asset_code"] for a in assets], ["DEV-AST-059"])
        self.assertTrue(assets[0]["manuals_ready"])

    def test_conversation_persists_and_resumes(self):
        uuid = self.start_conversation()
        r = self.as_customer("post", f"/demo-api/customer-support/conversations/{uuid}/messages/",
                          data={"message": "The indoor fan is running."}, content_type="application/json")
        self.assertEqual(r.status_code, 200)
        listed = self.as_customer("get", "/demo-api/customer-support/conversations/").json()
        self.assertIn(uuid, [c["conversation_uuid"] for c in listed["conversations"]])
        messages = self.as_customer("get", f"/demo-api/customer-support/conversations/{uuid}/messages/").json()["messages"]
        self.assertEqual([m["sender_role"] for m in messages], ["customer", "assistant", "customer", "assistant"])

    def test_other_customers_asset_is_refused(self):
        r = self.as_customer("post", "/demo-api/customer-support/conversations/",
                          data={"asset_id": self.other_asset.id, "message": "help"}, content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_ticket_with_zero_one_and_ten_attachments(self):
        for count in (0, 1, 10):
            with self.subTest(count=count):
                uuid = self.start_conversation(f"Poor cooling, case {count}")
                files = [_upload(f"p{i}.png", PNG + bytes([i]), "image/png") for i in range(count)]
                r = self.create_ticket(uuid, files)
                self.assertEqual(r.status_code, 201, r.content[:300])
                body = r.json()
                self.assertEqual(body["mode"], "SERVY_DEV_MYSQL")
                self.assertFalse(body["duplicate"])
                self.assertEqual(len(body["ticket"]["attachments"]), count)
                call = ServiceCall.objects.get(pk=body["ticket"]["id"])
                self.assertEqual(call.attachments.count(), count)
                self.assertIn("Call Type: Service", call.technician_notes)

    def test_eleven_attachments_and_invalid_files_create_no_ticket(self):
        cases = (
            [_upload(f"p{i}.png", PNG, "image/png") for i in range(11)],
            [_upload("run.exe", b"MZ\x90\x00", "application/x-msdownload")],
            [_upload("fake.png", b"MZnot a png", "image/png")],
            [_upload("big.pdf", b"%PDF-" + b"0" * (25 * 1024 * 1024), "application/pdf")],
        )
        for files in cases:
            with self.subTest(first=files[0].name, count=len(files)):
                uuid = self.start_conversation("Poor cooling")
                r = self.create_ticket(uuid, files)
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.json()["code"], "attachment_invalid")
                self.assertIsNone(CustomerConversation.objects.get(conversation_uuid=uuid).linked_service_call_id)
        self.assertEqual(ServiceCall.objects.count(), 0)
        self.assertEqual(ServiceCallAttachment.objects.count(), 0)

    def test_repeated_confirmation_returns_the_same_ticket_without_new_files(self):
        uuid = self.start_conversation()
        first = self.create_ticket(uuid, [_upload("a.png", PNG, "image/png")]).json()
        second = self.create_ticket(uuid, [_upload("b.png", PNG, "image/png")])
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.json()["duplicate"])
        self.assertEqual(second.json()["ticket"]["id"], first["ticket"]["id"])
        self.assertEqual(ServiceCall.objects.count(), 1)
        self.assertEqual(ServiceCallAttachment.objects.count(), 1)

    def test_confirmation_is_required(self):
        uuid = self.start_conversation()
        r = self.as_customer("post", f"/demo-api/customer-support/conversations/{uuid}/ticket/",
                          data={"confirm_ticket_creation": "false"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(ServiceCall.objects.count(), 0)

    def test_customer_downloads_only_their_own_attachments(self):
        uuid = self.start_conversation()
        body = self.create_ticket(uuid, [_upload("a.png", PNG, "image/png")]).json()
        att = body["ticket"]["attachments"][0]
        r = self.as_customer("get", f"/demo-api/tickets/{body['ticket']['id']}/attachments/{att['id']}/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(b"".join(r.streaming_content), PNG)
        other = ServiceCall.objects.create(tenant=self.tenant, servy_id=99999, customer=self.other_customer,
                                           complaint_text="x")
        self.assertEqual(self.as_customer("get", f"/demo-api/tickets/{other.id}/attachments/{att['id']}/").status_code, 404)


class TechnicianGatewayTests(GatewayTestBase):
    def _ticket(self, files=()):
        uuid = self.start_conversation()
        return self.create_ticket(uuid, files).json()["ticket"]

    def test_access_code_is_required_and_customer_code_is_refused(self):
        self.assertEqual(self.as_technician("get", "/tech-api/technician/inbox/", code="").status_code, 401)
        self.assertEqual(self.as_technician("get", "/tech-api/technician/inbox/", code=CUSTOMER_CODE).status_code, 401)
        # The technician gateway does not expose customer endpoints at all.
        self.assertEqual(self.as_technician("get", "/demo-api/inventory/").status_code, 404)

    def test_ticket_from_customer_is_visible_with_description_attachments_and_context(self):
        ticket = self._ticket([_upload("a.png", PNG, "image/png"),
                               _upload("r.pdf", b"%PDF-1.4 synthetic", "application/pdf")])
        inbox = self.as_technician("get", "/tech-api/technician/inbox/").json()["tickets"]
        self.assertIn(ticket["id"], [t["id"] for t in inbox])
        detail = self.as_technician("get", f"/tech-api/technician/inbox/{ticket['id']}/").json()["ticket"]
        self.assertIn("1. Customer-reported problem", detail["description"])
        self.assertEqual(len(detail["attachments"]), 2)
        self.assertTrue(detail["conversation"])
        att = detail["attachments"][1]
        r = self.as_technician("get", f"/tech-api/technician/inbox/{ticket['id']}/attachments/{att['id']}/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(b"".join(r.streaming_content), b"%PDF-1.4 synthetic")
        self.assertEqual(r["X-Content-Type-Options"], "nosniff")

    def test_status_updates_and_notes_persist_as_call_updates(self):
        ticket = self._ticket()
        description = ServiceCall.objects.get(pk=ticket["id"]).technician_notes
        r = self.as_technician("post", f"/tech-api/technician/inbox/{ticket['id']}/update/",
                            data={"status": "in_progress", "note": "On site, checking filter."},
                            content_type="application/json")
        self.assertEqual(r.status_code, 200, r.content[:300])
        call = ServiceCall.objects.get(pk=ticket["id"])
        self.assertEqual(call.status, "in_progress")
        self.assertEqual(call.technician_id, self.tech.id)
        self.assertEqual(call.technician_notes, description)          # generated description untouched
        self.assertTrue(CallUpdate.objects.filter(service_call=call, note="On site, checking filter.").exists())
        detail = self.as_technician("get", f"/tech-api/technician/inbox/{ticket['id']}/").json()["ticket"]
        self.assertIn("On site, checking filter.", [u["note"] for u in detail["updates"]])
        bad = self.as_technician("post", f"/tech-api/technician/inbox/{ticket['id']}/update/",
                              data={"status": "deleted"}, content_type="application/json")
        self.assertEqual(bad.status_code, 400)
        self.as_technician("post", f"/tech-api/technician/inbox/{ticket['id']}/update/", data={"status": "closed"},
                        content_type="application/json")
        after_close = self.as_technician("post", f"/tech-api/technician/inbox/{ticket['id']}/update/",
                                      data={"note": "late"}, content_type="application/json")
        self.assertEqual(after_close.status_code, 409)

    def test_other_tenant_and_other_technicians_tickets_are_hidden(self):
        other_customer = Customer.objects.create(tenant=self.other_tenant, name="X")
        hidden = ServiceCall.objects.create(tenant=self.other_tenant, servy_id=88888, customer=other_customer,
                                            complaint_text="x", status="open")
        self.assertEqual(self.as_technician("get", f"/tech-api/technician/inbox/{hidden.id}/").status_code, 404)
        other_tech_user = User.objects.create(username="other_tech")
        other_tech = StaffProfile.objects.create(tenant=self.tenant, user=other_tech_user, full_name="Other",
                                                 role="technician")
        ticket = self._ticket([_upload("a.png", PNG, "image/png")])
        ServiceCall.objects.filter(pk=ticket["id"]).update(technician=other_tech, status="assigned")
        self.assertEqual(self.as_technician("get", f"/tech-api/technician/inbox/{ticket['id']}/").status_code, 404)
        att = ServiceCallAttachment.objects.get(service_call_id=ticket["id"])
        self.assertEqual(self.as_technician(
            "get", f"/tech-api/technician/inbox/{ticket['id']}/attachments/{att.id}/").status_code, 404)


class GatewayIsolationCheckTests(TestCase):
    def test_gateway_isolation_check_against_the_live_test_database(self):
        if connection.vendor != "mysql" or connection.settings_dict["NAME"] != "test_servy_dev":
            self.skipTest("runs only on the isolated test_servy_dev database")
        import core.demo_gateway as gateway
        gateway._isolation_verified = False
        with override_settings(SERVY_DEMO_EXPECTED_DATABASE="test_servy_dev", SERVY_DEMO_EXPECTED_USER="servy_dev_test"):
            gateway.ensure_isolation()
        gateway._isolation_verified = False
        with override_settings(SERVY_DEMO_EXPECTED_DATABASE="servy_dev"):
            with self.assertRaises(gateway.GatewayUnavailable):
                gateway.ensure_isolation()
        gateway._isolation_verified = False


class GatewayConnectionPolicyTests(SimpleTestCase):
    def test_gateway_mode_disables_persistent_connections(self):
        """The threaded dev server must not keep one MySQL connection per finished thread."""
        from pathlib import Path
        from django.conf import settings as django_settings
        source = (Path(django_settings.BASE_DIR) / "servy_rag" / "settings.py").read_text(encoding="utf-8")
        block = source[source.index("if SERVY_DEMO_GATEWAY:"):]
        self.assertIn('DATABASES["default"]["CONN_MAX_AGE"] = 0', block)
