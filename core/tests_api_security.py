import io
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import (
    Tenant, Customer, Site, Asset, Product, ProductCategory, ProductDomain,
    Brand, ServiceCall, CustomerInteraction, StaffProfile, TenantMembership,
    KnowledgeDocument, InventoryItem, PartRequest, OperationalZone, Branch
)
from core.services.indexing import index_document


@override_settings(SERVY_LLM_PROVIDER="extractive", SERVY_RAG_MIN_SCORE=0.0)
class ApiSecurityAndPermissionsTests(TestCase):
    """
    Comprehensive tests for Servy API authorization, canonical contracts,
    ticket escalation idempotency, and the complete end-to-end flow.
    """

    def setUp(self):
        # 1. Tenants
        self.tenant_a = Tenant.objects.create(name="Tenant Alpha", slug="tenant-alpha", is_active=True)
        self.tenant_b = Tenant.objects.create(name="Tenant Beta", slug="tenant-beta", is_active=True)

        # 2. Branch & Zone for Tenant Alpha
        self.branch = Branch.objects.create(tenant=self.tenant_a, name="Alpha HQ", city="Springfield")
        self.zone = OperationalZone.objects.create(tenant=self.tenant_a, branch=self.branch, name="North Zone")

        # 3. Product Hierarchy for Tenant Alpha
        self.domain = ProductDomain.objects.create(tenant=self.tenant_a, name="Industrial")
        self.cat = ProductCategory.objects.create(tenant=self.tenant_a, domain=self.domain, name="Lifts")
        self.brand = Brand.objects.create(tenant=self.tenant_a, name="Aerolift")
        self.product = Product.objects.create(tenant=self.tenant_a, category=self.cat, brand=self.brand, name="Aerolift Hydraulic Lift")

        # 4. Customers & Sites
        self.customer_a1 = Customer.objects.create(tenant=self.tenant_a, name="Customer A1")
        self.customer_a2 = Customer.objects.create(tenant=self.tenant_a, name="Customer A2")
        self.site_a1 = Site.objects.create(tenant=self.tenant_a, customer=self.customer_a1, name="Plant 1", city="Springfield")
        self.site_a2 = Site.objects.create(tenant=self.tenant_a, customer=self.customer_a2, name="Plant 2", city="Shelbyville")

        # 5. Assets
        self.asset_a1 = Asset.objects.create(
            tenant=self.tenant_a, customer=self.customer_a1, site=self.site_a1, product=self.product,
            name="Lift-01", asset_code="AL-01", model_number="AL-Hydraulic", status="active"
        )
        self.asset_a2 = Asset.objects.create(
            tenant=self.tenant_a, customer=self.customer_a2, site=self.site_a2, product=self.product,
            name="Lift-02", asset_code="AL-02", model_number="AL-Hydraulic", status="active"
        )

        # 6. Inventory & Part Requests
        self.item = InventoryItem.objects.create(
            tenant=self.tenant_a, branch=self.branch, brand=self.brand, product=self.product,
            category="Hydraulics", spare_name="Hydraulic Valve Seal", ipn="HVS-990", quantity=25, unit="pcs"
        )

        # 7. Knowledge Documents
        self.doc_public = KnowledgeDocument.objects.create(
            tenant=self.tenant_a, product=self.product, asset=self.asset_a1, title="Aerolift Valve Troubleshooting",
            content_text="If hydraulic valve leaks or pressure drops, replace valve seal and check fluid levels.",
            is_confidential=False, is_rag_enabled=True, source_type="text"
        )
        index_document(self.doc_public)

        self.doc_confidential = KnowledgeDocument.objects.create(
            tenant=self.tenant_a, product=self.product, title="Aerolift Secret Calibration Blueprint",
            content_text="Confidential technician manual: bypass safety lock with master key 7741.",
            is_confidential=True, is_rag_enabled=True, source_type="text"
        )
        index_document(self.doc_confidential)

        # 8. Personas / Users
        # Superuser
        self.user_super = User.objects.create_superuser("super", password="SuperPassword!2026")

        # Admin
        self.user_admin = User.objects.create_user("admin_user", password="AdminPassword!2026")
        TenantMembership.objects.create(user=self.user_admin, tenant=self.tenant_a, role="admin", can_view_confidential=True)
        StaffProfile.objects.create(tenant=self.tenant_a, user=self.user_admin, full_name="Admin User", role="admin")

        # Manager
        self.user_manager = User.objects.create_user("manager_user", password="ManagerPassword!2026")
        TenantMembership.objects.create(user=self.user_manager, tenant=self.tenant_a, role="manager", can_view_confidential=True)
        StaffProfile.objects.create(tenant=self.tenant_a, user=self.user_manager, full_name="Manager User", role="manager")

        # Technician (non-confidential)
        self.user_tech = User.objects.create_user("tech_user", password="TechPassword!2026")
        TenantMembership.objects.create(user=self.user_tech, tenant=self.tenant_a, role="technician", can_view_confidential=False)
        self.tech_profile = StaffProfile.objects.create(
            tenant=self.tenant_a, user=self.user_tech, full_name="Tech Specialist", role="technician", branch=self.branch, zone=self.zone
        )

        # Technician (with confidential flag)
        self.user_tech_conf = User.objects.create_user("tech_conf", password="TechPassword!2026")
        TenantMembership.objects.create(user=self.user_tech_conf, tenant=self.tenant_a, role="technician", can_view_confidential=True)
        StaffProfile.objects.create(tenant=self.tenant_a, user=self.user_tech_conf, full_name="Senior Tech", role="technician")

        # Store Operator
        self.user_store = User.objects.create_user("store_user", password="StorePassword!2026")
        TenantMembership.objects.create(user=self.user_store, tenant=self.tenant_a, role="store_operator", can_view_confidential=False)
        StaffProfile.objects.create(tenant=self.tenant_a, user=self.user_store, full_name="Store Keeper", role="store_operator")

        # Store Admin
        self.user_store_admin = User.objects.create_user("store_admin_user", password="StoreAdminPassword!2026")
        TenantMembership.objects.create(user=self.user_store_admin, tenant=self.tenant_a, role="store_admin", can_view_confidential=False)
        StaffProfile.objects.create(tenant=self.tenant_a, user=self.user_store_admin, full_name="Store Manager", role="store_admin")

        # Customer 1
        self.user_cust1 = User.objects.create_user("cust1_user", password="CustPassword!2026")
        TenantMembership.objects.create(user=self.user_cust1, tenant=self.tenant_a, role="customer", customer=self.customer_a1)

        # Customer 2
        self.user_cust2 = User.objects.create_user("cust2_user", password="CustPassword!2026")
        TenantMembership.objects.create(user=self.user_cust2, tenant=self.tenant_a, role="customer", customer=self.customer_a2)

    # ------------------------------------------------------------------
    # 1. UsersListView Authorization (Admin & Manager & Superuser ONLY)
    # ------------------------------------------------------------------
    def test_users_list_view_permissions(self):
        url = reverse("api_users")

        # Anonymous -> 403 (DRF session auth returns 403 for unauthenticated)
        res = self.client.get(url)
        self.assertIn(res.status_code, [401, 403])

        # Customer -> 403
        self.client.login(username="cust1_user", password="CustPassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 403)

        # Technician -> 403
        self.client.login(username="tech_user", password="TechPassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 403)

        # Store Operator -> 403
        self.client.login(username="store_user", password="StorePassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 403)

        # Admin -> 200
        self.client.login(username="admin_user", password="AdminPassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertIn("users", res.data)
        self.assertTrue(any(u["full_name"] == "Tech Specialist" for u in res.data["users"]))

        # Manager -> 200
        self.client.login(username="manager_user", password="ManagerPassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)

        # Superuser (with active tenant in session) -> 200
        self.client.login(username="super", password="SuperPassword!2026")
        session = self.client.session
        session["active_tenant_id"] = self.tenant_a.id
        session.save()
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)

    # ------------------------------------------------------------------
    # 2. Canonical API Contract: { question: "..." }
    # ------------------------------------------------------------------
    def test_canonical_question_contract_customer_support(self):
        self.client.login(username="cust1_user", password="CustPassword!2026")
        url = reverse("api_customer_support_query")

        # Sending canonical 'question'
        res = self.client.post(url, {
            "site_id": self.site_a1.id,
            "asset_id": self.asset_a1.id,
            "question": "What should I check for hydraulic valve leaks?",
        }, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertIn("interaction_id", res.data)
        self.assertIn("valve", res.data["answer"].lower())

        # Backward compatibility: sending 'query'
        res2 = self.client.post(url, {
            "site_id": self.site_a1.id,
            "asset_id": self.asset_a1.id,
            "query": "What should I check for hydraulic valve leaks?",
        }, format="json")
        self.assertEqual(res2.status_code, 200)
        self.assertIn("interaction_id", res2.data)

    def test_canonical_question_contract_engineer_copilot(self):
        # Create a service call to query against
        call = ServiceCall.objects.create(
            tenant=self.tenant_a, servy_id=201, complaint_type="Leak", complaint_text="Hydraulic pressure loss",
            customer=self.customer_a1, site=self.site_a1, asset=self.asset_a1, status="open"
        )
        self.client.login(username="tech_user", password="TechPassword!2026")
        url = reverse("api_engineer_copilot_query")

        # Sending canonical 'question'
        res = self.client.post(url, {
            "call_id": call.id,
            "question": "Provide hydraulic valve repair procedure.",
        }, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertIn("answer", res.data)
        self.assertIn("references", res.data)

        # Backward compatibility: sending 'query'
        res2 = self.client.post(url, {
            "call_id": call.id,
            "query": "Provide hydraulic valve repair procedure.",
        }, format="json")
        self.assertEqual(res2.status_code, 200)

    # ------------------------------------------------------------------
    # 3. Canonical URLs and Escalation / Resolution IDOR Protection
    # ------------------------------------------------------------------
    def test_canonical_resolve_and_escalate_with_idor_protection(self):
        # Interaction for Customer 1
        interaction = CustomerInteraction.objects.create(
            tenant=self.tenant_a, created_by=self.user_cust1, customer=self.customer_a1,
            site=self.site_a1, asset=self.asset_a1, question="Hydraulic issue", answer="Fix valve"
        )

        resolve_url = reverse("api_customer_support_resolve", kwargs={"pk": interaction.id})
        escalate_url = reverse("api_customer_support_escalate", kwargs={"pk": interaction.id})

        # Customer 2 attempts to resolve or escalate Customer 1's interaction -> 404
        self.client.login(username="cust2_user", password="CustPassword!2026")
        res_res = self.client.post(resolve_url)
        self.assertEqual(res_res.status_code, 404)
        res_esc = self.client.post(escalate_url)
        self.assertEqual(res_esc.status_code, 404)

        # Customer 1 resolves own interaction -> 200
        self.client.login(username="cust1_user", password="CustPassword!2026")
        res_res_ok = self.client.post(resolve_url)
        self.assertEqual(res_res_ok.status_code, 200)
        self.assertEqual(res_res_ok.data["status"], "resolved")

    # ------------------------------------------------------------------
    # 4. Escalation Idempotency (Exactly ONE ServiceCall)
    # ------------------------------------------------------------------
    def test_escalation_idempotency_creates_exactly_one_call(self):
        interaction = CustomerInteraction.objects.create(
            tenant=self.tenant_a, created_by=self.user_cust1, customer=self.customer_a1,
            site=self.site_a1, asset=self.asset_a1, question="Valve leaking heavily", answer="Replace seal"
        )
        escalate_url = reverse("api_customer_support_escalate", kwargs={"pk": interaction.id})

        self.client.login(username="cust1_user", password="CustPassword!2026")

        # First call creates ServiceCall
        res1 = self.client.post(escalate_url)
        self.assertEqual(res1.status_code, 200)
        call_id_1 = res1.data["call_id"]

        # Second call returns the existing ServiceCall
        res2 = self.client.post(escalate_url)
        self.assertEqual(res2.status_code, 200)
        call_id_2 = res2.data["call_id"]

        self.assertEqual(call_id_1, call_id_2)
        # Verify exactly one service call exists for this interaction
        self.assertEqual(ServiceCall.objects.filter(source_interactions=interaction).count(), 1)

    def test_escalation_concurrency_and_retry(self):
        """Test that create_call_from_interaction handles OperationalError and IntegrityError with retry."""
        from unittest.mock import patch
        from django.db import OperationalError
        from core.services.ticketing import create_call_from_interaction

        interaction = CustomerInteraction.objects.create(
            tenant=self.tenant_a, created_by=self.user_cust1, customer=self.customer_a1,
            site=self.site_a1, asset=self.asset_a1, question="Hydraulic cylinder noisy", answer="Check seals"
        )

        orig_create = ServiceCall.objects.create
        call_count = [0]

        def flaky_create(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate SQLite lock error on first attempt
                raise OperationalError("database is locked")
            return orig_create(*args, **kwargs)

        with patch.object(ServiceCall.objects, "create", side_effect=flaky_create):
            call = create_call_from_interaction(interaction)

        self.assertIsNotNone(call)
        self.assertEqual(call_count[0], 2)
        self.assertEqual(ServiceCall.objects.filter(source_interactions=interaction).count(), 1)

    def test_store_roles_cannot_resolve_or_escalate_interaction(self):
        """Store operators and store admins must be forbidden from Customer AI Support resolve & escalate endpoints."""
        interaction = CustomerInteraction.objects.create(
            tenant=self.tenant_a, created_by=self.user_cust1, customer=self.customer_a1,
            site=self.site_a1, asset=self.asset_a1, question="Valve leaking", answer="Replace seal"
        )
        resolve_url = reverse("api_customer_support_resolve", kwargs={"pk": interaction.id})
        escalate_url = reverse("api_customer_support_escalate", kwargs={"pk": interaction.id})

        # Store Operator -> 403
        self.client.login(username="store_user", password="StorePassword!2026")
        res_res = self.client.post(resolve_url)
        self.assertEqual(res_res.status_code, 403)
        self.assertIn("Customer AI Support is not available for store roles", str(res_res.data))

        res_esc = self.client.post(escalate_url)
        self.assertEqual(res_esc.status_code, 403)
        self.assertIn("Customer AI Support is not available for store roles", str(res_esc.data))

        # Store Admin -> 403
        self.client.login(username="store_admin_user", password="StoreAdminPassword!2026")
        res_res2 = self.client.post(resolve_url)
        self.assertEqual(res_res2.status_code, 403)

        res_esc2 = self.client.post(escalate_url)
        self.assertEqual(res_esc2.status_code, 403)

    # ------------------------------------------------------------------
    # 5. Multi-Tenant Superuser Selection
    # ------------------------------------------------------------------
    def test_superuser_multiple_tenants_unselected_returns_400(self):
        self.client.login(username="super", password="SuperPassword!2026")
        # Ensure session has NO selected tenant
        session = self.client.session
        session.pop("active_tenant_id", None)
        session.pop("tenant_id", None)
        session.save()

        # /api/auth/me/ succeeds for superuser and returns role superuser and available_tenants
        res_me = self.client.get(reverse("api_auth_me"))
        self.assertEqual(res_me.status_code, 200)
        self.assertEqual(res_me.data["user"]["role"], "superuser")
        self.assertGreaterEqual(len(res_me.data["available_tenants"]), 2)

        # Any tenant-scoped endpoint returns 400
        res = self.client.get(reverse("api_dashboard"))
        self.assertEqual(res.status_code, 400)
        self.assertIn("Select an active tenant first", str(res.data))

        # Switch to Tenant Alpha -> succeeds
        switch_url = reverse("api_auth_switch_tenant")
        res_switch = self.client.post(switch_url, {"tenant_id": self.tenant_a.id}, format="json")
        self.assertEqual(res_switch.status_code, 200)

        # Now dashboard works
        res_dash = self.client.get(reverse("api_dashboard"))
        self.assertEqual(res_dash.status_code, 200)
        self.assertEqual(res_dash.data["tenant_name"], "Tenant Alpha")

    # ------------------------------------------------------------------
    # 6. Knowledge Base Confidentiality and File Upload Security
    # ------------------------------------------------------------------
    def test_customer_cannot_access_confidential_document(self):
        self.client.login(username="cust1_user", password="CustPassword!2026")
        detail_url = reverse("api_knowledge_detail", kwargs={"pk": self.doc_confidential.id})
        res = self.client.get(detail_url)
        self.assertEqual(res.status_code, 404)

        download_url = reverse("api_knowledge_download", kwargs={"pk": self.doc_confidential.id})
        res_dl = self.client.get(download_url)
        self.assertEqual(res_dl.status_code, 404)

    def test_technician_confidential_visibility(self):
        # Technician without confidential permission
        self.client.login(username="tech_user", password="TechPassword!2026")
        res = self.client.get(reverse("api_knowledge"))
        self.assertEqual(res.status_code, 200)
        titles = [d["title"] for d in res.data["documents"]]
        self.assertIn("Aerolift Valve Troubleshooting", titles)
        self.assertNotIn("Aerolift Secret Calibration Blueprint", titles)

        # Technician WITH confidential permission
        self.client.login(username="tech_conf", password="TechPassword!2026")
        res_conf = self.client.get(reverse("api_knowledge"))
        self.assertEqual(res_conf.status_code, 200)
        titles_conf = [d["title"] for d in res_conf.data["documents"]]
        self.assertIn("Aerolift Secret Calibration Blueprint", titles_conf)

    def test_store_role_cannot_access_knowledge_base(self):
        self.client.login(username="store_user", password="StorePassword!2026")
        res = self.client.get(reverse("api_knowledge"))
        self.assertEqual(res.status_code, 403)

    def test_kb_upload_authorization_and_safety_checks(self):
        url = reverse("api_knowledge_upload")

        # Technician cannot upload -> 403
        self.client.login(username="tech_user", password="TechPassword!2026")
        res = self.client.post(url, {"title": "Tech Upload", "content_text": "text"})
        self.assertEqual(res.status_code, 403)

        # Admin can upload -> 201
        self.client.login(username="admin_user", password="AdminPassword!2026")
        res_ok = self.client.post(url, {"title": "Admin Upload", "content_text": "Hydraulic specs"})
        self.assertEqual(res_ok.status_code, 201)

        # Upload with invalid extension -> 400
        bad_file = SimpleUploadedFile("script.exe", b"binary content", content_type="application/octet-stream")
        res_bad_ext = self.client.post(url, {"title": "Malware", "file": bad_file})
        self.assertEqual(res_bad_ext.status_code, 400)
        self.assertIn("Unsupported file extension", str(res_bad_ext.data))

        # Upload fake PDF (wrong magic bytes) -> 400
        fake_pdf = SimpleUploadedFile("fake.pdf", b"NOT_A_PDF_HEADER", content_type="application/pdf")
        res_fake_pdf = self.client.post(url, {"title": "Fake PDF", "file": fake_pdf})
        self.assertEqual(res_fake_pdf.status_code, 400)
        self.assertIn("PDF specification", str(res_fake_pdf.data))

        # Upload malformed DOCX (valid PK header but corrupted zip payload) -> 400 NOT 500
        corrupted_content = b"PK\x03\x04corrupted_payload_that_cannot_be_unzipped"
        bad_docx = SimpleUploadedFile("broken.docx", corrupted_content, content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        res_bad_docx = self.client.post(url, {
            "title": "Broken DOCX Test",
            "doc_type": "manual",
            "file": bad_docx,
        }, format="multipart")
        self.assertEqual(res_bad_docx.status_code, 400)
        self.assertIn("detail", res_bad_docx.data)
        self.assertFalse(KnowledgeDocument.objects.filter(title="Broken DOCX Test").exists())

    # ------------------------------------------------------------------
    # 7. Supporting Modules and Field Schema Verification
    # ------------------------------------------------------------------
    def test_inventory_list_contract_and_permissions(self):
        url = reverse("api_inventory")

        # Customer -> 403
        self.client.login(username="cust1_user", password="CustPassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 403)

        # Staff (Tech) -> 200
        self.client.login(username="tech_user", password="TechPassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertIn("items", res.data)
        item = res.data["items"][0]
        self.assertEqual(item["spare_name"], "Hydraulic Valve Seal")
        self.assertEqual(item["ipn"], "HVS-990")
        self.assertEqual(item["quantity"], 25)
        self.assertEqual(item["branch__name"], "Alpha HQ")

    def test_operations_list_contract(self):
        url = reverse("api_operations")
        self.client.login(username="tech_user", password="TechPassword!2026")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertIn("zones", res.data)
        zone = res.data["zones"][0]
        self.assertEqual(zone["name"], "North Zone")
        self.assertEqual(zone["branch__name"], "Alpha HQ")

    # ------------------------------------------------------------------
    # 8. Complete End-to-End Flow
    # ------------------------------------------------------------------
    def test_full_end_to_end_flow(self):
        """
        Customer login
        -> Select authorized Site & Asset
        -> Asks Aerolift troubleshooting question
        -> RAG retrieves permitted KB docs
        -> Grounded answer returned
        -> Customer escalates
        -> Exactly ONE ServiceCall created
        -> Visible in Call Register
        -> Technician logs in
        -> Technician opens existing call
        -> Engineer Copilot receives full call context
        """
        # Step A: Customer login
        login_res = self.client.login(username="cust1_user", password="CustPassword!2026")
        self.assertTrue(login_res)

        # Step B: Customer queries troubleshooting assistant
        query_url = reverse("api_customer_support_query")
        rag_res = self.client.post(query_url, {
            "site_id": self.site_a1.id,
            "asset_id": self.asset_a1.id,
            "question": "Aerolift hydraulic lift pressure is dropping and leaking at the valve. What should I do?",
        }, format="json")
        self.assertEqual(rag_res.status_code, 200)
        interaction_id = rag_res.data["interaction_id"]
        answer = rag_res.data["answer"]
        self.assertIn("valve", answer.lower())

        # Step C: Customer escalates unresolved issue
        escalate_url = reverse("api_customer_support_escalate", kwargs={"pk": interaction_id})
        esc_res = self.client.post(escalate_url)
        self.assertEqual(esc_res.status_code, 200)
        call_id = esc_res.data["call_id"]
        servy_id = esc_res.data["servy_id"]
        self.assertTrue(servy_id)

        # Step D: Call appears in Call Register
        calls_url = reverse("api_calls")
        calls_res = self.client.get(calls_url)
        self.assertEqual(calls_res.status_code, 200)
        self.assertTrue(any(c["id"] == call_id for c in calls_res.data["calls"]))

        # Step E: Technician logs in and opens that existing call
        self.client.logout()
        tech_login = self.client.login(username="tech_user", password="TechPassword!2026")
        self.assertTrue(tech_login)

        call_detail_url = reverse("api_call_detail", kwargs={"pk": call_id})
        detail_res = self.client.get(call_detail_url)
        self.assertEqual(detail_res.status_code, 200)
        self.assertEqual(detail_res.data["call"]["asset"]["name"], "Lift-01")

        # Step F: Engineer Copilot query against this call
        copilot_url = reverse("api_engineer_copilot_query")
        copilot_res = self.client.post(copilot_url, {
            "call_id": call_id,
            "question": "How do I calibrate and test the hydraulic valve seal replacement?",
        }, format="json")
        self.assertEqual(copilot_res.status_code, 200)
        self.assertIn("answer", copilot_res.data)
        self.assertIn("references", copilot_res.data)
        # Tech does NOT see confidential key because can_view_confidential=False
        self.assertNotIn("7741", copilot_res.data["answer"])

    def test_engineer_copilot_past_resolutions_source_kind(self):
        """Verify past_resolutions array in Engineer Copilot filters for source_kind == 'past_resolution'."""
        from unittest.mock import patch

        call = ServiceCall.objects.create(
            tenant=self.tenant_a,
            servy_id=42999,
            call_type="Service",
            complaint_type="Valve Leaking",
            complaint_text="Hydraulic valve seal broken",
            customer=self.customer_a1,
            site=self.site_a1,
            asset=self.asset_a1,
            status="assigned",
            priority="normal"
        )

        mock_references = [
            {
                "source_kind": "past_resolution",
                "service_call_id": call.id,
                "title": f"Service Call #{call.servy_id} - Valve Leaking",
                "reference": f"Service Call #{call.servy_id}",
                "doc_type": "service_resolution",
                "score": 0.95,
            },
            {
                "source_kind": "knowledge",
                "document_id": self.doc_public.id,
                "title": self.doc_public.title,
                "reference": self.doc_public.title,
                "doc_type": "reference",
                "score": 0.88,
            }
        ]

        def mock_ask_rag(*args, **kwargs):
            return {
                "answer": "Refer to previous resolution for valve seal replacement.",
                "references": mock_references,
                "engine": "mock",
                "retrieved": [],
            }

        self.client.login(username="tech_user", password="TechPassword!2026")
        url = reverse("api_engineer_copilot_query")
        with patch("core.api_views.ask_rag", side_effect=mock_ask_rag):
            res = self.client.post(url, {
                "call_id": call.id,
                "question": "How do I fix valve leak?",
            }, format="json")

        self.assertEqual(res.status_code, 200)
        self.assertIn("past_resolutions", res.data)
        self.assertEqual(len(res.data["past_resolutions"]), 1)
        self.assertEqual(res.data["past_resolutions"][0]["service_call_id"], call.id)
        self.assertEqual(res.data["past_resolutions"][0]["title"], f"Service Call #{call.servy_id} - Valve Leaking")
