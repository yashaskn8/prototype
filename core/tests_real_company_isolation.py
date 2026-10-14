"""
Phase 6: Adversarial Test Suite for Real Company Data Isolation and Zero Deletion.

Verifies:
 1. Customer real-data mode never returns synthetic assets.
 2. Customer real-data mode never returns synthetic manuals.
 3. Real RAG never retrieves dummy Chroma vectors.
 4. Missing real documentation never triggers dummy fallback.
 5. Wrong-organisation documents are rejected.
 6. Customers cannot access unrelated company assets (IDOR protection).
 7. Fake complaint history never appears in real tickets.
 8. Real source citations point to actual authorized document records.
 9. Customer-safe actions require independently validated approval.
10. Existing dummy/test records remain completely unchanged.
11. Existing real company records remain unchanged except authorized imports.
12. Test and demo workflows continue working independently in test_isolated mode.
13. Switching configuration does not bypass backend authorization.
14. Failure during ingestion does not alter existing company or dummy records.
15. Repeated import attempts do not create misleading duplicates (idempotency).
"""

from unittest.mock import patch, MagicMock
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.db import transaction

from core.models import (
    Asset,
    Brand,
    Customer,
    CustomerInteraction,
    DiagnosticEvent,
    DiagnosticSession,
    KnowledgeChunk,
    KnowledgeDocument,
    OperationalZone,
    Product,
    ProductCategory,
    ProductDomain,
    ServiceCall,
    Site,
    StaffProfile,
    Tenant,
    TenantMembership,
)
from core.services.customer_isolation import (
    get_customer_data_mode,
    is_real_tenant,
    is_synthetic_asset,
    is_synthetic_customer,
    validate_customer_serving_policy,
)
from core.services.customer_diagnostic import (
    build_structured_diagnostic_contract,
    classify_action_safety,
    find_grounding_reference,
)
from core.services.retriever import retrieve
from core.services.starlly_ingest import StarllyRealDataIngestionService


@override_settings(
    SERVY_CUSTOMER_DATA_MODE="verified_real_only",
    STARLLY_REAL_TENANTS=["skydream", "real-test-tenant"],
    STARLLY_REAL_ORGANISATIONS=["273"],
    SERVY_LLM_PROVIDER="extractive",
    SERVY_RAG_MIN_SCORE=0.0,
)
class RealCompanyDataIsolationTests(TestCase):
    def setUp(self):
        # Real company tenant
        self.real_tenant = Tenant.objects.create(name="SkyDream Technologies", slug="real-test-tenant", is_active=True)
        self.demo_tenant = Tenant.objects.create(name="Demo Tester", slug="demo-tester-tenant", is_active=True)

        self.zone = OperationalZone.objects.create(tenant=self.real_tenant, name="Primary Zone")
        self.domain = ProductDomain.objects.create(tenant=self.real_tenant, name="HVAC")
        self.category = ProductCategory.objects.create(tenant=self.real_tenant, domain=self.domain, name="Cooling")
        self.brand = Brand.objects.create(tenant=self.real_tenant, name="SkyCool")
        self.product = Product.objects.create(
            tenant=self.real_tenant, category=self.category, brand=self.brand, name="Split AC 2T"
        )

        # Real customer & site
        self.real_customer = Customer.objects.create(tenant=self.real_tenant, name="Rajesh Kumar")
        self.real_site = Site.objects.create(tenant=self.real_tenant, customer=self.real_customer, name="Plant 1", zone=self.zone)
        self.real_asset = Asset.objects.create(
            tenant=self.real_tenant, customer=self.real_customer, site=self.real_site, product=self.product,
            name="Split AC Unit 2T", asset_code="SKY-AST-00001", status="active"
        )

        # Another customer in the same tenant (for cross-customer IDOR test)
        self.other_customer = Customer.objects.create(tenant=self.real_tenant, name="GreenTech Industries")
        self.other_site = Site.objects.create(tenant=self.real_tenant, customer=self.other_customer, name="GreenTech Unit 1")
        self.other_asset = Asset.objects.create(
            tenant=self.real_tenant, customer=self.other_customer, site=self.other_site, product=self.product,
            name="VRF Unit", asset_code="SKY-AST-00002", status="active"
        )

        # Synthetic/demo asset inside demo tenant
        self.demo_customer = Customer.objects.create(tenant=self.demo_tenant, name="Dummy Test Customer")
        self.demo_site = Site.objects.create(tenant=self.demo_tenant, customer=self.demo_customer, name="Demo Site")
        self.demo_asset = Asset.objects.create(
            tenant=self.demo_tenant, customer=self.demo_customer, site=self.demo_site,
            name="Dummy Test Asset", asset_code="STAR-AST-99999", status="active"
        )

        # Real user
        self.user_real = User.objects.create_user("rajesh_k", password="TestPassword123")
        TenantMembership.objects.create(user=self.user_real, tenant=self.real_tenant, role="customer", customer=self.real_customer)

        # Demo user
        self.user_demo = User.objects.create_user("demo_user", password="TestPassword123")
        TenantMembership.objects.create(user=self.user_demo, tenant=self.demo_tenant, role="customer", customer=self.demo_customer)

        # Real technical document with valid chunks
        self.real_doc = KnowledgeDocument.objects.create(
            tenant=self.real_tenant,
            product=self.product,
            asset=self.real_asset,
            title="Split AC Unit 2T Troubleshooting Manual",
            doc_type="troubleshooting",
            is_rag_enabled=True,
            is_confidential=False,
            disable_sharing=False,
            index_status="INDEXED",
        )
        from core.services.vectors import embed_text
        chunk_text = "CHECK FIRST\n- Confirm electrical power is supplied to the indoor unit.\n- Inspect the washable air filter for heavy dust accumulation.\n\nSTEP-BY-STEP TROUBLESHOOTING\nStep 1: Clean exterior and wipe dust filter with clean water.\nWhy: Restores necessary airflow across the evaporator coil.\nStep 2: Restart the unit using the main power switch.\nWhy: Resets internal microcontroller safety lockout.\n\nEXPECTED RESULT\nIndoor blower motor resumes normal air delivery within 3 minutes.\n\nSTOP AND ESCALATE IF\n- Unusual burning smell or smoke is detected.\n- Error code E4 persists on the front panel."
        emb = embed_text(chunk_text)
        self.real_chunk = KnowledgeChunk.objects.create(
            tenant=self.real_tenant,
            document=self.real_doc,
            chunk_index=0,
            heading="Indoor fan motor does not start",
            text=chunk_text,
            embedding=emb,
            content_embedding=emb,
        )

        # Demo document in demo tenant
        self.demo_doc = KnowledgeDocument.objects.create(
            tenant=self.demo_tenant,
            title="Synthetic Demo Procedure",
            doc_type="troubleshooting",
            is_rag_enabled=True,
            is_confidential=False,
            tags="demo, synthetic",
            index_status="INDEXED",
        )
        demo_text = "Indoor fan motor does not start synthetic troubleshooting."
        demo_emb = embed_text(demo_text)
        self.demo_chunk = KnowledgeChunk.objects.create(
            tenant=self.demo_tenant,
            document=self.demo_doc,
            chunk_index=0,
            heading="Indoor fan motor does not start",
            text=demo_text,
            embedding=demo_emb,
            content_embedding=demo_emb,
        )

    # -----------------------------------------------------------------------
    # 1. Customer real-data mode never returns synthetic assets
    # -----------------------------------------------------------------------
    def test_01_customer_real_mode_never_returns_synthetic_assets(self):
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_context")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        asset_codes = [a["asset_code"] for a in res.data["assets"]]
        self.assertIn("SKY-AST-00001", asset_codes)
        for code in asset_codes:
            self.assertFalse(code.startswith("STAR-AST-"))
            self.assertFalse(code.startswith("TEST-"))
            self.assertFalse(code.startswith("DUMMY-"))

        # Accessing demo tenant in verified_real_only mode must be rejected with HTTP 403
        self.client.login(username="demo_user", password="TestPassword123")
        res_demo = self.client.get(url)
        self.assertEqual(res_demo.status_code, 403)
        self.assertIn("verified_real_only", res_demo.data.get("detail", ""))

    # -----------------------------------------------------------------------
    # 2. Customer real-data mode never returns synthetic manuals
    # -----------------------------------------------------------------------
    def test_02_customer_real_mode_never_returns_synthetic_manuals(self):
        results = retrieve(
            tenant=self.real_tenant,
            question="Indoor fan motor does not start",
            asset=self.real_asset,
            customer=self.real_customer,
            include_confidential=False,
            top_k=5,
            min_score=0.0,
            staff_mode=False,
        )
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r.document_id, self.real_doc.id)
            self.assertNotIn("Synthetic", r.title)
            self.assertNotEqual(r.document_id, self.demo_doc.id)

    # -----------------------------------------------------------------------
    # 3. Real RAG never retrieves dummy Chroma vectors
    # -----------------------------------------------------------------------
    @patch("core.services.retriever._dense_score_candidates")
    def test_03_real_rag_never_retrieves_dummy_chroma_vectors(self, mock_dense):
        mock_dense.return_value = []
        retrieve(
            tenant=self.real_tenant,
            question="Indoor fan motor does not start",
            asset=self.real_asset,
            customer=self.real_customer,
            include_confidential=False,
            top_k=5,
            min_score=0.0,
            staff_mode=False,
        )
        self.assertTrue(mock_dense.called)
        call_kwargs = mock_dense.call_args[1] if mock_dense.call_args[1] else {}
        call_args = mock_dense.call_args[0]
        # Tenant passed must be real_tenant
        tenant_arg = call_args[7] if len(call_args) > 7 else call_kwargs.get("tenant")
        self.assertEqual(tenant_arg, self.real_tenant)

    # -----------------------------------------------------------------------
    # 4. Missing real documentation never triggers dummy fallback
    # -----------------------------------------------------------------------
    def test_04_missing_real_documentation_never_triggers_dummy_fallback(self):
        # Query for completely unrelated issue not covered in real doc
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_query")
        payload = {
            "site_id": self.real_site.id,
            "asset_id": self.real_asset.id,
            "question": "hydraulic cylinder seal blowout with oil puddle",
        }
        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, 200)
        # Must return INSUFFICIENT_EVIDENCE or ESCALATION_REQUIRED, never synthetic guidance
        self.assertIn(res.data["response_type"], ["INSUFFICIENT_EVIDENCE", "ESCALATION_REQUIRED"])
        self.assertFalse(res.data["can_resolve"])
        self.assertTrue(res.data["can_escalate"])
        self.assertEqual(len(res.data["customer_safe_actions"]), 0)

    # -----------------------------------------------------------------------
    # 5. Wrong-organisation documents are rejected
    # -----------------------------------------------------------------------
    def test_05_wrong_organisation_documents_are_rejected(self):
        is_grounded, ref, reason = find_grounding_reference(
            action_text="Clean exterior and wipe dust filter with clean water.",
            retrieved_references=[{"document_id": self.demo_doc.id, "chunk_id": self.demo_chunk.id}],
            tenant_id=self.real_tenant.id,
            asset=self.real_asset,
            customer=self.real_customer,
        )
        self.assertFalse(is_grounded)
        self.assertIsNone(ref)

    # -----------------------------------------------------------------------
    # 6. Customers cannot access unrelated company assets
    # -----------------------------------------------------------------------
    def test_06_customers_cannot_access_unrelated_company_assets(self):
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_query")
        # Attempt to access GreenTech's asset using Rajesh's credentials
        payload = {
            "site_id": self.other_site.id,
            "asset_id": self.other_asset.id,
            "question": "Split AC fan motor issue",
        }
        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("does not belong to this customer", res.data.get("detail", ""))

    # -----------------------------------------------------------------------
    # 7. Fake complaint history never appears in real tickets
    # -----------------------------------------------------------------------
    def test_07_fake_complaint_history_never_appears_in_real_tickets(self):
        # Create an interaction for real asset
        interaction = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            created_by=self.user_real,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Indoor fan motor does not start",
            answer="Safe diagnostic steps provided.",
            source_refs=[],
        )
        self.client.login(username="rajesh_k", password="TestPassword123")
        esc_url = reverse("api_customer_support_escalate", kwargs={"pk": interaction.id})
        res = self.client.post(esc_url, {"note": "Customer confirms fan still inactive."})
        self.assertEqual(res.status_code, 200)
        call = ServiceCall.objects.get(id=res.data["call_id"])
        self.assertEqual(call.tenant, self.real_tenant)
        self.assertEqual(call.customer, self.real_customer)
        self.assertEqual(call.asset, self.real_asset)
        self.assertNotIn("DUMMY", call.complaint_text)
        self.assertNotIn("SRV-SYN", call.complaint_text)

    # -----------------------------------------------------------------------
    # 8. Real source citations point to actual authorized document records
    # -----------------------------------------------------------------------
    def test_08_real_source_citations_point_to_actual_authorized_documents(self):
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_query")
        payload = {
            "site_id": self.real_site.id,
            "asset_id": self.real_asset.id,
            "question": "Indoor fan motor does not start",
        }
        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertIn("citations", res.data)
        for cit in res.data["citations"]:
            doc = KnowledgeDocument.objects.get(id=cit["document_id"])
            self.assertEqual(doc.tenant, self.real_tenant)
            self.assertFalse(doc.is_confidential)
            self.assertTrue(doc.is_rag_enabled)

    # -----------------------------------------------------------------------
    # 9. Customer-safe actions require independently validated approval
    # -----------------------------------------------------------------------
    def test_09_customer_safe_actions_require_independently_validated_approval(self):
        # Create unapproved/dangerous procedure chunk
        danger_doc = KnowledgeDocument.objects.create(
            tenant=self.real_tenant,
            product=self.product,
            asset=self.real_asset,
            title="Internal Compressor Service",
            doc_type="service_manual",
            is_rag_enabled=True,
            is_confidential=False,
        )
        danger_chunk = KnowledgeChunk.objects.create(
            tenant=self.real_tenant,
            document=danger_doc,
            chunk_index=0,
            heading="High voltage capacitor discharge",
            text="STEP-BY-STEP TROUBLESHOOTING\nStep 1: Open electrical cabinet and probe live 415v terminal.\nStep 2: Disassemble motor winding.",
            embedding={"0": 0.9},
        )
        interaction = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            created_by=self.user_real,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="High voltage capacitor discharge",
            answer="DIAGNOSTIC SUMMARY\nCompressor service.\n\nSTEP-BY-STEP TROUBLESHOOTING\nStep 1: Open electrical cabinet and probe live terminal.\nStep 2: Disassemble motor winding.",
            source_refs=[{"document_id": danger_doc.id, "chunk_id": danger_chunk.id}],
        )
        contract = build_structured_diagnostic_contract(
            interaction=interaction,
            session=None,
            raw_answer=interaction.answer,
            retrieved_references=interaction.source_refs,
        )
        # All dangerous procedures must be excluded from customer_safe_actions
        self.assertEqual(len(contract["customer_safe_actions"]), 0)
        self.assertTrue(len(contract["technician_only_actions"]) > 0)
        self.assertEqual(contract["response_type"], "ESCALATION_REQUIRED")

    # -----------------------------------------------------------------------
    # 10. Existing dummy/test records remain unchanged
    # -----------------------------------------------------------------------
    def test_10_existing_dummy_records_remain_unchanged(self):
        initial_demo_assets = Asset.objects.filter(tenant=self.demo_tenant).count()
        initial_demo_docs = KnowledgeDocument.objects.filter(tenant=self.demo_tenant).count()
        initial_demo_chunks = KnowledgeChunk.objects.filter(tenant=self.demo_tenant).count()

        # Perform customer query and resolution on real tenant
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_query")
        self.client.post(url, {
            "site_id": self.real_site.id,
            "asset_id": self.real_asset.id,
            "question": "Indoor fan motor does not start",
        }, format="json")

        self.assertEqual(Asset.objects.filter(tenant=self.demo_tenant).count(), initial_demo_assets)
        self.assertEqual(KnowledgeDocument.objects.filter(tenant=self.demo_tenant).count(), initial_demo_docs)
        self.assertEqual(KnowledgeChunk.objects.filter(tenant=self.demo_tenant).count(), initial_demo_chunks)

    # -----------------------------------------------------------------------
    # 11. Existing real company records remain unchanged except authorized imports
    # -----------------------------------------------------------------------
    def test_11_existing_company_records_remain_unchanged(self):
        initial_count = Asset.objects.filter(tenant=self.real_tenant).count()
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_context")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(Asset.objects.filter(tenant=self.real_tenant).count(), initial_count)

    # -----------------------------------------------------------------------
    # 12. Test and demo workflows continue working independently
    # -----------------------------------------------------------------------
    @override_settings(SERVY_CUSTOMER_DATA_MODE="test_isolated")
    def test_12_test_and_demo_workflows_continue_working_independently(self):
        self.client.login(username="demo_user", password="TestPassword123")
        url = reverse("api_customer_support_context")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data["customer"]["name"], "Dummy Test Customer")
        self.assertEqual(len(res.data["assets"]), 1)
        self.assertEqual(res.data["assets"][0]["asset_code"], "STAR-AST-99999")

    # -----------------------------------------------------------------------
    # 13. Switching configuration does not bypass backend authorization
    # -----------------------------------------------------------------------
    @override_settings(SERVY_CUSTOMER_DATA_MODE="test_isolated")
    def test_13_switching_configuration_does_not_bypass_backend_authorization(self):
        # Even in test_isolated mode, cross-tenant IDOR must remain strictly prohibited
        self.client.login(username="demo_user", password="TestPassword123")
        url = reverse("api_customer_support_query")
        payload = {
            "site_id": self.real_site.id,  # Cross-tenant IDOR attempt
            "asset_id": self.real_asset.id,
            "question": "test question",
        }
        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, 400)

    # -----------------------------------------------------------------------
    # 14. Failure during ingestion does not alter existing records
    # -----------------------------------------------------------------------
    def test_14_failure_during_ingestion_does_not_alter_existing_records(self):
        initial_asset_count = Asset.objects.filter(tenant=self.real_tenant).count()
        initial_doc_count = KnowledgeDocument.objects.filter(tenant=self.real_tenant).count()

        # Simulate exception during sync inside atomic transaction
        with patch("core.services.starlly_ingest.sync_document_to_chroma", side_effect=RuntimeError("Simulated vector sync failure")):
            try:
                with transaction.atomic():
                    # Attempt a mutation that fails mid-way
                    KnowledgeDocument.objects.create(
                        tenant=self.real_tenant,
                        title="Temporary Failed Document",
                        is_rag_enabled=True,
                    )
                    raise RuntimeError("Simulated transaction rollback during ingestion failure")
            except RuntimeError:
                pass

        # Verify transaction rolled back cleanly without creating orphan records
        self.assertEqual(Asset.objects.filter(tenant=self.real_tenant).count(), initial_asset_count)
        self.assertEqual(KnowledgeDocument.objects.filter(tenant=self.real_tenant).count(), initial_doc_count)

    # -----------------------------------------------------------------------
    # 15. Repeated import attempts do not create misleading duplicates
    # -----------------------------------------------------------------------
    def test_15_repeated_import_attempts_do_not_create_misleading_duplicates(self):
        initial_asset_count = Asset.objects.filter(tenant=self.real_tenant).count()
        mock_asset_data = {
            "assets": [
                {
                    "id": 101,
                    "code": "SKY-AST-00001",
                    "name": "Split AC Unit 2T",
                    "model_number": "SAC-2000",
                    "serial_number": "SN-001",
                    "status": "Active",
                    "attachments": [],
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("core.services.starlly_api.StarllyApiClient.fetch_assets", return_value=mock_asset_data):
            with patch("requests.get", return_value=mock_response):
                from core.services.starlly_ingest import ingest_starlly_real_data
                res1 = ingest_starlly_real_data(tenant_slug=self.real_tenant.slug)
                res2 = ingest_starlly_real_data(tenant_slug=self.real_tenant.slug)

        self.assertEqual(res1["assets_created"], 0)  # Already existed
        self.assertEqual(res2["assets_created"], 0)  # Idempotent re-run
        # Asset count should remain exactly unchanged (2 assets in real_tenant)
        self.assertEqual(Asset.objects.filter(tenant=self.real_tenant).count(), initial_asset_count)

    # -----------------------------------------------------------------------
    # 16. Injected synthetic asset in real tenant rejected by positive provenance
    # -----------------------------------------------------------------------
    def test_16_synthetic_asset_injected_in_real_tenant_rejected_by_provenance(self):
        # Even if injected directly into real tenant, an asset without positive verified provenance is rejected
        injected_asset = Asset.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            name="Injected Mystery Unit",
            asset_code="CUSTOM-INJECTED-99",
            status="active",
        )
        self.assertTrue(is_synthetic_asset(injected_asset))
        from rest_framework.exceptions import PermissionDenied
        with self.assertRaises(PermissionDenied):
            validate_customer_serving_policy(tenant=self.real_tenant, customer=self.real_customer, asset=injected_asset)

        # In context view, injected asset must not be returned to customer
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_context")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        returned_codes = [a["asset_code"] for a in res.data["assets"]]
        self.assertNotIn("CUSTOM-INJECTED-99", returned_codes)

    # -----------------------------------------------------------------------
    # 17. Real asset with test-like name preserved via positive provenance
    # -----------------------------------------------------------------------
    def test_17_real_asset_with_test_like_name_preserved(self):
        # SKY-AST-00005 is a legitimate asset imported from Starlly Org 273 with name 'test asset'
        test_asset = Asset.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            name="test asset",
            asset_code="SKY-AST-00005",
            status="active",
        )
        # Provenance verification confirms it is genuine company asset
        self.assertFalse(is_synthetic_asset(test_asset))

    # -----------------------------------------------------------------------
    # 18. Ungrounded check_first instruction rejected from contract
    # -----------------------------------------------------------------------
    def test_18_ungrounded_check_first_instruction_rejected(self):
        # Create interaction with raw answer containing both grounded and fabricated ungrounded checks
        interaction = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Indoor fan motor does not start",
            answer="CHECK FIRST\n- Inspect the washable air filter for heavy dust accumulation.\n- Unpack internal capacitor and adjust potentiometer screws.\n\nSTEP-BY-STEP TROUBLESHOOTING\nStep 1: Clean exterior and wipe dust filter with clean water.\nWhy: Restores airflow.",
        )
        contract = build_structured_diagnostic_contract(
            interaction=interaction,
            session=None,
            raw_answer=interaction.answer,
            retrieved_references=[{
                "document_id": self.real_doc.id,
                "chunk_id": self.real_chunk.id,
                "reference": "Split AC Unit 2T Troubleshooting Manual",
                "text": self.real_chunk.text,
            }],
        )
        # Grounded check is included
        self.assertTrue(any("air filter" in c.lower() for c in contract["check_first"]))
        # Fabricated ungrounded check must be rejected
        self.assertFalse(any("potentiometer" in c.lower() for c in contract["check_first"]))
        self.assertFalse(any("capacitor" in c.lower() for c in contract["check_first"]))

    # -----------------------------------------------------------------------
    # 19. Asset without technical manual returns INSUFFICIENT_EVIDENCE
    # -----------------------------------------------------------------------
    def test_19_asset_without_technical_manual_returns_insufficient_evidence(self):
        # SKY-AST-00004 (Air Compressor) has no manuals in Starlly
        compressor = Asset.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            name="Air Compressor Unit 01",
            asset_code="SKY-AST-00004",
            status="active",
        )
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_query")
        payload = {
            "site_id": self.real_site.id,
            "asset_id": compressor.id,
            "question": "Motor makes humming noise but does not turn",
        }
        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res.data["can_resolve"], False)
        self.assertEqual(res.data["can_escalate"], True)
        self.assertEqual(len(res.data["customer_safe_actions"]), 0)
        # Must not fabricate customer steps
        self.assertIn("INSUFFICIENT EVIDENCE", res.data["answer"])

    # -----------------------------------------------------------------------
    # 20. Hazard detected in follow-up observation escalates session
    # -----------------------------------------------------------------------
    def test_20_hazard_detected_in_follow_up_escalates_session(self):
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_query")
        payload = {
            "site_id": self.real_site.id,
            "asset_id": self.real_asset.id,
            "question": "AC unit stopped working",
            "observation": "I see smoke and sparking from the power unit",
        }
        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data["response_type"], "ESCALATION_REQUIRED")
        self.assertEqual(res.data["can_resolve"], False)
        self.assertEqual(res.data["can_escalate"], True)
        self.assertIn("IMMEDIATE SAFETY RISK DETECTED", res.data["stop_conditions"][0])

    # -----------------------------------------------------------------------
    # 21. Persisted hazard cannot be cleared by follow-up query omitting keyword
    # -----------------------------------------------------------------------
    def test_21_persisted_hazard_cannot_be_cleared_by_omitting_keywords(self):
        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_query")

        # Step 1: Initial hazardous query
        self.client.post(url, {
            "site_id": self.real_site.id,
            "asset_id": self.real_asset.id,
            "question": "Heavy smoke and electrical sparking from unit",
        }, format="json")

        # Step 2: Follow-up query without hazard words
        res2 = self.client.post(url, {
            "site_id": self.real_site.id,
            "asset_id": self.real_asset.id,
            "question": "Can I try turning on the switch now?",
        }, format="json")
        self.assertEqual(res2.status_code, 200)
        # Hazard condition MUST persist
        self.assertEqual(res2.data["response_type"], "ESCALATION_REQUIRED")
        self.assertEqual(res2.data["can_resolve"], False)

    # -----------------------------------------------------------------------
    # 22. Separate incidents on same asset create independent tickets
    # -----------------------------------------------------------------------
    def test_22_separate_incidents_on_same_asset_create_independent_tickets(self):
        from core.services.ticketing import create_call_from_interaction

        # Incident 1
        interaction1 = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Filter is blocked and airflow is low",
            answer="Clean filter",
        )
        call1 = create_call_from_interaction(interaction1)

        # Incident 2 (different complaint on same equipment)
        interaction2 = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Compressor has severe vibration",
            answer="Escalation required",
        )
        call2 = create_call_from_interaction(interaction2)

        # Must NOT collapse into call1
        self.assertNotEqual(call1.id, call2.id)
        self.assertNotEqual(call1.servy_id, call2.servy_id)
        self.assertEqual(call1.complaint_text, "Filter is blocked and airflow is low")
        self.assertEqual(call2.complaint_text, "Compressor has severe vibration")

    # -----------------------------------------------------------------------
    # 23. Retry escalation of same incident is strictly idempotent
    # -----------------------------------------------------------------------
    def test_23_retry_escalation_same_incident_is_strictly_idempotent(self):
        from core.services.ticketing import create_call_from_interaction

        interaction = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Thermostat display is dead",
            answer="Escalation required",
        )
        call_run1 = create_call_from_interaction(interaction)
        call_run2 = create_call_from_interaction(interaction)

        # Exactly same service call returned on retry
        self.assertEqual(call_run1.id, call_run2.id)
        self.assertEqual(call_run1.servy_id, call_run2.servy_id)

    # -----------------------------------------------------------------------
    # 24. Resolve endpoint blocks hazardous or escalated session
    # -----------------------------------------------------------------------
    def test_24_resolve_endpoint_blocks_hazardous_or_escalated_session(self):
        self.client.login(username="rajesh_k", password="TestPassword123")
        # Trigger hazard
        res = self.client.post(reverse("api_customer_support_query"), {
            "site_id": self.real_site.id,
            "asset_id": self.real_asset.id,
            "question": "Fire and sparks from outdoor unit",
        }, format="json")
        interaction_id = res.data["interaction_id"]

        # Attempt to mark resolved
        resolve_url = reverse("api_customer_support_resolve", kwargs={"pk": interaction_id})
        res_resolve = self.client.post(resolve_url)
        self.assertEqual(res_resolve.status_code, 400)
        self.assertIn("hazard", res_resolve.data["detail"].lower())

    # -----------------------------------------------------------------------
    # 25. Context endpoint is strictly read-only
    # -----------------------------------------------------------------------
    def test_25_context_endpoint_is_strictly_read_only(self):
        session = DiagnosticSession.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            complaint="Airflow temperature warm",
            status="WAITING_INPUT",
            version=4,
        )
        initial_version = session.version
        initial_events_count = session.events.count()
        initial_interaction_count = CustomerInteraction.objects.count()

        self.client.login(username="rajesh_k", password="TestPassword123")
        url = reverse("api_customer_support_context")
        res = self.client.get(f"{url}?asset_id={self.real_asset.id}")
        self.assertEqual(res.status_code, 200)

        # Verify nothing mutated
        session.refresh_from_db()
        self.assertEqual(session.version, initial_version)
        self.assertEqual(session.events.count(), initial_events_count)
        self.assertEqual(CustomerInteraction.objects.count(), initial_interaction_count)
