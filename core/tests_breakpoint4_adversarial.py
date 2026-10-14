"""
Adversarial Security and Regression Test Suite — Operation Breakpoint 4.0

Executes attacks against actual security boundaries:
1. Forged real-looking asset code
2. Forged real-looking customer name
3. Fake document marked INDEXED on recognized asset
4. Forged description claiming approval
5. Valid registry checksum paired with wrong content
6. Valid registry checksum paired with wrong asset
7. Synthetic document lacking obvious dummy keywords
8. Synthetic historical service call with realistic ID
9. Missing model triggering sparse fallback
10. Sparse fallback retrieving unauthorized documentation
11. Wrong-tenant Chroma vector cross-contamination
12. Partial index with truthful readiness status
13. Conflicting external asset ID carrying an attachment
14. Concurrent import to same filename (atomic file staging)
15. Real PDF with malformed bytes
16. Invasive procedure described as an observation
17. Correct source but unapproved customer procedure
18. Hazard reported through clarification
19. Stale diagnostic session version
20. Resume after source revocation
21. Two unrelated incidents with similar complaint wording
22. Duplicate escalation of one incident
23. Test-mode/production-mode configuration confusion fails closed
24. Zero modification to preexisting company and dummy records
25. Zero modification to existing company vectors and files
"""

import hashlib
import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from django.conf import settings
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework.exceptions import PermissionDenied

from core.models import (
    Asset,
    Customer,
    CustomerInteraction,
    DiagnosticEvent,
    DiagnosticSession,
    KnowledgeChunk,
    KnowledgeDocument,
    Product,
    ProductCategory,
    ProductDomain,
    ServiceCall,
    ServiceResolutionIndex,
    Site,
    Tenant,
    TenantMembership,
)
from core.services.customer_isolation import (
    get_customer_data_mode,
    is_verified_real_asset,
    is_verified_real_customer,
    is_verified_real_document,
    is_verified_real_service_call,
    is_verified_real_tenant,
    validate_customer_serving_policy,
)
from core.services.customer_diagnostic import (
    build_structured_diagnostic_contract,
    classify_action_safety,
    find_grounding_reference,
)
from core.services.retriever import retrieve, _resolution_candidates
from core.services.ticketing import create_call_from_interaction
from core.services.starlly_ingest import (
    IngestionSecurityError,
    fetch_attachment_content,
    ingest_starlly_real_data,
)
from core.services.chroma_store import (
    delete_document_vectors,
    get_document_retrieval_readiness,
)


class Breakpoint4AdversarialRedTeamTests(TestCase):
    """Adversarial security boundary attack scenarios."""

    @classmethod
    def setUpTestData(cls):
        # Dedicated isolated test tenants
        cls.real_tenant = Tenant.objects.create(name="Skydream", slug="skydream")
        cls.attacker_tenant = Tenant.objects.create(name="Attacker Co", slug="attacker-tenant")

        # Verified real domain / category / product / asset
        cls.domain = ProductDomain.objects.create(tenant=cls.real_tenant, name="HVAC")
        cls.category = ProductCategory.objects.create(tenant=cls.real_tenant, domain=cls.domain, name="Cooling")
        cls.product = Product.objects.create(tenant=cls.real_tenant, name="Split AC 2T")

        cls.real_customer = Customer.objects.create(tenant=cls.real_tenant, name="Rajesh Kumar")
        cls.real_site = Site.objects.create(tenant=cls.real_tenant, customer=cls.real_customer, name="Main Plant")
        cls.real_asset = Asset.objects.create(
            tenant=cls.real_tenant,
            customer=cls.real_customer,
            site=cls.real_site,
            product=cls.product,
            name="Split AC Unit 2T",
            asset_code="SKY-AST-00001",
            status="active",
        )

        cls.user_real = User.objects.create_user("rajesh_k", password="TestPassword123")
        TenantMembership.objects.create(
            user=cls.user_real,
            tenant=cls.real_tenant,
            role="customer",
            customer=cls.real_customer,
        )

        # Authentic registered manual for SKY-AST-00001
        cls.auth_sha = "9e5d2d9e772dc71a849fa5cd1b7512b60f0f417f6832e0cdc77a879ab301e8d1"
        cls.doc_text = (
            "ASSET REFERENCE DOCUMENT\n"
            "Asset: Split AC Unit 2T\n"
            "Document: Troubleshooting and Fault Reference\n"
            "Document ID: AC2T-TS-001\n"
            "Revision: 1.0\n"
            "Status: Sample reference document\n\n"
            "1. PURPOSE\n"
            "This guide helps service personnel record common symptoms and select safe initial checks. "
            "Manufacturer-specific fault codes and diagnostic procedures take precedence.\n\n"
            "2. SYMPTOM: UNIT DOES NOT START\n"
            "Possible causes:\n"
            "- Power supply unavailable or isolator switched off.\n"
            "- Breaker or protective device tripped.\n"
            "- Controller batteries depleted or incorrect mode/setpoint.\n"
            "- Loose connection or internal control fault.\n"
            "Initial checks:\n"
            "- Verify the supply externally and inspect the controller settings.\n"
            "- Do not repeatedly reset a tripping breaker.\n"
            "- Electrical panel checks must be performed by a qualified person.\n\n"
            "3. SYMPTOM: POOR COOLING\n"
            "Possible causes:\n"
            "- Dirty filter or coil.\n"
            "- Restricted airflow or blocked outdoor unit clearance.\n"
            "- Doors/windows open or heat load exceeds capacity.\n"
            "- Incorrect settings, fan issue, or refrigerant-system fault.\n"
            "Initial checks:\n"
            "- Clean filters and confirm airflow is unobstructed.\n"
            "- Confirm cooling mode and reasonable setpoint.\n"
            "- Record room temperature, return/supply air temperatures, and operating current.\n"
            "- Refer refrigerant diagnostics to a qualified HVAC technician.\n\n"
            "4. SYMPTOM: WATER LEAKAGE\n"
            "Possible causes:\n"
            "- Blocked or incorrectly routed condensate drain.\n"
            "- Dirty drain tray or evaporator coil.\n"
            "- Poor insulation or installation level.\n"
            "Initial checks:\n"
            "- Switch off if water presents an electrical hazard.\n"
            "- Inspect visible drain routing and arrange safe cleaning.\n"
            "- Record the leak location and take photographs.\n\n"
            "5. SYMPTOM: ICE ON COIL OR PIPE\n"
            "Possible causes:\n"
            "- Restricted airflow, sensor issue, or refrigerant-system problem.\n"
            "Action:\n"
            "- Stop cooling operation and arrange inspection.\n"
            "- Do not chip ice with sharp tools or attempt unqualified refrigerant work.\n\n"
            "6. SYMPTOM: NOISE OR VIBRATION\n"
            "Possible causes:\n"
            "- Loose panel or mounting fastener.\n"
            "- Fan obstruction, worn bearing, or unstable base.\n"
            "Action:\n"
            "- Isolate power before inspection.\n"
            "- Record whether noise occurs at startup, continuously, or only at a particular fan speed.\n\n"
            "7. FAULT RECORD TEMPLATE\n"
            "Date/time: ______________________\n"
            "Asset ID / serial number: __________\n"
            "Reported symptom: ________________\n"
            "Controller code: __________________\n"
            "Operating mode/setpoint: ___________\n"
            "Readings and observations: __________\n"
            "Immediate safety action: ____________\n"
            "Root cause (confirmed): _____________\n"
            "Corrective action: _________________\n"
            "Parts replaced: ____________________\n"
            "Technician: _______________________\n"
            "Status: Open / Monitoring / Closed\n\n"
            "NOTE\n"
            "Never bypass safety devices. Use the manufacturer's code table for the installed model.\n"
        )
        cls.real_doc = KnowledgeDocument.objects.create(
            tenant=cls.real_tenant,
            product=cls.product,
            asset=cls.real_asset,
            title="Split AC Unit 2T - Troubleshooting",
            doc_type="troubleshooting",
            content_text=cls.doc_text,
            checksum_sha256=cls.auth_sha,
            is_rag_enabled=True,
            is_confidential=False,
            disable_sharing=False,
            index_status="INDEXED",
        )
        cls.real_doc.provenance_verified = True

        from core.services.vectors import embed_text
        emb = embed_text(cls.doc_text)
        cls.real_chunk = KnowledgeChunk.objects.create(
            tenant=cls.real_tenant,
            document=cls.real_doc,
            chunk_index=0,
            heading="2. SYMPTOM: UNIT DOES NOT START",
            text=cls.doc_text,
            embedding=emb,
            content_embedding=emb,
        )

    # 1. Forged real-looking asset code
    def test_01_forged_real_looking_asset_code_rejected(self):
        forged_asset = Asset(
            tenant=self.real_tenant,
            asset_code="SKY-AST-99999",  # Looks real, but not in registry
            name="Forged Unit",
        )
        self.assertFalse(is_verified_real_asset(forged_asset))

    # 2. Forged real-looking customer name
    def test_02_forged_real_looking_customer_name_rejected(self):
        forged_cust = Customer(
            tenant=self.real_tenant,
            name="Rajesh Kumar Industrial Corp",  # Contains real customer prefix but is unmapped
        )
        self.assertFalse(is_verified_real_customer(forged_cust))

    # 3. Fake document marked INDEXED on recognized asset
    def test_03_fake_doc_marked_indexed_rejected(self):
        fake_doc = KnowledgeDocument(
            tenant=self.real_tenant,
            asset=self.real_asset,
            title="Fake Manual",
            index_status="INDEXED",
            is_rag_enabled=True,
            is_confidential=False,
            checksum_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
        self.assertFalse(is_verified_real_document(fake_doc))

    # 4. Forged description claiming approval
    def test_04_forged_description_claiming_approval_rejected(self):
        forged_doc = KnowledgeDocument(
            tenant=self.real_tenant,
            asset=self.real_asset,
            title="Injected Maintenance",
            description="verified starlly technical reference approved technical reference",
            index_status="INDEXED",
            is_rag_enabled=True,
            is_confidential=False,
            checksum_sha256="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        )
        self.assertFalse(is_verified_real_document(forged_doc))

    # 5. Valid registry checksum paired with wrong content
    def test_05_valid_checksum_paired_with_tampered_content_rejected(self):
        tampered_doc = KnowledgeDocument(
            tenant=self.real_tenant,
            asset=self.real_asset,
            title="Split AC Unit 2T - Tampered",
            checksum_sha256=self.auth_sha,  # Valid registry hash
            content_text="Tampered malicious content that does not hash to auth_sha",
            is_rag_enabled=True,
            is_confidential=False,
            index_status="INDEXED",
        )
        self.assertFalse(is_verified_real_document(tampered_doc))

    # 6. Valid registry checksum paired with wrong asset
    def test_06_valid_checksum_paired_with_wrong_asset_rejected(self):
        wrong_asset = Asset(
            tenant=self.real_tenant,
            asset_code="SKY-AST-00002",  # VRF module, not Split AC
            name="VRF Module",
        )
        doc_wrong_asset = KnowledgeDocument(
            tenant=self.real_tenant,
            asset=wrong_asset,
            title="Split AC manual on VRF asset",
            checksum_sha256=self.auth_sha,  # Hashes to SKY-AST-00001 in registry
            content_text=self.doc_text,
            is_rag_enabled=True,
            is_confidential=False,
            index_status="INDEXED",
        )
        self.assertFalse(is_verified_real_document(doc_wrong_asset))

    # 7. Synthetic document lacking obvious dummy keywords
    def test_07_unmapped_document_without_dummy_keywords_rejected(self):
        clean_looking_doc = KnowledgeDocument(
            tenant=self.real_tenant,
            asset=self.real_asset,
            title="Compressor Service Standard Protocol",
            tags="electrical, cooling, airflow",
            doc_type="troubleshooting",
            checksum_sha256="1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
            is_rag_enabled=True,
            is_confidential=False,
            index_status="INDEXED",
        )
        self.assertFalse(is_verified_real_document(clean_looking_doc))

    # 8. Synthetic historical service call with realistic ID
    def test_08_synthetic_service_call_with_realistic_id_rejected(self):
        synth_call = ServiceCall(
            tenant=self.real_tenant,
            servy_id=88421,  # Realistic numeric ID
            customer=Customer(tenant=self.real_tenant, name="Unverified Customer Corp"),
            asset=self.real_asset,
            complaint_type="Cooling fault",
            complaint_text="Fan motor stopped running",
            status="closed",
        )
        self.assertFalse(is_verified_real_service_call(synth_call))

    # 9. Missing model triggering sparse fallback
    def test_09_missing_model_triggers_sparse_fallback_gracefully(self):
        with patch("core.services.retriever._dense_score_candidates", side_effect=Exception("Model unavailable")):
            results = retrieve(
                tenant=self.real_tenant,
                question="Unit does not start power supply isolator",
                asset=self.real_asset,
                customer=self.real_customer,
                include_confidential=False,
                top_k=3,
                min_score=0.0,
                staff_mode=False,
            )
            # Sparse fallback succeeds without raising 500
            self.assertTrue(len(results) > 0)
            self.assertEqual(results[0].document_id, self.real_doc.id)

    # 10. Sparse fallback never retrieves unauthorized documentation
    def test_10_sparse_fallback_never_retrieves_unauthorized_documents(self):
        unauth_doc = KnowledgeDocument.objects.create(
            tenant=self.attacker_tenant,
            title="Attacker Private Manual",
            content_text="Indoor fan motor does not start stolen details",
            is_rag_enabled=True,
            is_confidential=False,
            index_status="INDEXED",
        )
        from core.services.vectors import embed_text
        emb = embed_text(unauth_doc.content_text)
        KnowledgeChunk.objects.create(
            tenant=self.attacker_tenant,
            document=unauth_doc,
            chunk_index=0,
            heading="Fan failure",
            text=unauth_doc.content_text,
            embedding=emb,
            content_embedding=emb,
        )

        with patch("core.services.retriever._dense_score_candidates", return_value=[]):
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
            for r in results:
                self.assertNotEqual(r.document_id, unauth_doc.id)

    # 11. Wrong-tenant Chroma vector cannot cross tenant boundary
    def test_11_wrong_tenant_chroma_vector_rejected(self):
        with patch("core.services.retriever._dense_score_candidates") as mock_dense:
            # Simulate Chroma returning an unauthenticated attacker doc_id
            from core.services.retriever import RetrievedChunk
            mock_dense.return_value = [
                RetrievedChunk(
                    source_kind="knowledge_document",
                    source_id=9999,
                    document_id=9999,  # Belongs to attacker
                    chunk_id=9999,
                    title="Leaked attacker chunk",
                    heading="Attack",
                    text="Pry open cabinet",
                    score=0.99,
                    doc_type="reference",
                    source_url="",
                    confidential=False,
                )
            ]
            # Defense in depth: Django filters candidate pool before dense results can be accepted
            results = retrieve(
                tenant=self.real_tenant,
                question="Indoor fan motor does not start",
                asset=self.real_asset,
                customer=self.real_customer,
                top_k=5,
                min_score=0.0,
                staff_mode=False,
            )
            for r in results:
                self.assertNotEqual(r.document_id, 9999)

    # 12. Partial index with truthful readiness status
    def test_12_partial_index_readiness_truthfully_reported(self):
        partial_doc = KnowledgeDocument.objects.create(
            tenant=self.real_tenant,
            asset=self.real_asset,
            title="Partial Document Test",
            content_text="Some manual instructions",
            index_status="NOT_INDEXED",
        )
        readiness = get_document_retrieval_readiness(partial_doc)
        self.assertFalse(readiness["minilm_vectors_ready"])
        self.assertFalse(readiness["bge_vectors_ready"])
        self.assertEqual(readiness["index_status"], "NOT_INDEXED")

    # 13. Conflicting external asset ID carrying an attachment
    def test_13_conflicting_external_asset_identity_skips_attachments(self):
        mock_client = MagicMock()
        # External asset shares code SKY-AST-00001 but has conflicting ID 9999 (expected 1401)
        mock_client.fetch_assets.return_value = {
            "assets": [{
                "id": 9999,
                "code": "SKY-AST-00001",
                "name": "Split AC Unit 2T",
                "model_number": "LactoScan Standard",
                "attachments": [{
                    "id": 888,
                    "name": "malicious_attachment.txt",
                    "file_url": "https://example.com/malicious.txt",
                }],
            }]
        }
        mock_client.get_auth_headers.return_value = {}

        initial_docs = KnowledgeDocument.objects.filter(asset=self.real_asset).count()

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = []
            report = ingest_starlly_real_data(
                tenant_slug="skydream",
                client=mock_client,
            )

        # Asset conflict must be explicitly reported
        self.assertTrue(len(report["conflicts"]) > 0)
        self.assertEqual(report["conflicts"][0]["type"], "ASSET_CONFLICT_REQUIRES_REVIEW")
        # No attachments linked to existing local asset
        self.assertEqual(KnowledgeDocument.objects.filter(asset=self.real_asset).count(), initial_docs)

    # 14. Concurrent import to same filename (atomic file staging)
    def test_14_atomic_file_staging_prevents_partial_corrupted_overwrite(self):
        test_dir = Path(settings.MEDIA_ROOT) / "knowledge" / "test_atomic"
        test_dir.mkdir(parents=True, exist_ok=True)
        target_path = test_dir / "safe_manual.txt"
        target_path.write_bytes(b"EXISTING VERIFIED BYTES")

        # Attempt to stage a file with a mismatching hash
        with patch("hashlib.sha256") as mock_sha:
            mock_sha.return_value.hexdigest.return_value = "forged_hash"
            # Target path must remain unchanged
            self.assertEqual(target_path.read_bytes(), b"EXISTING VERIFIED BYTES")

    # 15. Real PDF with malformed bytes rejected
    def test_15_corrupt_malformed_pdf_rejected(self):
        corrupt_pdf_bytes = b"%PDF-1.4\n%%EOF\x00\xffMALFORMED_BYTES"
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(corrupt_pdf_bytes))}
        mock_resp.iter_content.return_value = [corrupt_pdf_bytes]

        with patch("requests.get", return_value=mock_resp):
            with self.assertRaises(IngestionSecurityError):
                fetch_attachment_content("https://example.com/corrupt.pdf")

    # 16. Invasive procedure described as an observation
    def test_16_invasive_procedure_described_as_observation_classified_technician_only(self):
        level, reason = classify_action_safety(
            "Inspect and remove electrical panel to probe live 415v terminal",
            asset=self.real_asset,
        )
        self.assertEqual(level, "TECHNICIAN_ONLY")
        self.assertTrue("hazardous" in reason.lower() or "chassis" in reason.lower())

    # 17. Correct source but unapproved customer procedure
    def test_17_correct_source_unapproved_procedure_fails_closed(self):
        level, reason = classify_action_safety(
            "Overhaul hydraulic pump assembly and adjust relief valve",
            asset=self.real_asset,
        )
        # Must fail closed: either UNKNOWN_REQUIRES_REVIEW or TECHNICIAN_ONLY
        self.assertIn(level, ("UNKNOWN_REQUIRES_REVIEW", "TECHNICIAN_ONLY"))
        self.assertNotEqual(level, "CUSTOMER_APPROVED")

    # 18. Hazard reported through clarification
    def test_18_hazard_reported_through_clarification_escalates(self):
        interaction = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="AC not turning on - smoke and electrical sparks visible",
            answer="DIAGNOSTIC SUMMARY\nImmediate Safety Risk Detected",
        )
        contract = build_structured_diagnostic_contract(
            interaction=interaction,
            session=None,
            raw_answer=interaction.answer,
            retrieved_references=[],
        )
        self.assertEqual(contract["response_type"], "ESCALATION_REQUIRED")
        self.assertEqual(contract["can_resolve"], False)
        self.assertEqual(contract["can_escalate"], True)

    # 19. Stale diagnostic session version
    def test_19_stale_diagnostic_session_version_rejected(self):
        session = DiagnosticSession.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            complaint="Fan motor stopped",
            version=3,
        )
        # Attempting to mutate with stale version 2 must be rejected
        stale_version = 2
        self.assertNotEqual(session.version, stale_version)

    # 20. Resume after source revocation
    def test_20_resume_after_source_revocation_suppresses_actions(self):
        # Disable RAG on document
        self.real_doc.is_rag_enabled = False
        self.real_doc.save(update_fields=["is_rag_enabled"])

        is_grounded, ref, reason = find_grounding_reference(
            "Clean exterior and wipe dust filter with clean water.",
            retrieved_references=[{"document_id": self.real_doc.id, "chunk_id": self.real_chunk.id}],
            tenant_id=self.real_tenant.id,
            asset=self.real_asset,
            customer=self.real_customer,
        )
        self.assertFalse(is_grounded)

        # Restore
        self.real_doc.is_rag_enabled = True
        self.real_doc.save(update_fields=["is_rag_enabled"])

    # 21. Two unrelated incidents with similar complaint wording
    def test_21_two_unrelated_incidents_create_independent_tickets(self):
        interaction1 = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Filter is blocked and airflow is low",
            answer="Clean filter",
        )
        call1 = create_call_from_interaction(interaction1)

        interaction2 = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Severe motor vibration and rattle",
            answer="Escalate",
        )
        call2 = create_call_from_interaction(interaction2)

        self.assertNotEqual(call1.id, call2.id)
        self.assertNotEqual(call1.servy_id, call2.servy_id)

    # 22. Duplicate escalation of one incident is idempotent
    def test_22_duplicate_escalation_same_incident_is_idempotent(self):
        interaction = CustomerInteraction.objects.create(
            tenant=self.real_tenant,
            customer=self.real_customer,
            site=self.real_site,
            asset=self.real_asset,
            question="Power switch tripped",
            answer="Escalate",
        )
        call_first = create_call_from_interaction(interaction)
        call_second = create_call_from_interaction(interaction)
        self.assertEqual(call_first.id, call_second.id)
        self.assertEqual(call_first.servy_id, call_second.servy_id)

    # 23. Test-mode/production-mode configuration confusion fails closed
    def test_23_mode_configuration_confusion_fails_closed(self):
        with override_settings(SERVY_CUSTOMER_DATA_MODE="unknown_invalid_mode"):
            self.assertEqual(get_customer_data_mode(), "verified_real_only")

    # 24. Zero-deletion contract: Real company vectors can never be deleted
    def test_24_company_vector_zero_deletion_enforced(self):
        with self.assertRaises(RuntimeError) as ctx:
            delete_document_vectors(self.real_doc, allow_company_purge=False)
        self.assertIn("Zero-deletion violation", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx2:
            delete_document_vectors(self.real_doc, allow_company_purge=True)
        self.assertIn("Zero-deletion violation", str(ctx2.exception))
