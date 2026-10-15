"""
Adversarial Unit Tests for Evidence Sufficiency and Claim Grounding (Operation 12.0).

Verifies the 8 Required Adversarial Attacks from Part 4:
1. Same vocabulary, different fault
2. Same fault code, different manufacturer
3. Same symptom, wrong model
4. Similar introductory paragraph, different procedure
5. Relevant topic, unsupported conclusion
6. Valid excerpt but incorrect answer
7. Incorrect numerical threshold
8. Genuine citation that does not support the claim.
"""
from django.test import TestCase
from unittest.mock import MagicMock
from core.models import Tenant, ProductDomain, ProductCategory, Product, Asset, KnowledgeDocument, KnowledgeChunk
from core.services.retriever import RetrievedChunk
from core.services.llm import (
    check_evidence_sufficiency,
    generate_answer,
    _extractive_answer,
    evaluate_chunk_claim_support,
    INSUFFICIENT_EVIDENCE_MESSAGE,
)


class EvidenceSufficiencyAttackTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tenant = Tenant.objects.create(name="Attack Test Tenant", slug="attack-tenant")

        # Two distinct manufacturers / products
        cls.prod_carrier = Product.objects.create(tenant=cls.tenant, name="Carrier Chiller 30XA")
        cls.prod_trane = Product.objects.create(tenant=cls.tenant, name="Trane Chiller RTHD")

        # Assets
        cls.asset_carrier = Asset.objects.create(
            tenant=cls.tenant, product=cls.prod_carrier,
            name="Carrier Chiller #1", asset_code="CAR-01", model_number="30XA-500",
        )
        cls.asset_trane = Asset.objects.create(
            tenant=cls.tenant, product=cls.prod_trane,
            name="Trane Chiller #1", asset_code="TRA-01", model_number="RTHD-250",
        )

    def test_attack_1_same_vocabulary_different_fault(self):
        """Attack 1: Casual vocabulary overlap (e.g. 'fan', 'motor') on a completely different fault
        must be rejected as insufficient evidence."""
        # Query asks about inverter drive communication fault
        question = "Indoor fan motor inverter drive communication fault troubleshooting"
        # Chunk has vocabulary overlap ('fan', 'motor') but is about physical bracket mounting screws
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=1, document_id=1, chunk_id=1,
            title="Fan Assembly", heading="Bracket Mounting",
            text="Secure the fan motor bracket mounting screws to the frame using M6 bolts.",
            score=0.40, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=True, source_verified=True, exact_identifier_match=False,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "no-grounded-overlap")

        answer, _ = generate_answer(question, self.asset_carrier, [chunk])
        self.assertEqual(answer, INSUFFICIENT_EVIDENCE_MESSAGE)

    def test_attack_2_same_fault_code_different_manufacturer(self):
        """Attack 2: Error code E04 belongs to Trane manual, but user query is for Carrier equipment.
        Equipment applicability check must fail closed."""
        question = "How to clear fault code E04?"
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=2, document_id=2, chunk_id=2,
            title="Trane Fault Guide", heading="Fault E04",
            text="Fault E04 indicates Trane oil pressure differential failure.",
            score=0.88, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=False,  # Bound to Trane, query is for Carrier
            source_verified=True, exact_identifier_match=True,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "no-applicable-equipment-evidence")

    def test_attack_3_same_symptom_wrong_model(self):
        """Attack 3: Same symptom (high condenser pressure) but document belongs to wrong equipment model."""
        question = "High condenser pressure warning troubleshooting"
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=3, document_id=3, chunk_id=3,
            title="Old Model Manual", heading="Condenser Pressure",
            text="High condenser pressure is caused by dirty water coils or scaling.",
            score=0.80, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=False,  # Excluded by model hierarchy
            source_verified=True, exact_identifier_match=False,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "no-applicable-equipment-evidence")

    def test_attack_4_similar_introductory_paragraph_different_procedure(self):
        """Attack 4: Introductory generic manual paragraph does not contain procedure claim support."""
        question = "What are the exact diagnostic steps for refrigerant leak detection?"
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=4, document_id=4, chunk_id=4,
            title="Safety and Overview", heading="General Overview",
            text="Welcome to the technical service manual. Always read all diagnostic instructions carefully before performing any maintenance on this unit.",
            score=0.35, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=True, source_verified=True, exact_identifier_match=False,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "no-grounded-overlap")

    def test_attack_5_relevant_topic_unsupported_conclusion(self):
        """Attack 5: Topic is related, but missing specific technical identifier requested in query."""
        question = "What does error code ERR-888 indicate on the control panel?"
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=5, document_id=5, chunk_id=5,
            title="Control Panel Guide", heading="Control Panel Alarms",
            text="Control panel displays alarms from ERR-001 through ERR-050.",
            score=0.60, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=True, source_verified=True, exact_identifier_match=False,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "missing-required-identifier")

    def test_attack_6_valid_excerpt_but_incorrect_answer(self):
        """Attack 6: Passage contains some words but fails multi-term overlap requirement."""
        question = "Evaporator freeze protection thermostat setting calibration"
        # Only matches 'evaporator' casually
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=6, document_id=6, chunk_id=6,
            title="Evaporator Cleaning", heading="Cleaning",
            text="Wash evaporator coil surface with low pressure water spray.",
            score=0.30, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=True, source_verified=True, exact_identifier_match=False,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "no-grounded-overlap")

    def test_attack_7_incorrect_numerical_threshold(self):
        """Attack 7: Query asks about 450 psi safety trip threshold; passage discusses 120 psi."""
        question = "What should I inspect if high pressure switch trips at 450 psi?"
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=7, document_id=7, chunk_id=7,
            title="Pressure Switches", heading="Low Pressure Switch",
            text="Low pressure cut-out switch trips when suction pressure drops below 120 psi.",
            score=0.65, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=True, source_verified=True, exact_identifier_match=False,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "no-grounded-overlap")

    def test_attack_8_genuine_citation_that_does_not_support_claim(self):
        """Attack 8: Real document in KB with high score but unverified source trust or missing claim."""
        question = "Compressor oil pump failure symptoms"
        chunk = RetrievedChunk(
            source_kind="knowledge_document", source_id=8, document_id=8, chunk_id=8,
            title="Company Manual", heading="Unverified Section",
            text="General notes on compressor components.",
            score=0.70, doc_type="Manual", source_url="", confidential=False,
            equipment_applicable=True,
            source_verified=False,  # Unverified source
            exact_identifier_match=False,
        )
        is_sufficient, reason = check_evidence_sufficiency(question, self.asset_carrier, [chunk])
        self.assertFalse(is_sufficient)
        self.assertEqual(reason, "all-sources-unverified")
