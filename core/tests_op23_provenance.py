"""Disposable SQLite/MySQL tests of read-only source mapping and DB uniqueness."""
from django.db import IntegrityError, transaction
from django.test import TestCase

from core.ml.op23_provenance_reconcile import reconcile_document
from core.ml.op23_source_admission import sha256
from core.models import Asset, KnowledgeChunk, KnowledgeDocument, SourceProvenanceRecord, Tenant


class ProvenanceDryRunTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="Source mapping test", slug="source-mapping-test")
        self.asset = Asset.objects.create(tenant=self.tenant, name="Test compressor", asset_code="T-COMP-1", model_number="MC-1")
        self.text = "COMPRESSOR MANUAL\nInspect the compressor casing for damage before operation."
        self.raw_hash = sha256(b"private test source")
        self.document = KnowledgeDocument.objects.create(
            tenant=self.tenant, asset=self.asset, title="Test source", source_type="text",
            content_text=self.text, checksum_sha256=self.raw_hash,
        )
        self.chunk = KnowledgeChunk.objects.create(tenant=self.tenant, document=self.document, chunk_index=0,
                                                   heading="Inspection", text="Inspect the compressor casing for damage before operation.")
        self.source = {"source_verification": "VERIFIED_ORIGINAL_COMPANY_SOURCE",
                       "source_organisation": "273", "tenant_id": self.tenant.id,
                       "asset_code": self.asset.asset_code, "external_asset_id": 332,
                       "external_attachment_id": 99, "equipment_model": "MC-1",
                       "raw_sha256": self.raw_hash, "text_sha256": sha256(self.text.encode())}

    def test_mapping_dry_run_is_read_only_and_idempotent(self):
        proposal = reconcile_document(self.source, self.asset, self.document, self.text, expected_org="273")
        self.assertEqual(proposal["status"], "WOULD_INSERT_PENDING_AUTHORIZED_WRITE")
        self.assertEqual(proposal["proposed_mapping"]["external_id"], "99")
        self.assertEqual(SourceProvenanceRecord.objects.count(), 0)
        record = SourceProvenanceRecord.objects.create(
            tenant=self.tenant, source_system="starlly", source_org="273", external_object_type="document",
            external_id="99", local_object_type="KnowledgeDocument", local_object_id=self.document.id,
            asset_code=self.asset.asset_code, raw_attachment_sha256=self.raw_hash,
        )
        existing = reconcile_document(self.source, self.asset, self.document, self.text, record, expected_org="273")
        self.assertEqual(existing["status"], "EXISTING_MATCH")
        self.assertEqual(SourceProvenanceRecord.objects.count(), 1)

    def test_wrong_tenant_asset_hash_and_conflicting_record_rejected(self):
        source = dict(self.source, tenant_id=999, external_attachment_id=100, raw_sha256="0" * 64)
        mismatch = reconcile_document(source, self.asset, self.document, self.text, expected_org="273")
        self.assertEqual(mismatch["status"], "CONFLICT")
        self.assertIn("tenant_mismatch", mismatch["reasons"])
        self.assertIn("document_asset_or_raw_hash_mismatch", mismatch["reasons"])
        record = SourceProvenanceRecord.objects.create(
            tenant=self.tenant, source_system="starlly", source_org="273", external_object_type="document",
            external_id="99", local_object_type="KnowledgeDocument", local_object_id=999,
            raw_attachment_sha256=self.raw_hash,
        )
        conflict = reconcile_document(self.source, self.asset, self.document, self.text, record, expected_org="273")
        self.assertIn("existing_provenance_conflicts", conflict["reasons"])

    def test_mysql_unique_external_identity_contract(self):
        fields = dict(tenant=self.tenant, source_system="starlly", source_org="273",
                      external_object_type="document", external_id="99",
                      local_object_type="KnowledgeDocument", local_object_id=self.document.id)
        SourceProvenanceRecord.objects.create(**fields)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SourceProvenanceRecord.objects.create(**fields)
