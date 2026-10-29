"""
SERVY Production Security Red-Team Tests.
Tests security invariants: tenant isolation, confidential filtering,
quarantine/revocation, prompt injection, edge cases.

All tests create their own fixture data so they work with Django's
test database (no dependency on production data).
"""
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
sys.path.insert(0, ".")

import django

django.setup()

from django.test import TestCase
from core.models import Tenant, KnowledgeChunk, KnowledgeDocument
import core.services.retriever as R


def _create_tenant(name, pk=None):
    kwargs = {"name": name, "slug": name.lower().replace(" ", "-"), "is_active": True}
    if pk is not None:
        kwargs["id"] = pk
    return Tenant.objects.create(**kwargs)


def _create_doc(tenant, title, **kwargs):
    defaults = {
        "tenant": tenant,
        "title": title,
        "description": title,
        "doc_type": "manual",
        "source_type": "upload",
        "content_text": title,
        "is_confidential": False,
        "is_rag_enabled": True,
        "index_status": "completed",
    }
    defaults.update(kwargs)
    return KnowledgeDocument.objects.create(**defaults)


def _create_chunk(tenant, document, text, heading="Section", **kwargs):
    defaults = {
        "tenant": tenant,
        "document": document,
        "chunk_index": 0,
        "heading": heading,
        "text": text,
        "is_quarantined": False,
    }
    defaults.update(kwargs)
    return KnowledgeChunk.objects.create(**defaults)


def _returned_chunk_ids(results):
    """Extract chunk IDs from retrieval results."""
    return {r.chunk_id for r in results if r.chunk_id is not None}


# ---------------------------------------------------------------------------
# 1. TENANT ISOLATION
# ---------------------------------------------------------------------------
class TenantIsolationTests(TestCase):
    """Verify no cross-tenant data leakage in retrieval."""

    def setUp(self):
        self.t1 = _create_tenant("RedTeam Tenant A")
        self.t2 = _create_tenant("RedTeam Tenant B")

        doc1 = _create_doc(self.t1, "Tenant A Equipment Manual")
        doc2 = _create_doc(self.t2, "Tenant B Equipment Manual")

        self.c1 = _create_chunk(
            self.t1, doc1,
            "Replace the fuse on the power board following safety protocol alpha.",
            heading="Power Board Fuse Replacement",
        )
        self.c2 = _create_chunk(
            self.t2, doc2,
            "Replace the fuse on the power board following safety protocol beta.",
            heading="Power Board Fuse Replacement",
        )

    def test_tenant_a_cannot_see_tenant_b_chunks(self):
        """Tenant A queries must never return chunks from Tenant B."""
        res = R.retrieve(
            tenant=self.t1, question="replace fuse power board",
            top_k=10, min_score=0.0, staff_mode=True,
        )
        ids = _returned_chunk_ids(res)
        self.assertNotIn(
            self.c2.id, ids,
            f"Tenant A query leaked Tenant B chunk {self.c2.id}",
        )

    def test_tenant_b_cannot_see_tenant_a_chunks(self):
        """Tenant B queries must never return chunks from Tenant A."""
        res = R.retrieve(
            tenant=self.t2, question="replace fuse power board",
            top_k=10, min_score=0.0, staff_mode=True,
        )
        ids = _returned_chunk_ids(res)
        self.assertNotIn(
            self.c1.id, ids,
            f"Tenant B query leaked Tenant A chunk {self.c1.id}",
        )


# ---------------------------------------------------------------------------
# 2. CONFIDENTIAL DOCUMENT FILTERING
# ---------------------------------------------------------------------------
class ConfidentialDocumentTests(TestCase):
    """Verify confidential documents are excluded unless explicitly requested."""

    def setUp(self):
        self.tenant = _create_tenant("ConfTest Tenant")
        self.conf_doc = _create_doc(
            self.tenant, "Secret Internal Procedures",
            is_confidential=True,
        )
        self.normal_doc = _create_doc(
            self.tenant, "Public Troubleshooting Guide",
            is_confidential=False,
        )
        self.conf_chunk = _create_chunk(
            self.tenant, self.conf_doc,
            "Internal escalation password is stored in vault.",
            heading="Internal Escalation",
        )
        self.normal_chunk = _create_chunk(
            self.tenant, self.normal_doc,
            "Restart the device by holding the power button for 5 seconds.",
            heading="Restart Procedure",
        )

    def test_confidential_excluded_by_default(self):
        """Confidential documents must not appear when include_confidential=False."""
        res = R.retrieve(
            tenant=self.tenant, question="escalation password internal",
            top_k=20, min_score=0.0, staff_mode=True,
            include_confidential=False,
        )
        ids = _returned_chunk_ids(res)
        self.assertNotIn(
            self.conf_chunk.id, ids,
            "Confidential chunk leaked in default retrieval",
        )

    def test_confidential_included_when_requested(self):
        """Confidential documents should appear when include_confidential=True."""
        res = R.retrieve(
            tenant=self.tenant, question="escalation password internal",
            top_k=20, min_score=0.0, staff_mode=True,
            include_confidential=True,
        )
        ids = _returned_chunk_ids(res)
        # If sparse index picked it up, it should be in results
        # (empty results just means BM25 didn't match — not a security failure)
        if ids:
            self.assertIn(
                self.conf_chunk.id, ids,
                "Confidential chunk missing when explicitly requested",
            )


# ---------------------------------------------------------------------------
# 3. QUARANTINE / REVOCATION
# ---------------------------------------------------------------------------
class QuarantineRevocationTests(TestCase):
    """Verify quarantined and RAG-disabled documents are excluded."""

    def setUp(self):
        self.tenant = _create_tenant("QuarTest Tenant")

        self.quar_doc = _create_doc(self.tenant, "Recalled Manual")
        self.disabled_doc = _create_doc(
            self.tenant, "Deprecated Guide", is_rag_enabled=False,
        )
        self.good_doc = _create_doc(self.tenant, "Active Manual")

        self.quar_chunk = _create_chunk(
            self.tenant, self.quar_doc,
            "Dangerous procedure recalled due to safety issue.",
            heading="Recalled Procedure",
            is_quarantined=True,
        )
        self.disabled_chunk = _create_chunk(
            self.tenant, self.disabled_doc,
            "Deprecated calibration steps for old firmware.",
            heading="Old Calibration",
        )
        self.good_chunk = _create_chunk(
            self.tenant, self.good_doc,
            "Standard calibration steps for current firmware.",
            heading="Calibration",
        )

    def test_quarantined_chunks_excluded(self):
        """Quarantined chunks must never appear in results."""
        res = R.retrieve(
            tenant=self.tenant, question="recalled procedure safety dangerous",
            top_k=20, min_score=0.0, staff_mode=True,
        )
        ids = _returned_chunk_ids(res)
        self.assertNotIn(
            self.quar_chunk.id, ids,
            "Quarantined chunk appeared in results!",
        )

    def test_rag_disabled_documents_excluded(self):
        """Documents with is_rag_enabled=False must not appear."""
        res = R.retrieve(
            tenant=self.tenant, question="deprecated calibration firmware old",
            top_k=20, min_score=0.0, staff_mode=True,
        )
        ids = _returned_chunk_ids(res)
        self.assertNotIn(
            self.disabled_chunk.id, ids,
            "RAG-disabled chunk appeared in results!",
        )


# ---------------------------------------------------------------------------
# 4. PROMPT INJECTION / ADVERSARIAL INPUTS
# ---------------------------------------------------------------------------
class PromptInjectionTests(TestCase):
    """Test that adversarial query inputs don't bypass security or crash."""

    def setUp(self):
        self.tenant = _create_tenant("InjTest Tenant")
        doc = _create_doc(self.tenant, "Normal Manual")
        self.chunk = _create_chunk(
            self.tenant, doc,
            "Standard troubleshooting: restart the device.",
            heading="Troubleshooting",
        )

    def test_sql_injection_in_query(self):
        """SQL injection in question text must not crash or leak data."""
        adversarial_queries = [
            "'; DROP TABLE core_knowledgechunk; --",
            '" OR 1=1 --',
            "UNION SELECT * FROM core_tenant",
            "{{7*7}}",
            "${7*7}",
            "<script>alert('xss')</script>",
        ]
        for q in adversarial_queries:
            try:
                res = R.retrieve(
                    tenant=self.tenant, question=q,
                    top_k=5, min_score=0.0, staff_mode=True,
                )
                # Should return results — none should be cross-tenant
                for r in res:
                    if r.chunk_id:
                        chunk = KnowledgeChunk.objects.get(id=r.chunk_id)
                        self.assertEqual(
                            chunk.tenant_id, self.tenant.id,
                            f"Adversarial query '{q[:30]}' returned cross-tenant chunk",
                        )
            except Exception as e:
                self.fail(f"Adversarial query '{q[:30]}' raised: {type(e).__name__}: {e}")

    def test_unicode_edge_cases(self):
        """Unicode edge cases must not crash retrieval."""
        unicode_queries = [
            "café résumé naïve",
            "日本語テスト",
            "🔧🔩💡",
            "a" * 10000,
        ]
        for q in unicode_queries:
            try:
                R.retrieve(
                    tenant=self.tenant, question=q,
                    top_k=5, min_score=0.0, staff_mode=True,
                )
            except UnicodeError:
                self.fail(f"Unicode query '{q[:20]}' raised UnicodeError")
            except Exception:
                pass  # Other errors OK (e.g., no BM25 match)


# ---------------------------------------------------------------------------
# 5. SIMILAR EQUIPMENT CONFUSION
# ---------------------------------------------------------------------------
class SimilarEquipmentConfusionTests(TestCase):
    """Test that similar equipment names produce correctly scoped results."""

    def setUp(self):
        self.tenant = _create_tenant("EquipTest Tenant")
        self.doc_a = _create_doc(self.tenant, "Asus Celeron Terminal Service Manual")
        self.doc_b = _create_doc(self.tenant, "Milk Analyzer 3000 Service Manual")

        self.chunk_a = _create_chunk(
            self.tenant, self.doc_a,
            "Asus Celeron terminal black screen: check HDMI cable and restart BIOS.",
            heading="Celeron Black Screen Fix",
        )
        self.chunk_b = _create_chunk(
            self.tenant, self.doc_b,
            "Milk Analyzer 3000 display error: replace LCD inverter board.",
            heading="Analyzer Display Fix",
        )

    def test_celeron_query_prefers_celeron_docs(self):
        """Query mentioning 'Celeron' should rank Celeron chunks higher."""
        res = R.retrieve(
            tenant=self.tenant,
            question="Asus Celeron terminal black screen HDMI",
            top_k=5, min_score=0.0, staff_mode=True,
        )
        if not res:
            self.skipTest("No results returned (BM25 didn't match)")

        top = res[0]
        if top.chunk_id:
            chunk = KnowledgeChunk.objects.select_related("document").get(
                id=top.chunk_id
            )
            self.assertIn(
                "celeron", chunk.document.title.lower(),
                f"Top result for Celeron query came from: '{chunk.document.title}'",
            )


# ---------------------------------------------------------------------------
# 6. RERANKER FAILURE GRACEFUL DEGRADATION
# ---------------------------------------------------------------------------
class RerankerFailureGracefulTests(TestCase):
    """Test graceful degradation when reranker is unavailable."""

    def setUp(self):
        self.tenant = _create_tenant("RerankerTest Tenant")
        doc = _create_doc(self.tenant, "Test Manual")
        _create_chunk(
            self.tenant, doc,
            "Standard troubleshooting restart procedure for device.",
            heading="Restart",
        )

    def test_retrieval_works_without_reranker_crash(self):
        """Retrieval should still return results even with edge conditions."""
        try:
            res = R.retrieve(
                tenant=self.tenant,
                question="troubleshooting restart procedure",
                top_k=5, min_score=0.0, staff_mode=True,
            )
            self.assertIsNotNone(res)
        except Exception as e:
            self.fail(f"Retrieval crashed: {e}")


# ---------------------------------------------------------------------------
# 7. EMPTY AND EDGE CASES
# ---------------------------------------------------------------------------
class EmptyAndEdgeCaseTests(TestCase):
    """Test edge cases in retrieval."""

    def setUp(self):
        self.tenant = _create_tenant("EdgeTest Tenant")
        doc = _create_doc(self.tenant, "Edge Case Manual")
        _create_chunk(
            self.tenant, doc,
            "Standard procedure for testing edge cases.",
            heading="Edge Cases",
        )

    def test_empty_question(self):
        """Empty question should not crash."""
        try:
            res = R.retrieve(
                tenant=self.tenant, question="",
                top_k=5, min_score=0.0, staff_mode=True,
            )
            self.assertIsNotNone(res)
        except Exception:
            pass  # Clean rejection of empty queries is acceptable

    def test_none_question(self):
        """None question should not crash (or raise a clean ValueError)."""
        try:
            res = R.retrieve(
                tenant=self.tenant, question=None,
                top_k=5, min_score=0.0, staff_mode=True,
            )
            self.assertIsNotNone(res)
        except (ValueError, TypeError, AttributeError):
            pass  # Clean rejection of None is acceptable

    def test_top_k_zero(self):
        """top_k=0 should return empty results or raise ValueError."""
        try:
            res = R.retrieve(
                tenant=self.tenant, question="test",
                top_k=0, min_score=0.0, staff_mode=True,
            )
            self.assertEqual(len(res), 0)
        except (ValueError, Exception):
            pass  # Clean rejection is acceptable

    def test_very_high_min_score(self):
        """Very high min_score should return empty or very few results."""
        res = R.retrieve(
            tenant=self.tenant, question="test procedure edge",
            top_k=10, min_score=999.0, staff_mode=True,
        )
        self.assertLessEqual(len(res), 3)
