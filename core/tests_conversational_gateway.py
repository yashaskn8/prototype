"""Multi-turn conversational RAG regressions through the real Customer demo gateway.

Every turn goes through ``/demo-api/.../messages-stream/`` with the headers the
React client sends (Demo authorization + JSON content type, no Accept override),
so authentication, SSE framing, context reconstruction, retrieval, claim
validation and composition are exercised together.
"""
import json
import os
from unittest import mock

from django.test import override_settings

from core.models import (
    Asset, Brand, CustomerMessage, KnowledgeDocument, Product, ProductCategory, ProductDomain, Site)
from core.services.indexing import index_document
from core.tests_demo_gateway import CUSTOMER_CODE, GatewayTestBase

VRF_MANUAL = """5. SYMPTOM: OUTDOOR MODULE NOT WORKING
Possible causes:
- Power supply to the outdoor module is interrupted.
- Communication fault between the indoor and outdoor units.
- High-pressure protection has tripped.

6. POWER SUPPLY INTERRUPTION
When the power supply to the outdoor module is interrupted, the compressor and outdoor fan cannot start.
A tripped breaker or a loose terminal connection interrupts the power supply to the outdoor module.

7. COMMUNICATION FAULT
A communication fault occurs when the signal wiring between the indoor and outdoor units is damaged or loose.
During a communication fault the outdoor module stops and the controller displays a communication alarm.

8. HIGH-PRESSURE PROTECTION
High-pressure protection trips when condenser airflow is blocked or the outdoor coil is heavily soiled.
After high-pressure protection trips, the outdoor module stays off until the protection is reset by a technician.
"""

QUESTION = "What are the possible causes of a VRF outdoor module not working?"


def parse_sse(response):
    body = b"".join(response.streaming_content).decode("utf-8")
    events = []
    for block in body.split("\n\n"):
        name, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name:
            events.append((name, json.loads("\n".join(data)) if data else None))
    return events


class _InlineThread:
    """Run the stream worker in the test thread so it sees the test transaction.

    Only thread placement and per-thread connection cleanup are replaced; the
    real view, gateway authentication, worker body and SSE generator still run.
    """

    def __init__(self, target, name=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


class VRFConversationGatewayTests(GatewayTestBase):
    def setUp(self):
        super().setUp()
        from core.services import rag_event_stream
        for name, value in (("Thread", _InlineThread), ("close_old_connections", lambda: None),
                            ("connections", mock.Mock())):
            patcher = mock.patch.object(rag_event_stream, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        domain = ProductDomain.objects.create(tenant=self.tenant, name="VRF")
        category = ProductCategory.objects.create(tenant=self.tenant, domain=domain, name="VRF")
        brand = Brand.objects.create(tenant=self.tenant, name="V")
        product = Product.objects.create(tenant=self.tenant, category=category, brand=brand,
                                         name="VRF Outdoor Module")
        site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Roof")
        self.vrf = Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site,
                                        product=product, name="VRF Outdoor Module", asset_code="DEV-VRF-1")
        doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=product, asset=self.vrf, title="VRF Outdoor Module - Service Manual",
            source_type="text", is_rag_enabled=True, content_text=VRF_MANUAL)
        index_document(doc)
        self.secret_doc = KnowledgeDocument.objects.create(
            tenant=self.other_tenant, title="Other tenant VRF notes", source_type="text", is_rag_enabled=True,
            content_text="7. COMMUNICATION FAULT\nOTHER-TENANT-SECRET wiring procedure for communication faults.")

    # -- helpers ---------------------------------------------------------
    def create(self):
        r = self.as_customer("post", "/demo-api/customer-support/conversations/",
                             data={"asset_id": self.vrf.id}, content_type="application/json")
        self.assertIn(r.status_code, (200, 201), r.content[:300])
        return r.json()["conversation_uuid"]

    def ask(self, uuid, message):
        """POST exactly as RealDemoPage.callRagStream does (no Accept header)."""
        with override_settings(ROOT_URLCONF="servy_rag.demo_customer_urls"):
            r = self.client.post(
                f"/demo-api/customer-support/conversations/{uuid}/messages-stream/",
                data=json.dumps({"message": message}), content_type="application/json",
                HTTP_HOST="127.0.0.1", HTTP_AUTHORIZATION=f"Demo {CUSTOMER_CODE}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r["Content-Type"].startswith("text/event-stream"))
        events = parse_sse(r)
        results = [payload for name, payload in events if name == "result"]
        self.assertEqual(len(results), 1, [name for name, _ in events])
        payload = results[0]
        contract = (payload.get("assistant_message") or {}).get("structured_contract") or payload.get("contract")
        logs = [payload for name, payload in events if name == "log"]
        if os.environ.get("SERVY_TRACE_TURNS"):
            print("\n>>>", message[:80], "|", contract.get("response_type"), "\n", contract.get("answer"))
        return contract, logs

    @staticmethod
    def segments(contract):
        return ((contract.get("presentation") or {}).get("response") or {}).get("segments") or []

    @staticmethod
    def cited_texts(contract):
        return [s["text"] for s in VRFConversationGatewayTests.segments(contract) if s.get("citation_ids")]

    def assert_citations_resolve(self, contract):
        citations = {c.get("id") for c in (contract.get("presentation") or {}).get("citations") or []}
        for seg in self.segments(contract):
            for cid in seg.get("citation_ids") or []:
                self.assertIn(cid, citations, f"segment cites unknown {cid}: {seg['text']}")
        self.assertNotIn("OTHER-TENANT-SECRET", json.dumps(contract))

    # -- the five-turn sequence -----------------------------------------
    def test_five_turn_vrf_sequence(self):
        uuid = self.create()

        first, _ = self.ask(uuid, QUESTION)
        causes = [s["text"] for s in self.segments(first) if s["kind"] == "list_item"]
        self.assertGreaterEqual(len(causes), 3, self.segments(first))
        self.assert_citations_resolve(first)

        for follow_up in ("Explain each issue.", "Explain each issue mentioned above."):
            with self.subTest(follow_up=follow_up):
                answer, logs = self.ask(uuid, follow_up)
                self.assert_explains_every_cause(answer, causes)

        pasted = "\n".join(["For outdoor module not working on the VRF Outdoor Module, the manual lists these possible causes:"]
                           + [f"- {c}" for c in causes]) + "\n\nPlease explain these issues."
        answer, _ = self.ask(uuid, pasted)
        self.assertNotEqual((answer.get("presentation") or {}).get("response_type"), "INSUFFICIENT_EVIDENCE",
                            answer.get("answer"))
        self.assert_explains_every_cause(answer, causes)

        greeting, logs = self.ask(uuid, "Hi.")
        self.assertEqual(greeting.get("response_type"), "GREETING", greeting.get("answer"))
        self.assertFalse(self.cited_texts(greeting))
        rag = [l for l in logs if l.get("stage") == "customer_rag_execution"]
        self.assertTrue(all(l.get("status") == "skipped" for l in rag), rag)

    # -- context preservation and negative cases ------------------------
    def listed(self, contract):
        return [s["text"] for s in self.segments(contract) if s["kind"] == "list_item"]

    def test_first_turn_greeting_does_not_retrieve_or_become_the_complaint(self):
        uuid = self.create()
        greeting, logs = self.ask(uuid, "Hello")
        self.assertEqual(greeting["response_type"], "GREETING")
        self.assertEqual(greeting["diagnostics"]["retrieval"]["passes"], 0)
        self.assertFalse([l for l in logs if l.get("stage") == "retrieval_pipeline"])
        first, _ = self.ask(uuid, QUESTION)
        self.assertGreaterEqual(len(self.listed(first)), 3)
        self.assertGreaterEqual(first["diagnostics"]["retrieval"]["passes"], 1)

    def test_greeting_and_refusal_do_not_erase_the_cited_cause_list(self):
        uuid = self.create()
        first, _ = self.ask(uuid, QUESTION)
        causes = self.listed(first)
        self.ask(uuid, "Hi.")
        refused, _ = self.ask(uuid, "Does this definitely prove the compressor is broken?")
        self.assertNotEqual(refused.get("response_type"), "DIAGNOSTIC_GUIDANCE")
        second, _ = self.ask(uuid, "Explain the second cause.")
        self.assertEqual(self.listed(second) or self.cited_texts(second)[:1], [causes[1]])
        every, _ = self.ask(uuid, "Explain each issue.")
        self.assert_explains_every_cause(every, causes)

    def test_pasted_unapproved_cause_is_never_explained_as_evidence(self):
        uuid = self.create()
        first, _ = self.ask(uuid, QUESTION)
        causes = self.listed(first)
        pasted = "- " + causes[0] + "\n- Gremlins have moved into the compressor housing.\nExplain these issues."
        answer, _ = self.ask(uuid, pasted)
        self.assertNotIn("Gremlins", json.dumps(answer.get("presentation")))
        items = answer["presentation"]["response"]["explanations"]
        self.assertEqual([i["cause"] for i in items], [causes[0]])

    def test_cause_without_explaining_source_is_marked_as_an_evidence_gap(self):
        KnowledgeDocument.objects.filter(asset=self.vrf).delete()
        doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, asset=self.vrf, title="VRF Outdoor Module - Short Manual", source_type="text",
            is_rag_enabled=True, content_text=VRF_MANUAL.split("8. HIGH-PRESSURE")[0])
        index_document(doc)
        uuid = self.create()
        first, _ = self.ask(uuid, QUESTION)
        causes = self.listed(first)
        answer, _ = self.ask(uuid, "Explain each issue.")
        items = {i["cause"]: i for i in answer["presentation"]["response"]["explanations"]}
        self.assertEqual(list(items), causes)
        high_pressure = next(c for c in causes if "High-pressure" in c)
        self.assertEqual(items[high_pressure]["status"], "insufficient_evidence")
        self.assertFalse(items[high_pressure]["evidence"])
        self.assertIn("cause_without_explaining_evidence", answer["diagnostics"]["rejection_reasons"])
        texts = [s["text"] for s in self.segments(answer)]
        self.assertIn("The approved manuals do not explain this cause further.", texts)

    def test_explain_each_without_a_previous_cause_list_is_not_invented(self):
        uuid = self.create()
        answer, _ = self.ask(uuid, "Explain each issue.")
        self.assertNotEqual(((answer.get("presentation") or {}).get("response") or {}).get("style"),
                            "cause_explanations")

    def assert_explains_every_cause(self, contract, causes):
        self.assert_citations_resolve(contract)
        response = (contract.get("presentation") or {}).get("response") or {}
        self.assertEqual(response.get("style"), "cause_explanations", contract.get("answer"))
        items = response.get("explanations") or []
        self.assertEqual([i["cause"] for i in items], causes)
        for item in items:
            # Either a cited explanation that is not just the cause restated, or an explicit gap.
            if item["status"] == "explained":
                self.assertTrue(item["evidence"], item)
                for ev in item["evidence"]:
                    self.assertTrue(ev["citation_ids"])
                    self.assertNotEqual(ev["text"].casefold(), item["cause"].casefold())
            else:
                self.assertEqual(item["status"], "insufficient_evidence")
        self.assertTrue(any(i["status"] == "explained" for i in items))
