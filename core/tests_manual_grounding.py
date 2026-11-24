"""R2-R4 through the real customer gateway on synthetic manuals (not the approved corpus)."""
from core.models import KnowledgeDocument
from core.services.indexing import index_document
from core.tests_conversation_state import install_split_ac_fixture
from core.tests_conversational_gateway import VRFConversationGatewayTests
from core.tests_demo_gateway import GatewayTestBase

TROUBLE = """1. SYMPTOM: UNIT DOES NOT START
Possible causes:
- Power supply unavailable or isolator switched off.
- Breaker or protective device tripped.
Initial checks:
- Do not repeatedly reset a tripping breaker.
- Electrical panel checks must be performed by a qualified person.

4. SYMPTOM: WATER LEAKAGE
Possible causes:
- Blocked drain or dirty drain pan.
Record the leak location and take photographs.

2. COMMUNICATION OR ADDRESSING FAULT
Possible causes:
- Communication cable damage or incorrect topology.

3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT
Possible causes:
- Blocked airflow, dirty coil, or fan issue.

5. COMPRESSOR OR INVERTER FAULT
Possible causes:
- Electrical supply issue, over-temperature, drive fault, or compressor protection.
"""
MAINTENANCE = """1. PURPOSE
Use this checklist to plan routine inspection and maintenance. Frequency should be based on manufacturer guidance and service history.

2. SUGGESTED MAINTENANCE INTERVALS
- Quarterly: inspect coils, drain pan, insulation, and supports.
- Annually: arrange a comprehensive service by a qualified technician.
- Monthly: inspect filters and the drain outlet.
- Every 6 months: document airflow, temperature split and control operation.

3. CHECKLIST
[ ] Isolate power before opening service panels.
[ ] Inspect evaporator and condenser coils.
[ ] Check terminals, cables, earthing, and protective devices.

5. INSTALLATION
Installation must be performed by qualified HVAC technicians.

4. WARNING SIGNS
Escalate if there is repeated breaker tripping, burning smell, abnormal vibration, or water leakage. Stop operation when continued use may be unsafe.
"""


class ManualGroundingTests(GatewayTestBase):
    def setUp(self):
        super().setUp()
        install_split_ac_fixture(self, TROUBLE)
        doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.asset.product, asset=self.asset, title="Maintenance Checklist",
            source_type="text", is_rag_enabled=True, content_text=MAINTENANCE)
        index_document(doc)

    create = VRFConversationGatewayTests.create
    ask = VRFConversationGatewayTests.ask

    def headings(self, contract):
        return {c.get("heading", "") for c in (contract.get("presentation") or {}).get("citations") or []}

    def test_r2_section_named_by_the_component_is_used(self):
        for text in ("the compressor keeps tripping", "the inverter trips every few minutes"):
            with self.subTest(text=text):
                answer, _ = self.ask(self.create(), text)
                self.assertIn("5. COMPRESSOR OR INVERTER FAULT", self.headings(answer), answer.get("answer"))
        other, _ = self.ask(self.create(), "the indoor units lose connection")
        self.assertNotIn("5. COMPRESSOR OR INVERTER FAULT", self.headings(other))

    def test_r1_timing_questions_answered_from_the_schedule(self):
        # Tests have no dense index; the sparse-findable wording exercises the same
        # timing-intent and interval-matching rules (live coverage set uses the original).
        for text, word in (("which items are inspected quarterly?", "coils"),
                           ("what does the yearly service involve?", "comprehensive")):
            with self.subTest(text=text):
                answer, _ = self.ask(self.create(), text)
                self.assertIn(word, answer.get("answer", "").casefold())

    def test_r3_warning_signs_are_shared(self):
        for text in ("what are the warning signs that i should stop using it", "when should I escalate this?"):
            with self.subTest(text=text):
                answer, _ = self.ask(self.create(), text)
                self.assertIn("burning smell", answer.get("answer", "").casefold())

    def test_r4_technician_only_steps_are_not_called_missing(self):
        answer, _ = self.ask(self.create(), "which terminals and cables does the technician check?")
        text = answer.get("answer", "").casefold()
        self.assertNotIn("isolate power", text)
        self.assertNotIn("earthing", text.split("documentation")[0] if "documentation" in text else "")

    def test_r4_withheld_checklist_lines_are_described_as_technician_work(self):
        answer, _ = self.ask(self.create(), "what is inspected on the checklist?")
        text = answer.get("answer", "").casefold()
        self.assertNotIn("isolate power", text)
        if answer.get("response_type") == "INSUFFICIENT_EVIDENCE":
            self.assertIn("qualified technician", text)

    def test_r3a_an_ordinal_follow_up_is_not_answered_with_warning_signs(self):
        conv = self.create()
        self.ask(conv, "the unit does not start")
        answer, _ = self.ask(conv, "what about the second one?")
        text = answer.get("answer", "").casefold()
        self.assertIn("breaker", text)
        self.assertNotIn("burning smell", text)

    def test_r5a_a_generic_failure_does_not_pull_in_a_leak_only_note(self):
        conv = self.create()
        self.ask(conv, "my unit isn't working properly")
        answer, _ = self.ask(conv, "how do I know which one is affecting it?")
        self.assertNotIn("leak location", answer.get("answer", "").casefold())
        leak, _ = self.ask(self.create(), "water is leaking from the unit, what should I note down?")
        self.assertIn("leak location", leak.get("answer", "").casefold())

    def test_r1a_every_six_months_is_not_monthly(self):
        answer, _ = self.ask(self.create(), "what maintenance is recommended every six months?")
        text = answer.get("answer", "").casefold()
        self.assertIn("temperature split", text)
        self.assertNotIn("inspect filters", text)

    def test_r10_a_named_hyphenated_section_is_used_and_one_shared_word_is_not_enough(self):
        answer, _ = self.ask(self.create(), "is high pressure a possibility here?")
        self.assertIn("blocked airflow", answer.get("answer", "").casefold())
        self.assertNotEqual(answer.get("reason_code"), "TECHNICIAN_ONLY_EVIDENCE")
        # "pressure" alone does not name the HIGH-PRESSURE section.
        other, _ = self.ask(self.create(), "what pressure should the system be tested at?")
        self.assertNotIn("blocked airflow", other.get("answer", "").casefold())

    def test_front_matter_answers_document_and_timing_questions_only(self):
        purpose, _ = self.ask(self.create(), "What does the maintenance checklist provide?")
        self.assertIn("plan routine inspection", purpose.get("answer", "").casefold())
        timing, _ = self.ask(self.create(), "What should the maintenance frequency be based on?")
        self.assertIn("manufacturer guidance", timing.get("answer", "").casefold())
        conv = self.create()
        for text in ("the compressor keeps tripping", "what could be causing this?", "what should I do next?"):
            with self.subTest(text=text):
                answer, _ = self.ask(conv, text)
                self.assertNotIn("plan routine inspection", answer.get("answer", "").casefold())

    def test_r12_escalation_role_start_and_schedule_questions(self):
        # "problem" reads as a cause word and "when should" as timing; neither may hide
        # the documented escalation conditions. (Sparse-only test retrieval needs "escalate".)
        escalate, _ = self.ask(self.create(), "when should I escalate a problem?")
        self.assertIn("burning smell", escalate.get("answer", "").casefold())
        timing, _ = self.ask(self.create(), "how often should the coils be inspected?")
        self.assertIn("quarterly", timing.get("answer", "").casefold())
        role, _ = self.ask(self.create(), "who is permitted to install it?")
        self.assertIn("qualified", role.get("answer", "").casefold())
        schedule, _ = self.ask(self.create(), "how often does it need a service?")
        text = schedule.get("answer", "").casefold()
        self.assertIn("quarterly", text)
        self.assertIn("annually", text)
        conv = self.create()
        start, _ = self.ask(conv, "the unit won't switch on")
        self.assertIn("breaker", start.get("answer", "").casefold())
        check, _ = self.ask(conv, "what can I safely check?")
        self.assertNotIn("drain", check.get("answer", "").casefold())

    def test_r14_timing_question_and_generic_overview_after_earlier_answers(self):
        conv = self.create()
        leak, _ = self.ask(conv, "water is leaking from the unit")
        self.assertIn("blocked drain", leak.get("answer", "").casefold())
        timing, _ = self.ask(conv, "how often does it need a service?")
        text = timing.get("answer", "").casefold()
        self.assertIn("quarterly", text)
        self.assertNotIn("blocked drain", text)
        # Topic labels of earlier answers must not narrow a generic overview.
        conv = self.create()
        self.ask(conv, "my unit isn't working properly")
        overview, _ = self.ask(conv, "what could be causing this?")
        text = overview.get("answer", "").casefold()
        self.assertIn("breaker", text)
        self.assertIn("drain", text)

    def test_live_vague_complaint_explanation_next_step_and_call_request(self):
        """The reported live conversation: vague complaint after a greeting, an ordinal
        explanation, a next-step question and an informal service-call request."""
        from core.models import CustomerConversation
        conv = self.create()
        self.ask(conv, "hi")
        vague, _ = self.ask(conv, "its not working")
        self.assertEqual(vague.get("response_type"), "CLARIFICATION", vague.get("answer"))
        self.assertIn("unit does not start", vague.get("answer", "").casefold())
        # Sparse-only test retrieval needs the heading's words; the live check uses dense.
        start, _ = self.ask(conv, "the unit does not start")
        self.assertIn("breaker", start.get("answer", "").casefold())
        explain, _ = self.ask(conv, "can you explain the second causes")
        text = explain.get("answer", "").casefold()
        self.assertIn("don't have enough information to explain it further", text)
        self.assertNotIn("manual", text)
        self.assertIn("do not repeatedly reset", text)
        nxt, _ = self.ask(conv, "what to do next?")
        text = nxt.get("answer", "").casefold()
        self.assertIn("qualified person", text)
        self.assertNotIn("blocked drain", text)
        call, _ = self.ask(conv, "can u create a call for me")
        self.assertEqual(call.get("reason_code"), "TICKET_HELP", call.get("answer"))
        self.assertFalse((call.get("presentation") or {}).get("citations"))
        self.assertIsNone(CustomerConversation.objects.get(conversation_uuid=conv).linked_service_call_id)

    def test_live_conversation_variants(self):
        conv = self.create()
        self.ask(conv, "hello")
        vague, _ = self.ask(conv, "my unit is acting up")
        self.assertEqual(vague.get("response_type"), "CLARIFICATION")
        leak, _ = self.ask(conv, "water is leaking from it")
        self.assertIn("blocked drain", leak.get("answer", "").casefold())
        nxt, _ = self.ask(conv, "ok what now")
        self.assertNotIn("breaker", nxt.get("answer", "").casefold())
        ticket, _ = self.ask(conv, "pls raise a complaint")
        self.assertEqual(ticket.get("reason_code"), "TICKET_HELP")
        # A specific first complaint is answered directly, never clarified.
        direct, _ = self.ask(self.create(), "water is dripping from the unit")
        self.assertNotEqual(direct.get("response_type"), "CLARIFICATION")
