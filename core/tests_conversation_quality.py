"""Root-cause regressions from the three-asset conversation evaluation.

Each test covers a shared defect with wording that differs from the evaluation
scripts, so the fix is general rather than tied to fixed questions.
"""
from types import SimpleNamespace

from django.test import SimpleTestCase

from core.services.conversation_context import (
    ConversationContext, _track_customer_turn, canonical_issue_question, conversation_intent,
    extract_symptoms_from_text, is_problem_report, names_own_subject, ordinal_beyond_list, topic_shift_query)
from core.services.customer_answer import (
    _FRONT_MATTER_HEADING, _METADATA_LINE, asks_about_other_equipment)
from core.services.customer_conversation import _TECHNICIAN_DECISION
from core.services.term_matching import normalize_domain_terms


def _run(messages):
    ctx = ConversationContext(conversation_uuid="")
    trail = []
    for text in messages:
        before = ctx.active_issue
        _track_customer_turn(ctx, text)
        trail.append("obs" if ctx.answers_pending_question else ("issue" if ctx.active_issue != before else "-"))
    return ctx, trail


class ActiveIssueTrackingTests(SimpleTestCase):
    """A: questions never replace the current problem; statements add detail."""

    def test_help_requests_and_questions_do_not_become_the_problem(self):
        ctx, _ = _run(["Hello", "Could you help me with the unit?", "Why does that happen?",
                       "The unit won't turn on at all.", "What could be wrong with it?"])
        self.assertEqual(ctx.active_issue, "The unit won't turn on at all")
        self.assertFalse(is_problem_report("Why is it not cooling?"))
        self.assertTrue(is_problem_report("It is not cooling the room"))

    def test_details_are_observations_and_new_symptoms_are_new_issues(self):
        ctx, trail = _run(["The unit is making a grinding noise.", "Only in the morning.",
                           "No code on the display.", "The fan does turn.", "Also water is dripping from it."])
        self.assertEqual(trail, ["issue", "obs", "obs", "obs", "issue"])
        self.assertEqual(ctx.issue_history, ["The unit is making a grinding noise"])

    def test_corrections_with_a_preface_sentence(self):
        for text in ("Sorry, I described that wrongly. It runs but the air is not cool.",
                     "Let me correct myself: it runs but the room is not cooling."):
            with self.subTest(text=text):
                shifted = topic_shift_query(text) or ""
                self.assertNotIn("wrong", shifted.casefold())
                self.assertNotIn("correct", shifted.casefold())
        ctx, trail = _run(["My AC is not working.", "Wait, I explained it badly. It starts but isn't cooling."])
        self.assertEqual(trail[-1], "issue")
        self.assertIn("isn't cooling", ctx.active_issue)

    def test_codes_and_doubtful_readings_are_details(self):
        ctx, trail = _run(["The indoor units keep losing connection.", "It shows code F-12 now.",
                           "I might have misread that number."])
        self.assertEqual(trail, ["issue", "obs", "obs"])
        self.assertIn("losing connection", ctx.active_issue)

    def test_negated_symptoms_are_never_new_problems(self):
        ctx, trail = _run(["Rooms on the first floor are hotter than the others.", "No water is leaking."])
        self.assertEqual(trail, ["issue", "obs"])
        self.assertNotIn("water leak", extract_symptoms_from_text("No water is leaking."))


class IssueQuestionTests(SimpleTestCase):
    """C: subject-less cause/check requests use the current issue; named subjects don't."""

    def setUp(self):
        self.ctx = ConversationContext(conversation_uuid="", active_issue="The airflow is weak in the bedroom")

    def test_canonical_forms(self):
        self.assertTrue(canonical_issue_question("So what might be causing it?", self.ctx).startswith(
            "What are the possible causes of this:"))
        self.assertTrue(canonical_issue_question("Are there more possibilities?", self.ctx).startswith(
            "What other possible causes are there for this:"))
        self.assertTrue(canonical_issue_question("How can I tell which one it is?", self.ctx).startswith(
            "What can I safely check or observe for this:"))

    def test_questions_with_their_own_subject_keep_it(self):
        for text in ("What causes a low-pressure alarm on this model?", "What are the reasons for a tripped breaker?"):
            with self.subTest(text=text):
                self.assertTrue(names_own_subject(text))
                self.assertIsNone(canonical_issue_question(text, self.ctx))

    def test_ordinal_past_the_displayed_list(self):
        contract = {"presentation": {"citations": [{"id": "s1", "document_id": 1, "chunk_id": 1}],
                                     "response": {"displayed_claims": [{
                                         "kind": "possible_cause", "source_text": "Duct leakage.",
                                         "source_heading": "3. SYMPTOM: UNEVEN ROOM TEMPERATURES",
                                         "evidence_ids": ["s1"]}]}}}
        ctx = ConversationContext(conversation_uuid="", last_cause_list_contract=contract)
        self.assertEqual(len(ordinal_beyond_list("And the third one?", ctx)), 1)
        self.assertIsNone(ordinal_beyond_list("Tell me about the first one", ctx))


class IntentGapTests(SimpleTestCase):
    """B: conversation-only requests phrased differently from the scripts."""

    def test_intents(self):
        cases = {
            "ticket_help": ["Please open a support ticket for me.", "Could you log a service ticket?"],
            "ticket_contents": ["What details will go in the ticket?", "What exactly will the technician get?"],
            "certainty": ["What has actually been confirmed so far?", "Is this only a possibility?"],
            "assessment": ["What's the most likely problem?", "What do you think the real issue is?"],
            "relation": ["Is the leak connected to the issue I described before?"],
        }
        for intent, texts in cases.items():
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(conversation_intent(text), intent)
        self.assertTrue(_TECHNICIAN_DECISION.search("Honestly, do you think we should call an engineer?"))


class EvidenceSelectionTests(SimpleTestCase):
    """E/F/G: front matter never answers; number words match digits; other equipment."""

    def test_front_matter_and_metadata(self):
        for heading in ("1. PURPOSE", "General Overview", "2. Scope", "Introduction"):
            self.assertTrue(_FRONT_MATTER_HEADING.match(heading), heading)
        self.assertFalse(_FRONT_MATTER_HEADING.match("3. SYMPTOM: POOR COOLING"))
        self.assertTrue(_METADATA_LINE.match("Asset: VRF Outdoor Module"))
        self.assertTrue(_METADATA_LINE.match("Document ID: AC2T-TS-001"))
        self.assertFalse(_METADATA_LINE.match("Dirty filter or coil."))

    def test_number_words_match_schedule_digits(self):
        self.assertIn("6 months", normalize_domain_terms("what is due every six months"))
        self.assertIn("12 months", normalize_domain_terms("and every twelve months?"))

    def test_other_equipment_questions(self):
        ductable = SimpleNamespace(name="Ductable AC 5T", model_number="")
        self.assertTrue(asks_about_other_equipment("Does a chiller alarm code mean anything for this ductable?", ductable))
        self.assertFalse(asks_about_other_equipment("Does the ductable unit need a filter change?", ductable))


class FollowUpReferenceTests(SimpleTestCase):
    def _one_cause_contract(self):
        return {"presentation": {"response_type": "DIAGNOSTIC_GUIDANCE",
                                 "citations": [{"id": "s1", "document_id": 1, "chunk_id": 1}],
                                 "supporting_claims": [{"kind": "possible_cause", "source_text": "Duct leakage.",
                                                        "source_heading": "3. SYMPTOM: UNEVEN ROOM TEMPERATURES",
                                                        "evidence_ids": ["s1"]}],
                                 "response": {"displayed_claims": [{
                                     "kind": "possible_cause", "source_text": "Duct leakage.",
                                     "source_heading": "3. SYMPTOM: UNEVEN ROOM TEMPERATURES",
                                     "evidence_ids": ["s1"]}]}}}

    def test_simpler_after_a_conversation_only_reply_uses_the_single_supported_cause(self):
        from core.services.conversation_context import resolve_cause_reference
        ctx = ConversationContext(conversation_uuid="", is_followup=True,
                                  last_contract={"response_type": "CONVERSATION_REPLY"},
                                  last_supported_contract=self._one_cause_contract())
        ref = resolve_cause_reference("Could you put that in plain English?", ctx)
        self.assertEqual((ref["selected"]["source_text"], ref["mode"]), ("Duct leakage.", "simpler"))

    def test_other_causes_request_shows_only_unshown_causes(self):
        from core.services.customer_response_composer import compose_response
        def cause(cid, text):
            return {"id": cid, "kind": "possible_cause", "text": "Possible documented cause: " + text,
                    "source_text": text, "source_heading": "3. SYMPTOM: POOR COOLING", "evidence_ids": ["s" + cid]}
        claims = [cause("1", "Dirty filter."), cause("2", "Blocked outdoor unit.")]
        contract = {"presentation": {"response_type": "DIAGNOSTIC_GUIDANCE", "equipment": {"name": "AC"},
                                     "sections": [{"number": 1, "title": "x", "claims": claims}],
                                     "citations": [{"id": "s1"}, {"id": "s2"}], "supporting_claims": claims}}
        shown = {"presentation": {"supporting_claims": [claims[0]]}}
        ctx = SimpleNamespace(is_followup=True, last_contract=shown, last_cause_list_contract={},
                              original_complaint="", active_topics=[], resolved_answers=[])
        text = compose_response(contract, "What other possible causes are there for this: not cooling?", ctx)["text"]
        self.assertIn("Blocked outdoor unit", text)
        self.assertNotIn("Dirty filter", text)


class TermFoldingTests(SimpleTestCase):
    """R1: one shared matcher; verbs meet their nouns, calendar words meet each other."""

    def test_derivational_and_interval_keys(self):
        from core.services.term_matching import match_keys
        for asked, documented in (("vibrates", "vibration"), ("inspected", "inspection"), ("installs", "installation"),
                                  ("yearly", "annually"), ("quarter", "quarterly"), ("month", "monthly")):
            with self.subTest(asked=asked):
                self.assertTrue(match_keys([asked]) & match_keys([documented]))
        for a, b in (("vibrant", "vibration"), ("station", "state"), ("yearly", "monthly")):
            with self.subTest(a=a):
                self.assertFalse(match_keys([a]) & match_keys([b]))
        # A plural carries a count: "every 6 months" is not "Monthly", "two years" is not
        # "Annually", so the plural never carries the shared interval key.
        for plural, key in (("months", "@month"), ("years", "@year"), ("quarters", "@quarter"), ("weeks", "@week")):
            with self.subTest(plural=plural):
                self.assertNotIn(key, match_keys([plural]))
                self.assertIn(key, match_keys([plural[:-1]]))

    def test_schedule_intervals_compare_by_length(self):
        # R1b: "every six months" is not "Monthly"; "every 3 months" is "Quarterly".
        from core.services.customer_answer import stated_intervals
        self.assertEqual(stated_intervals("what is done every six months?"), stated_intervals("Every 6 months: x"))
        self.assertEqual(stated_intervals("what happens every 3 months"), stated_intervals("Quarterly: x"))
        self.assertEqual(stated_intervals("what happens once a year?"), stated_intervals("Annually: x"))
        self.assertFalse(stated_intervals("every six months") & stated_intervals("Monthly: inspect filters"))
        self.assertEqual(stated_intervals("how often should filters be checked?"), set())
        self.assertEqual(stated_intervals("is every room affected?"), set())

    def test_timing_questions(self):
        from core.services.customer_answer import _FREQUENCY_QUESTION
        for text in ("what is checked every quarter?", "what happens once a year?", "how often is it serviced?"):
            self.assertTrue(_FREQUENCY_QUESTION.search(text), text)
        self.assertFalse(_FREQUENCY_QUESTION.search("is every room affected?"))


class EscalationGuidanceTests(SimpleTestCase):
    def test_prohibitions_are_not_unsafe_instructions(self):
        from core.services.customer_answer import _NEGATED_DIRECTIVE, _UNSAFE_DIRECTIVE
        safe = "Escalate repeated faults or a burnt smell. Do not repeatedly reset a fault without diagnosing its cause."
        self.assertFalse(_UNSAFE_DIRECTIVE.search(_NEGATED_DIRECTIVE.sub(" ", safe)))
        unsafe = "Escalate if needed, then reset the controller and open the panel."
        self.assertTrue(_UNSAFE_DIRECTIVE.search(_NEGATED_DIRECTIVE.sub(" ", unsafe)))

    def test_escalation_needs_an_escalation_question(self):
        # R3a: the customer's own words must ask when to stop or escalate; sharing a
        # word with an escalation sentence ("breaker", "tripping") is not enough.
        from core.services.customer_answer import _ESCALATION_QUESTION
        for text in ("what are the warning signs that i should stop using it", "when should a problem be escalated?",
                     "is it safe to keep using it?", "should i keep resetting the breaker?",
                     "when should I stop using the AC?", "is this dangerous?"):
            with self.subTest(text=text):
                self.assertTrue(_ESCALATION_QUESTION.search(text))
        for text in ("What about the second one?", "Breaker or protective device tripped",
                     "Can you explain which of those are mentioned in the manual?", "why does the breaker trip?",
                     "the fan stopped", "how often should filters be checked?"):
            with self.subTest(text=text):
                self.assertFalse(_ESCALATION_QUESTION.search(text))


class HoldoutRootCauseTests(SimpleTestCase):
    """Root causes found by the sealed holdout, tested with different wording."""

    def test_chat_shorthand_reaches_the_intent_and_question_rules(self):
        from core.services.conversation_context import _is_question, normalize_chat_text
        self.assertEqual(conversation_intent("have u raised a ticket yet?"), "ticket_status")
        self.assertEqual(conversation_intent("what does the engineer receive"), "ticket_contents")
        self.assertEqual(conversation_intent("ok so whats the real problem"), "assessment")
        self.assertTrue(_is_question("well whats wrong with it"))
        # Statements stay statements; refrigerant names and part names are untouched.
        self.assertFalse(_is_question("cant get it to start"))
        self.assertFalse(is_problem_report("whats wrong with it"))
        self.assertEqual(normalize_chat_text("R-32 in the U-bend"), "R-32 in the U-bend")

    def test_certainty_visit_and_relation_intents(self):
        self.assertEqual(conversation_intent("has anything been confirmed yet?"), "certainty")
        self.assertEqual(conversation_intent("are all of these confirmed causes?"), "certainty")
        self.assertIsNone(conversation_intent("is the drain confirmed clear"))
        for text in ("could someone come and check it?", "send a technician please", "i need someone to look at it"):
            with self.subTest(text=text):
                self.assertEqual(conversation_intent(text), "ticket_help")
        self.assertIsNone(conversation_intent("I have someone coming tomorrow"))
        self.assertTrue(_TECHNICIAN_DECISION.search("Do you think I should get a technician?"))
        self.assertEqual(conversation_intent("is the rattling connected to the leaking issue?"), "relation")
        self.assertIsNone(conversation_intent("Is that the same as a communication fault?"))

    def test_subjectless_cause_and_note_requests_use_the_current_issue(self):
        ctx, _ = _run(["The indoor units keep losing connection."])
        for text in ("what's going on with it?", "what could be going on here"):
            with self.subTest(text=text):
                self.assertTrue((canonical_issue_question(text, ctx) or "").startswith("What are the possible causes"))
        self.assertTrue((canonical_issue_question("what should I write down for the installer?", ctx) or "")
                        .startswith("What can I safely check"))
        self.assertFalse(names_own_subject("what should I record for the technician"))
        self.assertTrue(names_own_subject("what are the causes of a fan fault"))

    def test_a_correction_without_a_new_problem_refines_the_issue(self):
        ctx, trail = _run(["The indoor units keep losing connection.",
                           "Correction: it's just the one unit on the second floor."])
        self.assertEqual(trail, ["issue", "obs"])
        self.assertIn("losing connection", ctx.active_issue)
        ctx, trail = _run(["The indoor units keep losing connection.",
                           "Correction: water is leaking from it, it's not losing connection."])
        self.assertEqual(trail[-1], "issue")

    def test_a_withdrawn_problem_restores_the_earlier_one(self):
        ctx, _ = _run(["The airflow in two rooms is weak.", "Also there's water dripping near it.",
                       "Sorry, the water is coming from the roof, not from the unit."])
        self.assertEqual(ctx.active_issue, "The airflow in two rooms is weak")
        self.assertEqual(ctx.retracted_issues, ["there's water dripping near it"])
        self.assertTrue(ctx.just_retracted)
        ctx, _ = _run(["The unit is making a rattling noise.", "The noise is from the outdoor unit, not the indoor unit."])
        self.assertEqual(ctx.retracted_issues, [])

    def test_conversation_requests_are_not_details_and_set_aside_words_are_dropped(self):
        ctx, _ = _run(["The unit is making a grinding noise.", "sum up what i've said", "pls make a ticket"])
        self.assertEqual(ctx.resolved_answers, [])
        ctx, _ = _run(["It is not cooling.", "ok forget that, now there's a rattling noise"])
        self.assertEqual(ctx.active_issue, "there's a rattling noise")
        self.assertEqual(ctx.issue_history, ["It is not cooling"])

    def test_symptom_wording_found_missing(self):
        for text, label in (("the fan runs but the office stays hot", "poor cooling"),
                            ("half the indoor units went offline", "communication fault"),
                            ("the air coming out of the ceiling vents upstairs is weak", "weak airflow")):
            with self.subTest(text=text):
                self.assertIn(label, extract_symptoms_from_text(text))
        for text in ("the living room is fine", "I disconnected the power", "water drops off the coil"):
            with self.subTest(text=text):
                self.assertNotIn("communication fault", extract_symptoms_from_text(text))
                self.assertNotIn("poor cooling", extract_symptoms_from_text(text))

    def test_pressure_faults_are_named_concepts(self):
        # R10: "high pressure" names the HIGH-PRESSURE section; "pressure" alone names no fault.
        from core.services.term_matching import heading_states_symptom, symptom_concepts
        for text in ("could high pressure be involved?", "the module shows a high-pressure alarm",
                     "is the head pressure too high?"):
            with self.subTest(text=text):
                self.assertIn("high_pressure", symptom_concepts(text))
        self.assertIn("low_pressure", symptom_concepts("we keep getting a low pressure trip"))
        for text in ("what pressure should the system be tested at?", "what refrigerant pressure should it run at?"):
            with self.subTest(text=text):
                self.assertFalse({"high_pressure", "low_pressure"} & symptom_concepts(text))
        self.assertTrue(heading_states_symptom("3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT", {"high_pressure"}))
        self.assertFalse(heading_states_symptom("4. LOW-PRESSURE OR CAPACITY FAULT", {"high_pressure"}))


class EvaluationRoundTwelveTests(SimpleTestCase):
    """R12: classifier gaps found by the manual-coverage set, with different wording."""

    def test_value_questions_phrased_with_should(self):
        from core.services.customer_answer import _VALUE_QUESTION
        for text in ("what voltage should the supply be?", "what pressure must it be charged to?",
                     "what airflow should this unit deliver?"):
            with self.subTest(text=text):
                self.assertTrue(_VALUE_QUESTION.search(text))
        self.assertFalse(_VALUE_QUESTION.search("what airflow problem should I expect?"))

    def test_role_questions_accept_one_shared_term_only_for_performed_by_statements(self):
        from core.services.customer_answer import _ROLE_QUESTION, _required_overlap
        self.assertTrue(_ROLE_QUESTION.search("who can service this unit?"))
        self.assertFalse(_ROLE_QUESTION.search("who makes this unit?"))
        self.assertEqual(_required_overlap(True, "Servicing must be carried out by qualified technicians."), 1)
        self.assertEqual(_required_overlap(True, "Check the filters monthly."), 2)
        self.assertEqual(_required_overlap(False, "Servicing must be carried out by qualified technicians."), 2)

    def test_does_not_start_is_its_own_symptom(self):
        from core.services.term_matching import heading_states_symptom, symptom_concepts
        for text in ("it won't switch on", "the unit does not start", "there's no power at all"):
            with self.subTest(text=text):
                self.assertIn("no_start", symptom_concepts(text))
        self.assertNotIn("no_start", symptom_concepts("The AC does start, but it isn't cooling properly."))
        self.assertTrue(heading_states_symptom("2. SYMPTOM: UNIT DOES NOT START", {"no_start"}))
        self.assertFalse(heading_states_symptom("3. SYMPTOM: POOR COOLING", {"no_start"}))

    def test_a_negated_heading_needs_a_negated_question(self):
        # R13: "UNIT DOES NOT START" is not stated by "the AC does start" or "when I start it".
        from core.services.term_matching import question_states_heading
        heading = "2. SYMPTOM: UNIT DOES NOT START"
        for text in ("The AC does start, but it isn't cooling properly.", "It happens only when I start it.",
                     "issue: the unit starts fine but is noisy"):
            with self.subTest(text=text):
                self.assertFalse(question_states_heading(text, heading))
        for text in ("it isn't starting", "my unit won't start", "the AC does not start at all"):
            with self.subTest(text=text):
                self.assertTrue(question_states_heading(text, heading))
        self.assertTrue(question_states_heading("the module shows a high pressure alarm",
                                                "3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT"))


class LivePathRootCauseTests(SimpleTestCase):
    """R14: defects found by replaying the live customer path, tested with other wording."""

    def test_a_question_is_never_a_completed_check(self):
        from core.services.conversation_context import _confirmed_checks
        self.assertEqual(_confirmed_checks("what is inspected during the annual service?", []), [])
        self.assertEqual(_confirmed_checks("Have the filters been cleaned?", []), [])
        self.assertTrue(_confirmed_checks("I already cleaned the filter", []))
        # A statement beside a question still reports the check.
        self.assertTrue(_confirmed_checks("I checked that already. What next?", ["Inspect the air filter"]))

    def test_a_self_contained_question_gets_no_filler_context(self):
        from core.services.conversation_context import contextualize_customer_query
        ctx = ConversationContext(conversation_uuid="", asset_name="VRF Outdoor Module")
        ctx.is_followup, ctx.turn_count = True, 1
        query = contextualize_customer_query("when should a fault be escalated?", ctx)
        self.assertNotIn("troubleshooting", query)
        self.assertNotIn("symptoms:", query)

    def test_fault_qualifiers_are_counted_by_family(self):
        from core.services.llm import evaluate_chunk_claim_support
        comm = SimpleNamespace(text="Possible causes: communication cable damage or incorrect topology.",
                               heading="2. COMMUNICATION OR ADDRESSING FAULT")
        ok, reason = evaluate_chunk_claim_support(
            comm, {"network", "communication", "cable"}, set(), set(),
            q_text="units dropping off the network; symptoms: communication fault cable")
        self.assertNotEqual(reason, "FAULT_QUALIFIER_MISMATCH")
        _ok, reason = evaluate_chunk_claim_support(
            comm, {"inverter", "communication", "cable"}, set(), set(),
            q_text="inverter drive communication fault cable")
        self.assertEqual(reason, "FAULT_QUALIFIER_MISMATCH")

    def test_self_check_questions_use_the_current_issue(self):
        # R15: "what can I safely check myself?" is a check request about the current issue.
        ctx, _ = _run(["The unit won't switch on."])
        for text in ("what can I safely check myself?", "is there anything I could check?", "can I check it myself?"):
            with self.subTest(text=text):
                self.assertTrue((canonical_issue_question(text, ctx) or "").startswith("What can I safely check"))
        self.assertIsNone(canonical_issue_question("can you check the ticket status?", ctx))
