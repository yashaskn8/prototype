"""Answer-stage regression tests over the nine real source documents.

Retrieval is replaced by an oracle: every numbered section of the selected
asset's three documents is offered as an authorized candidate, exactly as the
exhaustive approved-document scan would. The real evidence-sufficiency check,
sentence classification, safety classification, citation signing and dialogue
focusing then run unchanged. These tests therefore verify the answer stage
only; they say nothing about retrieval ranking or whether a document is indexed.

No database, network or model is used. The source files live under the
git-ignored ``media/`` folder, so the tests skip when they are absent.
"""
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rest_framework.exceptions import NotFound

from core.services import customer_answer as ca
from core.services.term_matching import (
    heading_states_symptom, match_keys, question_states_heading, stem, symptom_concepts,
)
from tools.servy_real_demo_dialogue import focus_answer

DOCS = Path(__file__).resolve().parent.parent / "media" / "knowledge" / "9" / "2026" / "10"
ASSETS = {
    "split": ("Split AC Unit 2T", 59, ("9e5d", "c476", "2883")),
    "vrf": ("VRF Outdoor Module", 60, ("18f1", "4aa0", "b65b")),
    "duct": ("Ductable AC 5T", 61, ("75ab", "225b", "33fe")),
}
HAVE_DOCS = DOCS.is_dir() and len(list(DOCS.glob("*.txt"))) == 9


def _chunks(prefixes, first_document_id):
    """One chunk per numbered section, as the indexed documents are split."""
    chunks = {}
    for offset, prefix in enumerate(prefixes):
        path = next(DOCS.glob(prefix + "*"))
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^Document: (.+)$", text, re.M).group(1).strip()
        document = SimpleNamespace(id=first_document_id + offset, title=title, index_version=1)
        for part in re.split(r"\n(?=\d+\. [A-Z])", text)[1:]:
            heading, _, body = part.partition("\n")
            chunk_id = len(chunks) + 1 + first_document_id * 100
            chunks[chunk_id] = SimpleNamespace(
                id=chunk_id, document_id=document.id, document=document,
                heading=heading.strip(), text=body.strip())
    return chunks


def ask(asset_key, question, *, last_topic=None, previous_facts=(), extra_chunks=None,
        denied=()):
    """Run the real answer stage for one customer question."""
    name, asset_id, prefixes = ASSETS[asset_key]
    chunks = _chunks(prefixes, asset_id * 10)
    chunks.update(extra_chunks or {})
    tenant = SimpleNamespace(id=9)
    asset = SimpleNamespace(id=asset_id, name=name, model_number="", asset_code="",
                            tenant_id=9, customer_id=14,
                            product=SimpleNamespace(name=name))
    customer = SimpleNamespace(id=14, tenant_id=9)

    def source_chunk(*, tenant, customer, asset, document_id, chunk_id):
        chunk = chunks.get(chunk_id)
        if chunk is None or chunk.document_id != document_id or chunk_id in denied:
            raise NotFound("Source unavailable.")
        return chunk

    interaction = SimpleNamespace(id=None, created_by_id=-1, tenant=tenant,
                                  customer=customer, asset=asset, question=question)
    references = [{"document_id": c.document_id, "chunk_id": c.id, "score": 0.9}
                  for c in chunks.values()]
    with patch.object(ca, "source_chunk", side_effect=source_chunk):
        contract = ca.build_customer_answer(interaction, None, references,
                                            original_question=question)
    for citation in contract["presentation"].get("citations", []):
        citation["url"] = citation["url"].replace(
            "/api/customer-support/source/", "/demo-api/source/", 1)
    focused, state = focus_answer(contract, question, last_topic=last_topic,
                                  previous_facts=previous_facts)
    presentation = focused["presentation"]
    citations = {c["id"]: c for c in presentation.get("citations", [])}
    claims = [c for c in presentation.get("supporting_claims", [])]
    return SimpleNamespace(
        type=focused["response_type"], message=presentation.get("message", ""),
        texts=[c["source_text"] for c in claims],
        kinds=[c["kind"] for c in claims],
        cited=[(citations[c["evidence_ids"][0]]["title"],
                citations[c["evidence_ids"][0]]["heading"]) for c in claims],
        claims=claims, chunks=chunks, raw=focused, state=state,
        dump=json.dumps(presentation),
    )


class MatchingRuleTests(unittest.TestCase):
    def test_inflections_fold_but_identifiers_and_numbers_do_not(self):
        self.assertEqual(stem("rooms"), "room")
        self.assertEqual(stem("cleaned"), "clean")
        self.assertEqual(stem("tripped"), "trip")
        self.assertEqual(stem("E-04"), "e-04")
        self.assertEqual(stem("AC-600s"), "ac-600s")
        self.assertFalse(match_keys(["e-04"]) & match_keys(["e-05"]))
        self.assertFalse(match_keys(["ac-60"]) & match_keys(["ac-600"]))

    def test_symptom_paraphrases_map_only_to_their_own_heading(self):
        cold = symptom_concepts("My AC is running but the room is not getting cold.")
        uneven = symptom_concepts("Some rooms are much colder than others.")
        comms = symptom_concepts("The indoor units have lost connection with the outdoor module.")
        self.assertEqual(cold, {"poor_cooling"})
        self.assertEqual(uneven, {"uneven_temperature"})
        self.assertEqual(comms, {"communication"})
        self.assertTrue(heading_states_symptom("3. SYMPTOM: POOR COOLING", cold))
        self.assertFalse(heading_states_symptom("4. SYMPTOM: WATER LEAKAGE", cold))
        self.assertFalse(heading_states_symptom("3. SYMPTOM: POOR COOLING", uneven))
        self.assertFalse(heading_states_symptom("2. SYMPTOM: WEAK AIRFLOW", comms))
        # Related-sounding wording that states no documented symptom.
        for text in ("Is the cold storage room on the same circuit?",
                     "How do I connect the remote to my phone?",
                     "What is the room temperature rating?"):
            self.assertEqual(symptom_concepts(text), frozenset(), text)


class HeadingRuleTests(unittest.TestCase):
    def test_question_must_name_the_whole_symptom_of_one_heading_alternative(self):
        uneven = "3. SYMPTOM: UNEVEN ROOM TEMPERATURES"
        self.assertFalse(question_states_heading(
            "Why is the temperature of the server room logged?", uneven))
        self.assertTrue(question_states_heading("Why are room temperatures uneven?", uneven))
        pressure = "3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT"
        self.assertTrue(question_states_heading("What causes a high pressure fault?", pressure))
        self.assertFalse(question_states_heading("What causes a low pressure fault?", pressure))
        self.assertFalse(question_states_heading("Why is the pressure gauge new?", pressure))
        self.assertTrue(question_states_heading(
            "Why is water leaking?", "4. SYMPTOM: WATER LEAKAGE"))
        self.assertFalse(question_states_heading("", uneven))


class CustomerSafeFactRuleTests(unittest.TestCase):
    def test_schedule_and_plain_statements_qualify(self):
        self.assertEqual(ca.customer_safe_fact(
            "Annually: perform a comprehensive service by a qualified HVAC technician.",
            "UNKNOWN_REQUIRES_REVIEW"), "maintenance_schedule")
        self.assertEqual(ca.customer_safe_fact(
            "Alarm codes differ by manufacturer and product generation.",
            "UNKNOWN_REQUIRES_REVIEW"), "descriptive")

    def test_instructions_checklist_items_and_restricted_levels_fail_closed(self):
        for text, level in (
            ("[ ] Isolate power before opening service panels.", "UNKNOWN_REQUIRES_REVIEW"),
            ("Clean filters and confirm airflow is unobstructed.", "INFORMATIONAL_OBSERVATION"),
            ("Pressure-test, evacuate, and charge the refrigerant circuit only as specified.",
             "UNKNOWN_REQUIRES_REVIEW"),
            ("Do not repeatedly reset a tripping breaker.", "UNKNOWN_REQUIRES_REVIEW"),
            ("Annually: perform a comprehensive service by a qualified HVAC technician.",
             "TECHNICIAN_ONLY"),
            ("The interlock circuit is described in the wiring diagram.", "PROHIBITED"),
            ("", "UNKNOWN_REQUIRES_REVIEW"),
        ):
            self.assertIsNone(ca.customer_safe_fact(text, level), text)


@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class InformationalContentTests(unittest.TestCase):
    def test_documented_annual_interval_is_answered_with_its_qualification(self):
        r = ask("split", "How often should a comprehensive service be performed?")
        self.assertEqual(r.type, "DIAGNOSTIC_GUIDANCE")
        self.assertIn("Annually: perform a comprehensive service by a qualified HVAC technician.",
                      r.texts)
        self.assertTrue(any("planning intervals only" in t for t in r.texts))
        self.assertEqual(set(r.cited), {("Preventive Maintenance Checklist",
                                         "2. SUGGESTED MAINTENANCE INTERVALS")})
        self.assertEqual(set(r.kinds), {"information"})
        self.assertIn("not work for you to carry out", r.message)

    def test_documented_maintenance_frequencies(self):
        r = ask("split", "When are the filters supposed to be cleaned?")
        self.assertTrue(any(t.startswith("Quarterly: clean filters") for t in r.texts), r.texts)
        r = ask("duct", "What should be done every 6 months?")
        self.assertTrue(any(t.startswith("Every 6 months: document airflow") for t in r.texts))
        self.assertFalse(any(t.startswith(("Quarterly", "Annually")) for t in r.texts), r.texts)
        r = ask("duct", "How frequently do the filters need checking at a dusty site?")
        self.assertTrue(any(t.startswith("Monthly or more often in dusty environments")
                            for t in r.texts), r.texts)

    def test_every_disclosed_fact_is_verbatim_and_individually_cited(self):
        r = ask("duct", "What should be done every 6 months?")
        for claim, (title, heading) in zip(r.claims, r.cited):
            self.assertEqual(claim["text"], claim["source_text"])
            self.assertEqual(len(claim["evidence_ids"]), 1)
            source = next(c for c in r.chunks.values() if c.heading == heading
                          and c.document.title == title)
            self.assertIn(claim["source_text"], source.text.replace("- ", ""))

    def test_whole_section_is_not_exposed_for_one_safe_sentence(self):
        r = ask("split", "How often should a comprehensive service be performed?")
        self.assertLessEqual(len(r.texts), 2)
        self.assertNotIn("Monthly", r.dump)
        self.assertNotIn("Quarterly", r.dump)

    def test_commissioning_acceptance_criteria_stay_protected(self):
        for asset, question in (
            ("split", "What are the acceptance criteria after commissioning?"),
            ("duct", "What are the acceptance criteria for commissioning?"),
            ("vrf", "What is required for acceptance of the installation?"),
        ):
            r = ask(asset, question)
            self.assertEqual(r.type, "INSUFFICIENT_EVIDENCE", question)
            self.assertEqual(r.texts, [])

    def test_technician_only_and_restricted_procedures_are_never_disclosed(self):
        cases = (
            ("vrf", "Which conditions require escalation?"),
            ("vrf", "What vacuum level should be reached during evacuation?"),
            ("vrf", "How is the pressure test performed?"),
            ("split", "How is the refrigerant circuit charged?"),
            ("duct", "What does the checklist say about isolating power?"),
            ("vrf", "What information has to be written down when the system is commissioned?"),
        )
        restricted = re.compile(
            r"pressure-test|evacuate|refrigerant charge|brazing|isolate power|"
            r"interlock|bypass|lockout", re.I)
        for asset, question in cases:
            r = ask(asset, question)
            self.assertFalse(restricted.search(" ".join(r.texts)), (question, r.texts))
            for claim in r.claims:
                self.assertNotEqual(claim.get("kind"), "customer_procedure")

    def test_cause_questions_do_not_pick_up_unrelated_facts(self):
        r = ask("split", "What are the possible causes of poor cooling?")
        self.assertEqual(set(r.kinds), {"possible_cause"})
        self.assertEqual(len(r.texts), 4)

    def test_sentence_without_an_authorized_source_is_not_answered(self):
        name, asset_id, prefixes = ASSETS["split"]
        interval = [cid for cid, c in _chunks(prefixes, asset_id * 10).items()
                    if c.heading == "2. SUGGESTED MAINTENANCE INTERVALS"]
        r = ask("split", "How often should a comprehensive service be performed?",
                denied=interval)
        self.assertEqual(r.type, "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("Annually", r.dump)

    def test_document_for_an_unrelated_asset_is_not_used(self):
        # The VRF asset's authorized scope has no "every 6 months" entry; the
        # Ductable document that does is outside that scope.
        r = ask("vrf", "What should be done every 6 months?")
        self.assertEqual(r.type, "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("static pressure, temperature split", r.dump)


@unittest.skipUnless(HAVE_DOCS, "real source documents are not present")
class ParaphraseTests(unittest.TestCase):
    COOLING = [
        "Dirty filter or coil.",
        "Restricted airflow or blocked outdoor unit clearance.",
        "Doors/windows open or heat load exceeds capacity.",
        "Incorrect settings, fan issue, or refrigerant-system fault.",
    ]
    COMMS = [
        "Communication cable damage or incorrect topology.",
        "Duplicate/missing addresses or configuration mismatch.",
        "Power loss to a connected unit or module.",
    ]
    UNEVEN = ["Air balancing problem, duct leakage, closed dampers, or uneven heat load."]

    def test_direct_and_paraphrased_questions_reach_the_same_cited_causes(self):
        cases = (
            ("split", "My AC is running but the room is not getting cold. Why?", self.COOLING,
             "3. SYMPTOM: POOR COOLING"),
            ("split", "What causes poor cooling?", self.COOLING, "3. SYMPTOM: POOR COOLING"),
            ("duct", "Some rooms are much colder than others. What could be the reason?",
             self.UNEVEN, "3. SYMPTOM: UNEVEN ROOM TEMPERATURES"),
            ("duct", "Why are room temperatures uneven?", self.UNEVEN,
             "3. SYMPTOM: UNEVEN ROOM TEMPERATURES"),
            ("vrf", "The indoor units have lost connection with the outdoor module. "
                    "What could be wrong?", self.COMMS, "2. COMMUNICATION OR ADDRESSING FAULT"),
            ("vrf", "Why is there a communication fault?", self.COMMS,
             "2. COMMUNICATION OR ADDRESSING FAULT"),
        )
        for asset, question, expected, heading in cases:
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertEqual(r.type, "DIAGNOSTIC_GUIDANCE")
                self.assertEqual(r.texts, expected)
                self.assertEqual({h for _t, h in r.cited}, {heading})
                self.assertEqual(set(r.kinds), {"possible_cause"})
                self.assertIn("not", r.message.casefold())   # possibility, not diagnosis

    def test_superficial_word_overlap_is_not_evidence(self):
        for asset, question in (
            ("split", "Why is the cold storage room on a different circuit?"),
            ("vrf", "Why does my phone lose its internet connection?"),
            ("duct", "Why is the cold drink machine in room 4 louder than others?"),
        ):
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertEqual(r.texts, [], r.texts)

    def test_partial_heading_overlap_does_not_return_that_sections_causes(self):
        # Verified false positive: "room" and "temperature" overlap the heading
        # "UNEVEN ROOM TEMPERATURES" but the question states no such symptom.
        for asset, question in (
            ("duct", "Why is the temperature of the server room logged?"),
            ("vrf", "Why is the temperature of the server room logged?"),
            ("split", "Why is the water bill for this unit so high?"),
            ("vrf", "Why is the high voltage panel locked?"),
        ):
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertEqual(r.type, "INSUFFICIENT_EVIDENCE")
                self.assertEqual(r.texts, [])
                self.assertNotIn("Air balancing problem", r.dump)

    def test_established_symptom_wording_still_reaches_its_section(self):
        cases = (
            ("split", "Why is water leaking from the unit?", "4. SYMPTOM: WATER LEAKAGE"),
            ("split", "Why is there ice on the coil?", "5. SYMPTOM: ICE ON COIL OR PIPE"),
            ("split", "Why is the unit making noise?", "6. SYMPTOM: NOISE OR VIBRATION"),
            ("split", "Why does the unit not start?", "2. SYMPTOM: UNIT DOES NOT START"),
            ("vrf", "What causes a high pressure fault?",
             "3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT"),
            ("vrf", "Why is there a compressor fault?", "5. COMPRESSOR OR INVERTER FAULT"),
            ("duct", "What causes weak airflow?", "2. SYMPTOM: WEAK AIRFLOW"),
            ("duct", "Why are the room temperatures uneven?",
             "3. SYMPTOM: UNEVEN ROOM TEMPERATURES"),
        )
        for asset, question, heading in cases:
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertTrue(r.texts, question)
                self.assertEqual({h for _t, h in r.cited}, {heading})

    def test_paraphrase_does_not_cross_to_an_asset_without_that_section(self):
        r = ask("duct", "The indoor units have lost connection. What could be wrong?")
        self.assertEqual(r.texts, [])
        self.assertNotIn("Communication cable", r.dump)

    def test_incorrect_fault_code_and_unsupported_questions_stay_refused(self):
        for asset, question in (
            ("vrf", "What does fault code E-99 mean?"),
            ("vrf", "The room is not getting cold and it shows E-99. What does E-99 mean?"),
            ("split", "What refrigerant type does this unit use?"),
            ("duct", "What is the design static pressure for this 5-ton unit?"),
        ):
            with self.subTest(question=question):
                r = ask(asset, question)
                self.assertEqual(r.type, "INSUFFICIENT_EVIDENCE")
                self.assertEqual(r.texts, [])

    def test_question_naming_a_different_model_is_refused(self):
        r = ask("split", "Model XR-900 is not getting cold. Why?")
        self.assertEqual(r.texts, [])

    def test_named_component_still_limits_a_paraphrased_answer(self):
        r = ask("split", "The room is not getting cold. Could the filter be the cause?")
        self.assertEqual(r.texts, ["Dirty filter or coil."])


if __name__ == "__main__":
    unittest.main()


# --- whole conversation turns through the demo answer path -----------------
from types import ModuleType
from unittest.mock import Mock

from tools import servy_real_demo as demo
from tools.servy_real_demo_dialogue import CATEGORY_QUESTION

THREAD = "thread-answer-stage-0001"


def converse(asset_key, questions, *, scan=None, thread_id=THREAD, continuation=None):
    """Run consecutive customer turns through ``_answer`` over the real documents."""
    name, asset_id, prefixes = ASSETS[asset_key]
    chunks = _chunks(prefixes, asset_id * 10)
    customer_id = demo.ASSET_SCOPE[asset_id]
    tenant = SimpleNamespace(id=9)
    customer = SimpleNamespace(id=customer_id, tenant_id=9)
    asset = SimpleNamespace(id=asset_id, name=name, model_number="", asset_code="",
                            tenant_id=9, customer_id=customer_id,
                            product=SimpleNamespace(name=name))

    def source_chunk(*, tenant, customer, asset, document_id, chunk_id):
        chunk = chunks.get(chunk_id)
        if chunk is None or chunk.document_id != document_id:
            raise NotFound("Source unavailable.")
        return chunk

    rag = ModuleType("core.services.rag")
    rag.ask_rag = Mock(return_value={
        "engine": "extractive", "retrieval_backend": "oracle", "reranker_applied": False,
        "reranker_model": None,
        "references": [{"document_id": c.document_id, "chunk_id": c.id, "score": 0.9}
                       for c in chunks.values()]})
    results = []
    scan_patch = (patch.object(demo, "search_all_approved_documents", return_value=scan)
                  if scan is not None else
                  patch.object(demo, "search_all_approved_documents",
                               side_effect=RuntimeError("scan unavailable")))
    with patch.dict("sys.modules", {"core.services.rag": rag}), scan_patch, \
            patch.object(ca, "source_chunk", side_effect=source_chunk), \
            patch.object(demo, "_asset", return_value=(tenant, customer, asset)), \
            patch.object(demo, "plan_turn", return_value=None):
        for question in questions:
            result = demo._answer(asset_id, question, continuation=continuation,
                                  thread_id=thread_id)
            continuation = result["continuation"]
            results.append(result)
    return results


def _shown(result):
    presentation = result.get("presentation") or {}
    citations = {c["id"]: c for c in presentation.get("citations", [])}
    rows = []
    for section in presentation.get("sections", []):
        for claim in section["claims"]:
            evidence = claim.get("evidence_ids") or []
            rows.append((section["title"], claim["kind"], claim["text"],
                         claim.get("source_text", ""),
                         citations[evidence[0]]["heading"] if evidence else ""))
    return rows
