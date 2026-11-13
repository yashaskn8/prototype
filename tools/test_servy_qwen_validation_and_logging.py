"""Regression tests for Qwen source validation and the customer terminal log.

Offline only: the model, network and database are replaced. The validator is
exercised with the explanations the real local model produced for the nine
approved manuals (see the root-cause notes in each test).
"""
import io
import json
import os
import threading
import unittest
from contextlib import nullcontext, redirect_stdout
from unittest.mock import Mock, patch

os.environ.setdefault("LOCALAPPDATA", "/tmp")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

from tools import servy_real_demo as demo
from tools import servy_real_demo_explainer as explainer

POOR_COOLING = "Dirty filter or coil."
COMMS = "Communication cable damage or incorrect topology."
SCHEDULE = "Annually: perform a comprehensive service by a qualified HVAC technician."


def _candidate(ident, kind, source, heading="3. SYMPTOM: POOR COOLING"):
    return {"id": ident, "kind": kind, "source_text": source,
            "source_context": "Section heading: " + heading
                              + "\nExact approved source statement: " + source}


def _row(ident, explanation, quote):
    return {"claim_id": ident, "explanation": explanation, "source_quote": quote}


class SourceQuoteBindingTests(unittest.TestCase):
    def test_json_noise_after_the_exact_quote_is_tolerated(self):
        # Root cause 1: under constrained decoding the model appended a stray
        # comma to 10 of 15 quotes, so every one of those rows was rejected.
        cand = _candidate("c1", "possible_cause", COMMS, "2. COMMUNICATION OR ADDRESSING FAULT")
        good = "The manual lists communication cable damage or incorrect topology as a possible cause."
        for quote in (COMMS, COMMS + ",", COMMS + " ,", "  " + COMMS + "  "):
            with self.subTest(quote=quote):
                self.assertEqual(
                    explainer._validate_explanations({"explanations": [_row("c1", good, quote)]}, [cand]),
                    {"c1": good})

    def test_any_change_to_the_quoted_words_is_still_rejected(self):
        cand = _candidate("c1", "possible_cause", COMMS, "2. COMMUNICATION OR ADDRESSING FAULT")
        good = "The manual lists communication cable damage or incorrect topology as a possible cause."
        for quote in ("Communication cable damage or correct topology.",
                      "communication cable damage or incorrect topology.",
                      "Communication cable damage.", "Possible documented cause: " + COMMS, ""):
            with self.subTest(quote=quote):
                self.assertIsNone(explainer._validate_explanations(
                    {"explanations": [_row("c1", good, quote)]}, [cand]))

    def test_schema_only_allows_exact_supplied_source_statements(self):
        presentation = {
            "equipment": {"name": "Split AC Unit 2T", "model_number": ""},
            "citations": [{"id": "s1", "url": "/demo-api/source/?citation=synthetic-token",
                           "document_id": 168, "chunk_id": 9}],
            "supporting_claims": [{"id": "c1", "kind": "possible_cause", "source_text": POOR_COOLING,
                                   "text": "Possible documented cause: " + POOR_COOLING,
                                   "evidence_ids": ["s1"]}],
        }
        session = Mock()
        session.post.return_value.status_code = 200
        session.post.return_value.content = b"{}"
        session.post.return_value.json.return_value = {"message": {"content": "{}"}}
        with patch.object(explainer, "_online", return_value=True), \
                patch.object(explainer.requests, "Session", return_value=session):
            explainer.explain_claims_with_status(
                "Why is it not cooling?", presentation,
                source_context_by_claim={"c1": "Section heading: x\nExact approved source statement: "
                                         + POOR_COOLING},
                enabled=True, require_source_context=True)
        schema = session.post.call_args.kwargs["json"]["format"]
        quote = schema["properties"]["explanations"]["items"]["properties"]["source_quote"]
        self.assertEqual(quote["enum"], [POOR_COOLING])


class VocabularyGateTests(unittest.TestCase):
    def test_limitation_wording_the_prompt_asks_for_is_accepted(self):
        # Root cause 2: the system prompt asks Qwen to say the manual "does not
        # state its mechanism", but the gate rejected state/explain/how/though.
        cand = _candidate("c1", "possible_cause", COMMS, "2. COMMUNICATION OR ADDRESSING FAULT")
        for explanation in (
            "The manual lists communication cable damage or incorrect topology as a possible cause, "
            "but does not state its mechanism.",
            "Communication cable damage or incorrect topology is listed as a possible cause, though "
            "the manual does not explain how it leads to the fault.",
        ):
            with self.subTest(explanation=explanation):
                self.assertEqual(explainer._validate_explanations(
                    {"explanations": [_row("c1", explanation, COMMS)]}, [cand]), {"c1": explanation})

    def test_inflections_of_source_words_count_as_source_words(self):
        source = "Air balancing problem, duct leakage, closed dampers, or uneven heat load."
        cand = _candidate("c1", "possible_cause", source, "3. SYMPTOM: UNEVEN ROOM TEMPERATURES")
        explanation = ("The manual lists air balancing problems, duct leakage, closed dampers or "
                       "uneven heat load as possible causes of uneven room temperatures.")
        self.assertEqual(explainer._validate_explanations(
            {"explanations": [_row("c1", explanation, source)]}, [cand]), {"c1": explanation})

    def test_selected_equipment_name_is_allowed_but_not_other_equipment(self):
        cand = _candidate("c1", "possible_cause", POOR_COOLING)
        named = ("For the Ductable AC 5T system, the manual lists dirty filter or coil as a "
                 "possible cause of poor cooling.")
        raw = {"explanations": [_row("c1", named, POOR_COOLING)]}
        self.assertIsNone(explainer._validate_explanations(raw, [cand]))
        self.assertEqual(explainer._validate_explanations(
            raw, [cand], extra_vocabulary=frozenset({"ductable", "ac", "5t"})), {"c1": named})

    def test_genuinely_added_content_is_still_rejected(self):
        source = "Blocked airflow, dirty coil, fan issue, or operating conditions outside design."
        cand = _candidate("c1", "possible_cause", source, "3. HIGH-PRESSURE OR HEAT-EXCHANGE FAULT")
        rejected = {
            "unsupported_vocabulary": (
                "A high-pressure fault may be caused by blocked airflow or operating conditions "
                "that are not within the design specifications."),
            "unsupported_component": (
                "The manual lists blocked airflow as a possible cause; a failed compressor is "
                "another possible cause."),
            "unsupported_number_or_code": (
                "The manual lists blocked airflow as a possible cause when pressure exceeds 42 bar."),
            "customer_directive": (
                "The manual lists blocked airflow as a possible cause, so you should check the coil."),
            "missing_uncertainty": (
                "Blocked airflow, dirty coil, fan issue, or operating conditions outside design "
                "produce this fault."),
        }
        # Each row adds something the manual does not say. The vocabulary
        # gate usually fires first; the later rules are independent backstops.
        rule_names = {"unsupported_vocabulary", "unsupported_component",
                      "unsupported_number_or_code", "customer_directive", "missing_uncertainty"}
        for label, explanation in rejected.items():
            with self.subTest(label=label):
                reasons = {}
                self.assertIsNone(explainer._validate_explanations(
                    {"explanations": [_row("c1", explanation, source)]}, [cand], reasons))
                self.assertIn(reasons.get("c1"), rule_names)

    def test_backstop_rules_reject_even_glue_only_wording(self):
        # Wording made only of permitted words still cannot drop uncertainty
        # or give the customer a directive.
        cand = _candidate("c1", "possible_cause", COMMS, "2. COMMUNICATION OR ADDRESSING FAULT")
        cases = {
            "missing_uncertainty": "Communication cable damage or incorrect topology is the cause here.",
        }
        for expected, explanation in cases.items():
            with self.subTest(expected=expected):
                reasons = {}
                self.assertIsNone(explainer._validate_explanations(
                    {"explanations": [_row("c1", explanation, COMMS)]}, [cand], reasons))
                self.assertEqual(reasons, {"c1": expected})

    def test_obligation_wording_of_a_schedule_entry_stays_rejected(self):
        cand = _candidate("c1", "information", SCHEDULE, "2. SUGGESTED MAINTENANCE INTERVALS")
        explanation = "A comprehensive service should be performed annually by a qualified HVAC technician."
        reasons = {}
        self.assertIsNone(explainer._validate_explanations(
            {"explanations": [_row("c1", explanation, SCHEDULE)]}, [cand], reasons))
        self.assertEqual(reasons, {"c1": "unsupported_vocabulary"})


class PerClaimAcceptanceTests(unittest.TestCase):
    def test_one_invalid_row_no_longer_discards_valid_rows(self):
        # Root cause 3: the validator was all-or-nothing, so one rejected row
        # threw away every valid explanation in the same answer.
        a = _candidate("c1", "possible_cause", POOR_COOLING)
        b = _candidate("c2", "possible_cause", COMMS, "2. COMMUNICATION OR ADDRESSING FAULT")
        good = "The manual lists dirty filter or coil as a possible cause of poor cooling."
        bad = "Communication cable damage means the compressor must be replaced."
        reasons = {}
        result = explainer._validate_explanations(
            {"explanations": [_row("c1", good, POOR_COOLING), _row("c2", bad, COMMS)]}, [a, b], reasons)
        self.assertEqual(result, {"c1": good})
        self.assertEqual(set(reasons), {"c2"})

    def test_malformed_response_structure_still_rejects_everything(self):
        a = _candidate("c1", "possible_cause", POOR_COOLING)
        b = _candidate("c2", "possible_cause", COMMS, "2. COMMUNICATION OR ADDRESSING FAULT")
        good = "The manual lists dirty filter or coil as a possible cause of poor cooling."
        for rows in ([_row("c1", good, POOR_COOLING)],
                     [_row("c1", good, POOR_COOLING), _row("c1", good, POOR_COOLING)],
                     [_row("c1", good, POOR_COOLING), _row("c9", good, COMMS)]):
            with self.subTest(rows=len(rows)):
                self.assertIsNone(explainer._validate_explanations({"explanations": rows}, [a, b]))

    def test_partial_result_is_reported_with_rule_names_only(self):
        presentation = {
            "equipment": {"name": "Split AC Unit 2T", "model_number": ""},
            "citations": [
                {"id": "s1", "url": "/demo-api/source/?citation=synthetic-token-1",
                 "document_id": 168, "chunk_id": 9},
                {"id": "s2", "url": "/demo-api/source/?citation=synthetic-token-2",
                 "document_id": 168, "chunk_id": 9}],
            "supporting_claims": [
                {"id": "c1", "kind": "possible_cause", "source_text": POOR_COOLING,
                 "text": "Possible documented cause: " + POOR_COOLING, "evidence_ids": ["s1"]},
                {"id": "c2", "kind": "possible_cause", "source_text": COMMS,
                 "text": "Possible documented cause: " + COMMS, "evidence_ids": ["s2"]}],
        }
        content = json.dumps({"explanations": [
            _row("c1", "The manual lists dirty filter or coil as a possible cause of poor cooling.",
                 POOR_COOLING),
            _row("c2", "Secret technician wiring diagram detail about the compressor.", COMMS)]})
        session = Mock()
        session.post.return_value.status_code = 200
        session.post.return_value.content = content.encode()
        session.post.return_value.json.return_value = {"message": {"content": content}}
        contexts = {c["id"]: "Section heading: 3. SYMPTOM: POOR COOLING\nExact approved source statement: "
                    + c["source_text"] for c in presentation["supporting_claims"]}
        with patch.object(explainer, "_online", return_value=True), \
                patch.object(explainer.requests, "Session", return_value=session):
            result = explainer.explain_claims_with_status(
                "Why?", presentation, source_context_by_claim=contexts,
                enabled=True, require_source_context=True)
        generation = result["generation"]
        self.assertEqual(generation["status"], "applied_partial")
        self.assertEqual((generation["accepted_claims"], generation["rejected_claims"]), (1, 1))
        self.assertEqual(set(result["explanations"]), {"c1"})
        self.assertNotIn("wiring", json.dumps(generation))      # no model text in the report
        applied = explainer.apply_customer_explanations(presentation, result["explanations"])
        self.assertEqual(applied, 1)
        self.assertIn("Explanation:", presentation["supporting_claims"][0]["text"])
        self.assertEqual(presentation["supporting_claims"][1]["text"],
                         "Possible documented cause: " + COMMS)     # exact fallback kept


class TerminalLogTests(unittest.TestCase):
    def _post(self, path, payload, answer=None):
        handler = object.__new__(demo.Handler)
        handler.path = path
        handler.command = "POST"
        handler._authorized = lambda: True
        handler._send = lambda status, value: (status, value)
        handler._send_rag_stream = lambda work: (200, work())
        data = json.dumps(payload).encode("utf-8")
        handler.headers = {"Content-Length": str(len(data))}
        handler.rfile = io.BytesIO(data)
        output = io.StringIO()
        with redirect_stdout(output), \
                patch.object(demo, "read_only_database", return_value=nullcontext()), \
                patch.object(demo, "_answer", return_value=answer or {"response_type": "CLARIFICATION"}):
            result = handler.do_POST()
        return result, output.getvalue()

    def test_each_customer_query_logs_exactly_one_accepted_line(self):
        question = "My private complaint text about the unit"
        for path in ("/demo-api/query/", "/demo-api/query-stream/"):
            with self.subTest(path=path):
                (code, _response), text = self._post(path, {"asset_id": 60, "question": question})
                self.assertEqual(code, 200)
                lines = text.splitlines(keepends=True)
                accepted = [line for line in lines if "RAG_REQUEST_ACCEPTED" in line]
                self.assertEqual(len(accepted), 1)
                self.assertTrue(accepted[0].endswith("\n"))
                self.assertEqual(accepted[0].count("RAG_REQUEST_ACCEPTED"), 1)
                self.assertIn("route=" + path, accepted[0])
                self.assertNotIn("private complaint", text)
                self.assertNotIn("RAG_STAGE", text)

    def test_invalid_or_unauthenticated_requests_are_not_logged_as_accepted(self):
        (code, _), text = self._post("/demo-api/query/", {"asset_id": "sixty", "question": "x"})
        self.assertEqual(code, 400)
        self.assertNotIn("RAG_REQUEST_ACCEPTED", text)
        handler = object.__new__(demo.Handler)
        handler.path = "/demo-api/query/"
        handler._authorized = lambda: False
        handler._send = lambda status, value: (status, value)
        output = io.StringIO()
        with redirect_stdout(output):
            code, _ = handler.do_POST()
        self.assertEqual(code, 401)
        self.assertNotIn("RAG_REQUEST_ACCEPTED", output.getvalue())

    def test_concurrent_terminal_lines_are_never_concatenated(self):
        output = io.StringIO()
        lines = [f"line-{n}-" + "x" * 200 for n in range(200)]
        with redirect_stdout(output):
            threads = [threading.Thread(target=demo._terminal_line, args=(line,)) for line in lines]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        written = output.getvalue().splitlines()
        self.assertEqual(sorted(written), sorted(lines))

    def test_embedded_line_breaks_cannot_split_or_forge_a_log_line(self):
        output = io.StringIO()
        with redirect_stdout(output):
            demo._terminal_line("first\nRAG_REQUEST_ACCEPTED forged\r")
        self.assertEqual(output.getvalue().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
