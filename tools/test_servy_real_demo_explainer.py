"""Offline tests for local, exact-source-bound customer explanations."""
import json
import os
from unittest import TestCase
from unittest.mock import Mock, patch

from tools import servy_real_demo_explainer as explainer


class ExplanationValidationTests(TestCase):
    def _candidate(self, ident, kind, source):
        return {"id": ident, "kind": kind, "source_text": source}

    def test_explanations_are_supported_for_all_three_asset_families(self):
        assets = (
            ("Split AC Unit 2T", "A blocked filter may reduce airflow."),
            ("VRF Outdoor Module", "Operating conditions may affect outdoor-unit operation."),
            ("Ductable AC 5T", "A fan or temperature-sensor issue may affect operation."),
        )
        for index, (asset, source) in enumerate(assets, start=1):
            with self.subTest(asset=asset):
                candidate = self._candidate("c1", "possible_cause", source)
                explanation = (
                    "The manual lists this as one possible cause category. "
                    "This passage does not identify the exact mechanism or confirm "
                    "that the condition applies to this equipment."
                )
                result = explainer._validate_explanations({
                    "explanations": [{
                        "claim_id": "c1", "explanation": explanation,
                        "source_quote": source,
                    }],
                }, [candidate])
                self.assertEqual(result, {"c1": explanation})

    def test_missing_or_changed_source_quote_is_rejected(self):
        candidate = self._candidate("c1", "possible_cause", "A sensor issue may affect operation.")
        raw = {"explanations": [{
            "claim_id": "c1",
            "explanation": "The manual lists this as one possible cause category, not a confirmed fault.",
            "source_quote": "A different sentence.",
        }]}
        self.assertIsNone(explainer._validate_explanations(raw, [candidate]))

    def test_unsupported_actions_certainty_and_new_numbers_are_rejected(self):
        source = "Operating conditions may affect outdoor-unit operation."
        candidate = self._candidate("c1", "possible_cause", source)
        cases = (
            "You should open the unit and inspect it to confirm this possible cause.",
            "This is definitely the cause of the reported failure.",
            "The unit may require 240 volts for this operating condition.",
        )
        for explanation in cases:
            with self.subTest(explanation=explanation):
                raw = {"explanations": [{
                    "claim_id": "c1", "explanation": explanation,
                    "source_quote": source,
                }]}
                self.assertIsNone(explainer._validate_explanations(raw, [candidate]))

    def test_unknown_duplicate_or_missing_claim_ids_are_rejected(self):
        candidate = self._candidate("c1", "possible_cause", "A sensor issue may affect operation.")
        good = "The manual lists this as one possible cause category, not a confirmed fault."
        for rows in (
            [{"claim_id": "c9", "explanation": good, "source_quote": candidate["source_text"]}],
            [{"claim_id": "c1", "explanation": good, "source_quote": candidate["source_text"]}] * 2,
            [],
        ):
            self.assertIsNone(explainer._validate_explanations({"explanations": rows}, [candidate]))


class SourceContextGenerationTests(TestCase):
    def test_explanation_may_use_details_from_the_same_authorized_passage_only(self):
        source = "Restricted airflow or blocked outdoor unit clearance."
        context = (
            "Restricted airflow or blocked outdoor unit clearance can reduce "
            "heat rejection and cooling capacity."
        )
        candidate = {
            "id": "c1", "kind": "possible_cause",
            "source_text": source, "source_context": context,
        }
        supported = (
            "The manual lists restricted airflow or blocked outdoor unit clearance "
            "as a possible cause that can reduce heat rejection and cooling capacity."
        )
        self.assertEqual(
            explainer._validate_explanations({
                "explanations": [{
                    "claim_id": "c1", "explanation": supported,
                    "source_quote": source,
                }],
            }, [candidate]),
            {"c1": supported},
        )
        invented = (
            "The manual lists restricted airflow as a possible cause that can "
            "damage the compressor."
        )
        self.assertIsNone(explainer._validate_explanations({
            "explanations": [{
                "claim_id": "c1", "explanation": invented,
                "source_quote": source,
            }],
        }, [candidate]))

    def test_live_path_requires_reauthorized_full_source_context(self):
        presentation = {
            "equipment": {"name": "Split AC Unit 2T", "model_number": ""},
            "citations": [{
                "id": "s1", "document_id": 168, "chunk_id": 160,
                "url": "/demo-api/source/?citation=synthetic-token",
            }],
            "supporting_claims": [{
                "id": "c1", "kind": "possible_cause",
                "text": "Possible documented cause: Dirty filter or coil.",
                "source_text": "Dirty filter or coil.",
                "evidence_ids": ["s1"],
            }],
        }
        with patch.object(explainer, "_online", return_value=False):
            result = explainer.explain_claims_with_status(
                "Explain the documented cause.", presentation,
                enabled=True, model="qwen3.5:2b-q4_K_M",
                source_context_by_claim={}, require_source_context=True,
            )
        self.assertEqual(result["generation"]["status"], "no_eligible_claims")
        self.assertEqual(result["generation"]["eligible_claims"], 0)
        self.assertFalse(result["generation"]["attempted"])

    def test_apply_preserves_original_source_claim_and_citation_bindings(self):
        presentation = {
            "supporting_claims": [{
                "id": "c1", "kind": "possible_cause",
                "text": "Possible documented cause: Dirty filter or coil.",
                "source_text": "Dirty filter or coil.",
                "evidence_ids": ["s1"],
            }],
            "sections": [{
                "number": 1, "title": "What the document says",
                "claims": [{
                    "id": "c1", "kind": "possible_cause",
                    "text": "Under Poor cooling: Possible documented cause: Dirty filter or coil.",
                    "source_text": "Dirty filter or coil.",
                    "evidence_ids": ["s1"],
                }],
            }],
            "citations": [{
                "id": "s1", "document_id": 168, "chunk_id": 160,
                "url": "/demo-api/source/?citation=synthetic-token",
            }],
        }
        explanation = (
            "The manual lists this as one possible cause category, not a confirmed fault."
        )
        full_passage_citation = {
            "id": "s2", "document_id": 168, "chunk_id": 160,
            "url": "/demo-api/source/?citation=full-passage-token",
        }
        applied = explainer.apply_customer_explanations(
            presentation, {"c1": explanation},
            context_citations_by_claim={"c1": full_passage_citation},
        )
        self.assertEqual(applied, 1)
        for claim in (
            presentation["supporting_claims"][0],
            presentation["sections"][0]["claims"][0],
        ):
            self.assertTrue(claim["text"].endswith("Explanation: " + explanation))
            self.assertEqual(claim["source_text"], "Dirty filter or coil.")
            self.assertEqual(claim["evidence_ids"], ["s1", "s2"])
        self.assertEqual(
            [item["id"] for item in presentation["citations"]],
            ["s1", "s2"],
        )
        self.assertEqual(
            presentation["citations"][0]["url"],
            "/demo-api/source/?citation=synthetic-token",
        )
        self.assertEqual(
            presentation["citations"][1]["url"],
            "/demo-api/source/?citation=full-passage-token",
        )

    def test_generated_text_is_not_applied_without_its_full_passage_citation(self):
        presentation = {
            "supporting_claims": [{
                "id": "c1", "kind": "possible_cause",
                "text": "Possible documented cause: Dirty filter or coil.",
                "source_text": "Dirty filter or coil.",
                "evidence_ids": ["s1"],
            }],
            "sections": [{
                "number": 1, "title": "What the document says",
                "claims": [{
                    "id": "c1", "kind": "possible_cause",
                    "text": "Possible documented cause: Dirty filter or coil.",
                    "source_text": "Dirty filter or coil.",
                    "evidence_ids": ["s1"],
                }],
            }],
            "citations": [{
                "id": "s1", "document_id": 168, "chunk_id": 160,
                "url": "/demo-api/source/?citation=sentence-token",
            }],
        }
        applied = explainer.apply_customer_explanations(
            presentation,
            {"c1": "The manual lists this as one possible cause category, not a confirmed fault."},
            context_citations_by_claim={},
        )
        self.assertEqual(applied, 0)
        self.assertEqual(
            presentation["supporting_claims"][0]["text"],
            "Possible documented cause: Dirty filter or coil.",
        )
        self.assertEqual(
            presentation["sections"][0]["claims"][0]["text"],
            "Possible documented cause: Dirty filter or coil.",
        )
        self.assertEqual(len(presentation["citations"]), 1)


class LocalExplanationCallTests(TestCase):
    def setUp(self):
        self.presentation = {
            "equipment": {"id": 60, "name": "VRF Outdoor Module", "model_number": "VRF-OUT"},
            "citations": [{
                "id": "s1", "document_id": 9, "chunk_id": 22,
                "url": "/demo-api/source/?citation=synthetic-token",
            }],
            "supporting_claims": [{
                "id": "c1", "kind": "possible_cause",
                "text": "Possible documented cause: Operating conditions may affect operation.",
                "source_text": "Operating conditions may affect operation.",
                "evidence_ids": ["s1"],
            }],
        }

    def test_local_call_sends_only_approved_claims_to_loopback(self):
        session = Mock()
        session.trust_env = True
        response = Mock()
        source = self.presentation["supporting_claims"][0]["source_text"]
        explanation = (
            "The manual lists this as one possible cause category. "
            "This passage does not identify the exact mechanism or confirm "
            "that the condition applies to this equipment."
        )
        response.status_code = 200
        response.content = b'{}'
        response.json.return_value = {"message": {"content": json.dumps({
            "explanations": [{
                "claim_id": "c1", "explanation": explanation,
                "source_quote": source,
            }],
        })}}
        session.post.return_value = response
        with patch.dict(os.environ, {
            "SERVY_REAL_DEMO_DIALOGUE_MODE": "local",
            "SERVY_REAL_DEMO_CHAT_MODEL": "qwen2.5:3b",
        }), patch.object(explainer, "_online", return_value=True), \
             patch.object(explainer.requests, "Session", return_value=session):
            result = explainer.explain_claims(
                "Explain each issue in detail.", self.presentation,
            )
        self.assertEqual(result, {"c1": explanation})
        self.assertFalse(session.trust_env)
        args, kwargs = session.post.call_args
        self.assertEqual(args, (explainer.ENDPOINT,))
        self.assertEqual(explainer.ENDPOINT, "http://127.0.0.1:11434/api/chat")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["json"]["model"], "qwen2.5:3b")
        self.assertIn("Operating conditions may affect operation.", kwargs["json"]["messages"][1]["content"])
        session.close.assert_called_once_with()

    def test_explicit_customer_qwen_runs_even_when_dialogue_planner_is_in_rules_mode(self):
        session = Mock()
        response = Mock()
        source = self.presentation["supporting_claims"][0]["source_text"]
        explanation = (
            "The manual lists this as one possible cause category. "
            "This passage does not identify the exact mechanism or confirm "
            "that the condition applies to this equipment."
        )
        response.status_code = 200
        response.content = b"{}"
        response.json.return_value = {"message": {"content": json.dumps({
            "explanations": [{
                "claim_id": "c1", "explanation": explanation,
                "source_quote": source,
            }],
        })}}
        session.post.return_value = response
        with patch.dict(os.environ, {
            "SERVY_REAL_DEMO_DIALOGUE_MODE": "rules",
            "SERVY_CUSTOMER_ANSWER_GENERATION": "ollama",
        }), patch.object(explainer, "_online", return_value=True), \
             patch.object(explainer.requests, "Session", return_value=session):
            result = explainer.explain_claims_with_status(
                "Explain each documented issue.", self.presentation,
                enabled=True,
                model="qwen3.5:2b-q4_K_M",
                source_context_by_claim={"c1": source},
                require_source_context=True,
            )
        self.assertEqual(result["generation"]["status"], "applied")
        self.assertEqual(result["generation"]["provider"], "ollama")
        self.assertEqual(result["generation"]["model"], "qwen3.5:2b-q4_K_M")
        self.assertTrue(result["generation"]["attempted"])
        self.assertEqual(result["generation"]["accepted_claims"], 1)
        self.assertEqual(result["explanations"], {"c1": explanation})
        args, kwargs = session.post.call_args
        self.assertEqual(kwargs["json"]["model"], "qwen3.5:2b-q4_K_M")
        self.assertIn(source, kwargs["json"]["messages"][1]["content"])
        session.close.assert_called_once_with()

    def test_rules_mode_and_unavailable_local_model_fail_closed(self):
        with patch.dict(os.environ, {"SERVY_REAL_DEMO_DIALOGUE_MODE": "rules"}):
            with patch.object(explainer, "_online", side_effect=AssertionError("must not probe")):
                self.assertEqual(explainer.explain_claims("Explain this.", self.presentation), {})
        with patch.dict(os.environ, {"SERVY_REAL_DEMO_DIALOGUE_MODE": "local"}), \
             patch.object(explainer, "_online", return_value=False):
            self.assertEqual(explainer.explain_claims("Explain this.", self.presentation), {})

    def test_invalid_citation_scope_never_reaches_local_model(self):
        changed = dict(self.presentation)
        changed["citations"] = [{
            "id": "s1", "document_id": 9, "chunk_id": 22,
            "url": "https://untrusted.invalid/source",
        }]
        with patch.dict(os.environ, {"SERVY_REAL_DEMO_DIALOGUE_MODE": "local"}), \
             patch.object(explainer, "_online", side_effect=AssertionError("must not call")):
            self.assertEqual(explainer.explain_claims("Explain this.", changed), {})
