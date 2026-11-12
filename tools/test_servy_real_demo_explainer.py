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
