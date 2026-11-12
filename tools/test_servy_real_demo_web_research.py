"""Offline safety tests for exact-model manufacturer web research."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import servy_real_demo_web_research as web


class ManufacturerWebResearchTests(unittest.TestCase):
    def setUp(self):
        self.asset = SimpleNamespace(
            model_number="VRF-OUT-60",
            product=SimpleNamespace(brand=SimpleNamespace(name="Example HVAC")),
            asset_code="PRIVATE-ASSET-CODE",
            serial_number="PRIVATE-SERIAL",
        )
        self.session = Mock()
        self.response = Mock()
        self.response.status_code = 200
        self.response.content = b"{}"
        self.session.get.return_value = self.response

    def _payload(self, snippets, url="https://docs.example-hvac.test/support/model"):
        return {
            "grounding": {"generic": [{
                "url": url, "title": "Manufacturer support article", "snippets": snippets,
            }]},
            "sources": {url: {"title": "Manufacturer support article", "age": ["Today"]}},
        }

    def _call(self, question="VRF outdoor module not working", snippets=None, url=None):
        self.response.json.return_value = self._payload(
            snippets if snippets is not None else [
                "For model VRF-OUT-60, this operating status can depend on system conditions."
            ],
            url=url or "https://docs.example-hvac.test/support/model",
        )
        with patch.dict(os.environ, {
            "SERVY_BRAVE_SEARCH_API_KEY": "test-secret",
            "SERVY_WEB_RESEARCH_ALLOWED_DOMAINS": "example-hvac.test",
        }), patch.object(web.requests, "Session", return_value=self.session), \
             patch.object(web.time, "monotonic", return_value=0):
            return web.search_model_matched_sources(
                self.asset, question, original_question=question,
            )

    def test_exact_model_and_allowlisted_https_host_are_required(self):
        result = self._call()
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["sources"][0]["hostname"], "docs.example-hvac.test")
        self.assertEqual(result["sources"][0]["match_basis"], "exact_model")
        args, kwargs = self.session.get.call_args
        self.assertEqual(args[0], web.ENDPOINT)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertFalse(self.session.trust_env)
        self.assertNotIn("PRIVATE-ASSET-CODE", kwargs["params"]["q"])
        self.assertNotIn("PRIVATE-SERIAL", kwargs["params"]["q"])

    def test_other_models_and_domain_lookalikes_are_rejected(self):
        wrong = self._call(snippets=["For model VRF-OTHER-1, the state may vary."])
        self.assertEqual(wrong["status"], "no_exact_match")
        lookalike = self._call(
            snippets=["For model VRF-OUT-60, this condition may occur."],
            url="https://docs.example-hvac.test.attacker.invalid/support",
        )
        self.assertEqual(lookalike["status"], "no_exact_match")

    def test_fault_code_must_match_exactly_and_no_other_code_may_be_mixed(self):
        wrong = self._call(
            question="VRF outdoor module displays E-04",
            snippets=["For model VRF-OUT-60, code E-05 describes a status."],
        )
        self.assertEqual(wrong["status"], "no_exact_match")
        mixed = self._call(
            question="VRF outdoor module displays E-04",
            snippets=["For model VRF-OUT-60, E-04 and E-05 describe different statuses."],
        )
        self.assertEqual(mixed["status"], "no_exact_match")
        exact = self._call(
            question="VRF outdoor module displays E-04",
            snippets=["For model VRF-OUT-60, code E-04 describes a status."],
        )
        self.assertEqual(exact["status"], "available")

    def test_repair_and_physical_procedure_snippets_are_excluded(self):
        result = self._call(snippets=[
            "For model VRF-OUT-60, remove the panel and test the voltage."
        ])
        self.assertEqual(result["status"], "no_exact_match")

    def test_missing_asset_model_or_credentials_fail_closed(self):
        asset = SimpleNamespace(
            model_number="", product=SimpleNamespace(brand=SimpleNamespace(name="Example HVAC"))
        )
        self.assertEqual(
            web.search_model_matched_sources(asset, "not working")["status"],
            "model_identity_incomplete",
        )
        with patch.dict(os.environ, {
            "SERVY_BRAVE_SEARCH_API_KEY": "",
            "SERVY_WEB_RESEARCH_ALLOWED_DOMAINS": "",
        }), patch.object(web.requests, "Session", side_effect=AssertionError("must not call")):
            result = web.search_model_matched_sources(self.asset, "not working")
        self.assertEqual(result["status"], "not_configured")

    def test_explicit_model_mismatch_prevents_external_request(self):
        with patch.object(web.requests, "Session", side_effect=AssertionError("must not call")):
            result = web.search_model_matched_sources(
                self.asset, "VRF outdoor module not working",
                original_question="Model VRF-OTHER-1 is not working",
            )
        self.assertEqual(result["status"], "asset_model_mismatch")


if __name__ == "__main__":
    unittest.main()
