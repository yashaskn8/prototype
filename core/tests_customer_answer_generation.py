"""Offline tests for Qwen wording of already-authorized customer claims."""
import json
from unittest import TestCase
from unittest.mock import Mock, patch

from django.test import override_settings

from core.services import customer_answer_generation as generation


def _presentation():
    source = "A blocked filter may reduce airflow."
    return {
        "schema_version": "customer-answer-v1",
        "response_type": "DIAGNOSTIC_GUIDANCE",
        "equipment": {"id": 60, "name": "VRF Outdoor Module", "model_number": "VRF-OUT"},
        "citations": [{
            "id": "s1", "document_id": 9, "chunk_id": 22,
            "url": "/api/customer-support/source/?citation=synthetic-token",
        }],
        "supporting_claims": [{
            "id": "c1", "kind": "possible_cause",
            "text": "Possible documented cause: " + source,
            "source_text": source, "evidence_ids": ["s1"],
        }],
        "sections": [{
            "number": 1, "title": "Documented possible causes",
            "claims": [{
                "id": "c1", "kind": "possible_cause",
                "text": "Possible documented cause: " + source,
                "source_text": source, "evidence_ids": ["s1"],
            }],
        }],
    }


def _good_text():
    return (
        "The manual lists a blocked filter as one possible cause that may reduce airflow; "
        "whether it applies to the unit is unverified."
    )


def _raw(text=None, source=None):
    return {"paraphrases": [{
        "claim_id": "c1",
        "display_text": text or _good_text(),
        "source_quote": source or "A blocked filter may reduce airflow.",
    }]}


class CustomerGenerationStartupValidationTests(TestCase):
    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_URL="http://127.0.0.1:11434/api/chat",
        SERVY_CUSTOMER_GENERATION_MODEL="qwen3.5:2b-q4_K_M",
        SERVY_CUSTOMER_GENERATION_TIMEOUT=45,
        SERVY_CUSTOMER_GENERATION_MAX_CLAIMS=6,
        SERVY_ALLOW_REMOTE_LLM=False,
    )
    def test_valid_quantized_qwen_tag_and_installed_local_model_pass(self):
        client = Mock()
        response = Mock()
        response.status_code = 200
        response.content = b'{"models":[{"name":"qwen3.5:2b-q4_K_M"}]}'
        response.json.return_value = {"models": [{"name": "qwen3.5:2b-q4_K_M"}]}
        client.get.return_value = response
        with patch("core.services.customer_answer_generation.requests.Session", return_value=client):
            config = generation.validate_customer_generation_configuration(check_runtime=True)
        self.assertEqual(config["provider"], "ollama")
        self.assertEqual(config["model"], "qwen3.5:2b-q4_K_M")
        self.assertTrue(config["model_ready"])
        self.assertFalse(client.trust_env)
        args, kwargs = client.get.call_args
        self.assertEqual(args, ("http://127.0.0.1:11434/api/tags",))
        self.assertFalse(kwargs["allow_redirects"])
        client.close.assert_called_once_with()

    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_URL="https://example.com/api/chat",
        SERVY_ALLOW_REMOTE_LLM=False,
    )
    def test_remote_customer_qwen_endpoint_fails_startup_validation(self):
        with self.assertRaisesRegex(RuntimeError, "loopback Ollama endpoint"):
            generation.validate_customer_generation_configuration(check_runtime=True)

    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_URL="http://127.0.0.1:11434/api/chat",
        SERVY_CUSTOMER_GENERATION_MODEL="qwen3.5:2b-q4_K_M",
        SERVY_ALLOW_REMOTE_LLM=True,
    )
    def test_remote_llm_flag_fails_startup_validation(self):
        with self.assertRaisesRegex(RuntimeError, "remote model access"):
            generation.validate_customer_generation_configuration()

    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_URL="http://127.0.0.1:11434/api/chat",
        SERVY_CUSTOMER_GENERATION_MODEL="qwen3.5:2b-q4_K_M",
        SERVY_ALLOW_REMOTE_LLM=False,
    )
    def test_requested_model_missing_from_ollama_fails_startup_validation(self):
        client = Mock()
        response = Mock()
        response.status_code = 200
        response.content = b'{"models":[]}'
        response.json.return_value = {"models": []}
        client.get.return_value = response
        with patch("core.services.customer_answer_generation.requests.Session", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                generation.validate_customer_generation_configuration(check_runtime=True)
        client.close.assert_called_once_with()

    @override_settings(SERVY_CUSTOMER_ANSWER_GENERATION="disabled")
    def test_extract_only_mode_does_not_require_or_probe_ollama(self):
        with patch("core.services.customer_answer_generation.requests.Session") as session:
            config = generation.validate_customer_generation_configuration(check_runtime=True)
        self.assertEqual(config, {"enabled": False, "provider": "disabled"})
        session.assert_not_called()


class CustomerClaimTextValidationTests(TestCase):
    def test_safe_paraphrase_preserves_all_source_terms_and_uncertainty(self):
        item = _presentation()["supporting_claims"][0]
        self.assertTrue(generation._validate_paraphrase(
            item["source_text"], _good_text(), "possible_cause"
        ))

    def test_new_component_or_mechanism_is_rejected(self):
        unsafe = (
            "The manual lists a blocked filter as one possible cause that may reduce airflow "
            "and damage the compressor; this does not confirm that it applies to the unit."
        )
        self.assertFalse(generation._validate_paraphrase(
            "A blocked filter may reduce airflow.", unsafe, "possible_cause"
        ))

    def test_reordered_source_terms_are_rejected(self):
        source = "Blocked filter may reduce airflow."
        reordered = (
            "The manual lists a filter blocked as one possible cause that may reduce airflow; "
            "this does not confirm that it applies to the unit."
        )
        self.assertFalse(generation._validate_paraphrase(
            source, reordered, "possible_cause"
        ))

    def test_new_numbers_and_fault_codes_are_rejected(self):
        unsafe = (
            "The manual lists a blocked filter as one possible cause that may reduce airflow "
            "by 30%; this does not confirm that it applies to the unit."
        )
        self.assertFalse(generation._validate_paraphrase(
            "A blocked filter may reduce airflow.", unsafe, "possible_cause"
        ))

    def test_removed_negation_or_uncertainty_is_rejected(self):
        source = "The sensor may not indicate a confirmed fault."
        self.assertFalse(generation._validate_paraphrase(
            source, "The manual states the sensor indicates a confirmed fault.", "information"
        ))

    def test_new_customer_action_or_instruction_is_rejected(self):
        unsafe = (
            "The manual lists a blocked filter as one possible cause that may reduce airflow; "
            "you should open the unit and inspect it."
        )
        self.assertFalse(generation._validate_paraphrase(
            "A blocked filter may reduce airflow.", unsafe, "possible_cause"
        ))

    def test_claim_id_quote_and_complete_row_set_are_required(self):
        candidate = [{"id": "c1", "kind": "possible_cause",
                      "source_text": "A blocked filter may reduce airflow."}]
        self.assertEqual(generation._validate_response(
            _raw(source="A different sentence."), candidate
        ), {})
        changed = _raw()
        changed["paraphrases"][0]["claim_id"] = "c9"
        self.assertEqual(generation._validate_response(changed, candidate), {})
        changed = _raw()
        changed["paraphrases"].append(dict(changed["paraphrases"][0]))
        self.assertEqual(generation._validate_response(changed, candidate), {})

    def test_valid_claim_survives_an_invalid_sibling_paraphrase(self):
        candidates = [
            {"id": "c1", "kind": "possible_cause",
             "source_text": "A blocked filter may reduce airflow."},
            {"id": "c2", "kind": "possible_cause",
             "source_text": "Blocked outdoor clearance may reduce cooling."},
        ]
        raw = {"paraphrases": [
            {"claim_id": "c1", "display_text": _good_text(),
             "source_quote": "A blocked filter may reduce airflow."},
            {"claim_id": "c2",
             "display_text": (
                 "Blocked outdoor clearance may reduce cooling and damage the compressor."
             ),
             "source_quote": "Blocked outdoor clearance may reduce cooling."},
        ]}
        self.assertEqual(
            generation._validate_response(raw, candidates),
            {"c1": _good_text()},
        )

    def test_document_fact_and_customer_procedure_claims_are_not_rephrased(self):
        presentation = _presentation()
        presentation["supporting_claims"][0]["kind"] = "document_fact"
        self.assertEqual(generation._eligible_claims(presentation), [])

    def test_unauthorized_kinds_and_untrusted_urls_are_not_candidates(self):
        presentation = _presentation()
        presentation["supporting_claims"].append({
            "id": "c2", "kind": "customer_procedure",
            "source_text": "Turn the unit off.", "evidence_ids": ["s1"],
        })
        presentation["citations"][0]["url"] = "https://untrusted.invalid/source"
        self.assertEqual(generation._eligible_claims(presentation), [])

    def test_application_changes_display_text_not_source_binding(self):
        presentation = _presentation()
        source = presentation["supporting_claims"][0]["source_text"]
        with patch.object(generation, "generate_customer_claim_texts",
                          return_value={"c1": _good_text()}):
            self.assertTrue(generation.apply_customer_answer_generation(
                presentation, "What causes poor airflow?"
            ))
        displayed = presentation["sections"][0]["claims"][0]
        backing = presentation["supporting_claims"][0]
        self.assertEqual(displayed["text"], _good_text())
        self.assertEqual(backing["text"], _good_text())
        self.assertEqual(displayed["source_text"], source)
        self.assertEqual(displayed["evidence_ids"], ["s1"])
        self.assertEqual(presentation["citations"][0]["document_id"], 9)


    def test_demo_heading_prefix_is_preserved_after_focus(self):
        presentation = _presentation()
        original = presentation["supporting_claims"][0]
        original["text"] = "Under Symptom: poor cooling: Possible documented cause: " + original["source_text"]
        # The focused dialogue creates a separate display claim; both bindings remain intact.
        focused = dict(original)
        presentation["supporting_claims"] = [focused]
        presentation["sections"][0]["claims"] = [focused]
        with patch.object(generation, "generate_customer_claim_texts", return_value={"c1": _good_text()}):
            self.assertTrue(generation.apply_customer_answer_generation(
                presentation, "What causes poor airflow?"
            ))
        self.assertTrue(focused["text"].startswith("Under Symptom: poor cooling: "))
        self.assertIn(_good_text(), focused["text"])
        self.assertEqual(focused["source_text"], "A blocked filter may reduce airflow.")

class CustomerParaphraseValidationRegressionTests(TestCase):
    def test_source_backed_open_state_is_not_misclassified_as_an_instruction(self):
        source = "Doors/windows open or heat load exceeds capacity."
        generated = (
            "A possible cause is doors/windows open or heat load exceeds capacity."
        )
        self.assertTrue(generation._validate_paraphrase(
            source, generated, "possible_cause"
        ))

    def test_model_cannot_add_an_unstated_maintenance_instruction(self):
        source = "Dirty filter or coil."
        generated = "You should clean the dirty filter or coil."
        self.assertFalse(generation._validate_paraphrase(
            source, generated, "possible_cause"
        ))


class LocalQwenTransportTests(TestCase):
    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_URL="http://127.0.0.1:11434/api/chat",
        SERVY_CUSTOMER_GENERATION_MODEL="qwen3.5:2b-q4_K_M",
        SERVY_CUSTOMER_GENERATION_TIMEOUT=45,
        SERVY_CUSTOMER_GENERATION_MAX_CLAIMS=6,
    )
    def test_ollama_receives_only_approved_claims_without_proxy_or_redirects(self):
        session = Mock()
        response = Mock()
        response.status_code = 200
        response.content = b'{}'
        response.json.return_value = {"message": {"content": json.dumps(_raw())}}
        session.post.return_value = response
        with (
            patch.object(generation, "_server_online", return_value=True),
            patch("core.services.customer_answer_generation.requests.Session", return_value=session),
        ):
            result = generation.generate_customer_claim_texts(
                "What causes poor airflow?", _presentation()
            )
        self.assertEqual(result, {"c1": _good_text()})
        self.assertFalse(session.trust_env)
        args, kwargs = session.post.call_args
        self.assertEqual(args, ("http://127.0.0.1:11434/api/chat",))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["json"]["model"], "qwen3.5:2b-q4_K_M")
        self.assertEqual(kwargs["json"]["options"]["num_ctx"], 4096)
        self.assertEqual(kwargs["json"]["options"]["num_predict"], 384)
        self.assertIn("A blocked filter may reduce airflow.", kwargs["json"]["messages"][1]["content"])
        self.assertNotIn("customer_id", kwargs["json"]["messages"][1]["content"])
        session.close.assert_called_once_with()

        # The serving path exposes the same outcome to its caller, rather than
        # silently collapsing all failures into an extractive engine label.
        with (
            patch.object(generation, "_server_online", return_value=True),
            patch("core.services.customer_answer_generation.requests.Session", return_value=session),
        ):
            result_with_status = generation.generate_customer_claim_texts_with_status(
                "What causes poor airflow?", _presentation()
            )
        self.assertEqual(result_with_status["texts"], {"c1": _good_text()})
        self.assertEqual(result_with_status["generation"]["status"], "applied")
        self.assertTrue(result_with_status["generation"]["attempted"])
        self.assertEqual(result_with_status["generation"]["eligible_claims"], 1)
        self.assertEqual(result_with_status["generation"]["accepted_claims"], 1)

    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_URL="https://example.com/api/chat",
    )
    def test_remote_endpoint_is_rejected_without_network_call(self):
        with patch("core.services.customer_answer_generation.requests.Session") as session:
            self.assertEqual(generation.generate_customer_claim_texts(
                "Question", _presentation()
            ), {})
        session.assert_not_called()

    @override_settings(SERVY_CUSTOMER_ANSWER_GENERATION="disabled")
    def test_disabled_provider_does_not_probe_ollama(self):
        with patch.object(generation, "_server_online",
                          side_effect=AssertionError("generation must be disabled")):
            self.assertEqual(generation.generate_customer_claim_texts(
                "Question", _presentation()
            ), {})

    @override_settings(SERVY_CUSTOMER_ANSWER_GENERATION="ollama")
    def test_offline_model_preserves_source_wording(self):
        presentation = _presentation()
        before = json.dumps(presentation, sort_keys=True)
        with patch.object(generation, "_server_online", return_value=False):
            status = generation.generate_customer_claim_texts_with_status(
                "What causes poor airflow?", presentation
            )
            self.assertEqual(status["generation"]["status"], "ollama_unreachable")
            self.assertFalse(status["generation"]["attempted"])
            self.assertFalse(generation.apply_customer_answer_generation(
                presentation, "What causes poor airflow?"
            ))
        self.assertEqual(json.dumps(presentation, sort_keys=True), before)

    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_MAX_CLAIMS="not-an-integer",
    )
    def test_malformed_limit_fails_closed_without_breaking_answer(self):
        with patch.object(generation, "_server_online",
                          side_effect=AssertionError("malformed settings must fail before probing")):
            self.assertEqual(
                generation.generate_customer_claim_texts("Question", _presentation()), {}
            )

    @override_settings(
        SERVY_CUSTOMER_ANSWER_GENERATION="ollama",
        SERVY_CUSTOMER_GENERATION_URL="http://127.0.0.1:11434/api/chat",
    )
    def test_invalid_model_output_is_rejected(self):
        presentation = _presentation()
        with patch.object(generation, "generate_customer_claim_texts", return_value={}):
            self.assertFalse(generation.apply_customer_answer_generation(
                presentation, "What causes poor airflow?"
            ))
        self.assertEqual(
            presentation["sections"][0]["claims"][0]["text"],
            "Possible documented cause: A blocked filter may reduce airflow.",
        )


class EvidenceGroundedParaphraseTests(TestCase):
    """Legitimate paraphrases pass; unsupported or altered claims are still rejected."""

    def accepted(self, source, text, kind="information"):
        return generation._validate_paraphrase(source, generation._compose_display(text, kind), kind)

    def test_possible_cause_is_hedged_by_the_server_not_rejected(self):
        display = generation._compose_display("Dirty filter or coil.", "possible_cause")
        self.assertTrue(display.startswith("Possible cause: "))
        self.assertTrue(self.accepted("Dirty filter or coil.", "Dirty filter or coil.", "possible_cause"))
        self.assertTrue(self.accepted("Dirty filter or coil.", "A dirty coil or filter.", "possible_cause"))

    def test_reordering_and_inflection_of_source_terms_are_accepted(self):
        self.assertTrue(self.accepted("Quarterly: clean filters", "Clean the filters quarterly."))
        self.assertTrue(self.accepted("Blocked or incorrectly routed condensate drain",
                                      "A condensate drain that is blocked or incorrectly routed.",
                                      "possible_cause"))

    def test_unsupported_additions_are_rejected(self):
        source = "Dirty filter or coil."
        for text in ("Dirty filter, coil or low refrigerant charge.",      # invented cause
                     "Dirty filter or coil after 3 months of use.",        # invented number
                     "The cause is a dirty filter or coil.",               # certainty
                     "Dirty filter or coil; replace the compressor."):     # invented action
            with self.subTest(text=text):
                self.assertFalse(self.accepted(source, text, "possible_cause"))

    def test_source_terms_cannot_be_dropped_or_negated(self):
        self.assertFalse(self.accepted("Dirty filter or coil.", "A dirty filter.", "possible_cause"))
        self.assertFalse(self.accepted("Quarterly: clean filters", "Do not clean the filters quarterly."))

    def test_model_cannot_create_or_address_instructions(self):
        # A statement may not become an instruction ...
        self.assertFalse(self.accepted("Filters are cleaned quarterly by technicians",
                                       "Check the filters, which are cleaned quarterly by technicians."))
        # ... and the customer may never be addressed directly, even for a source imperative.
        self.assertFalse(self.accepted("Quarterly: clean filters", "You should clean the filters quarterly."))

    def test_sequences_keep_their_order(self):
        source = "Isolate power before opening the access panel"
        self.assertFalse(self.accepted(source, "Open the access panel before isolating the power."))

    def test_spelled_numbers_match_but_changed_numbers_do_not(self):
        source = "Every 6 months: document airflow and static pressure"
        self.assertTrue(self.accepted(source, "Every six months, document airflow and static pressure."))
        self.assertFalse(self.accepted(source, "Every 3 months, document airflow and static pressure."))
        self.assertFalse(self.accepted(source, "Every twelve months, document airflow and static pressure."))
