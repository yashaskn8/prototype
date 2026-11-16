"""Regression tests for the real-data demo's local-Qwen opt-in and fallback."""
from pathlib import Path
import re
import unittest


class RealDemoLauncherConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = Path(__file__).with_name("Start-ServyRealDemo.ps1").read_text(
            encoding="utf-8"
        )
        # Only inspect commands that can write to the visible PowerShell console.
        cls.visible_output = "\n".join(
            line for line in cls.launcher.splitlines()
            if re.search(r"\\b(?:Write-Output|Write-Host|Write-Warning|Write-Information)\\b", line)
        )

    def test_retrieval_and_dialogue_remain_deterministic_while_qwen_is_enabled(self):
        # The customer demo is the official Django gateway on servy_dev.
        self.assertIn("$env:SERVY_DEMO_GATEWAY = 'customer'", self.launcher)
        self.assertIn("$env:SERVY_ALLOW_REMOTE_LLM = '0'", self.launcher)
        self.assertIn("$env:SERVY_LLM_PROVIDER = 'extractive'", self.launcher)
        self.assertNotIn("$env:SERVY_LLM_PROVIDER = 'ollama'", self.launcher)
        self.assertIn("$env:SERVY_CUSTOMER_ANSWER_GENERATION = 'ollama'", self.launcher)
        self.assertIn("$env:SERVY_CUSTOMER_GENERATION_MODEL = 'qwen3.5:2b-q4_K_M'", self.launcher)
        self.assertIn("Qwen writes customer-facing wording only from reauthorized cited source chunks", self.launcher)

        # Internal readiness/configuration messages must not be printed to the terminal.
        for hidden in (
            "Customer RAG answer generation: local Qwen enabled",
            "Qwen local generation: enabled.",
            "Qwen model check: PASS.",
            "servy_dev isolation check: PASS",
        ):
            with self.subTest(hidden=hidden):
                self.assertNotIn(hidden, self.visible_output)

    def test_qwen_is_loopback_only_and_model_failure_keeps_exact_source_wording(self):
        for required in (
            "http://127.0.0.1:11434/api/chat",
            "http://127.0.0.1:11434/api/tags",
            "Qwen is enabled but local Ollama is not reachable",
            "Qwen is enabled but qwen3.5:2b-q4_K_M is not installed",
            "exact cited source wording",
            "SERVY_CUSTOMER_GENERATION_MODEL = 'qwen3.5:2b-q4_K_M'",
            "SERVY_CUSTOMER_GENERATION_TIMEOUT = '45'",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.launcher)

        # The previous visible status line was intentionally removed; validation
        # and fallback behavior are exercised by the dedicated generation tests.
        self.assertNotIn(
            "cited-source validation and extractive fallback retained",
            self.visible_output,
        )

    def test_no_local_chat_switch_disables_generation_and_uses_extractive_fallback(self):
        self.assertIn("[switch]$NoLocalChat", self.launcher)
        self.assertIn("if ($NoLocalChat)", self.launcher)
        self.assertIn("$env:SERVY_CUSTOMER_ANSWER_GENERATION = 'disabled'", self.launcher)
        self.assertIn("$env:SERVY_LLM_PROVIDER = 'extractive'", self.launcher)

    def test_launcher_does_not_configure_remote_generation(self):
        self.assertIn("$env:SERVY_ALLOW_REMOTE_LLM = '0'", self.launcher)
        self.assertNotIn("https://", self.launcher)


if __name__ == "__main__":
    unittest.main()
