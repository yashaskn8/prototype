"""Regression tests for the real-data demo's no-Ollama startup contract."""
from pathlib import Path
import unittest


class RealDemoLauncherConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = Path(__file__).with_name("Start-ServyRealDemo.ps1").read_text(
            encoding="utf-8"
        )

    def test_real_demo_uses_extractive_answers_and_deterministic_dialogue(self):
        self.assertIn("$env:SERVY_LLM_PROVIDER = 'extractive'", self.launcher)
        self.assertIn("$env:SERVY_REAL_DEMO_DIALOGUE_MODE = 'rules'", self.launcher)
        self.assertIn("Dialogue mode: deterministic and source-grounded", self.launcher)

    def test_real_demo_does_not_probe_or_require_ollama(self):
        for forbidden in ("11434", "/api/tags", "qwen2.5:3b", "Invoke-RestMethod"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.launcher)

    def test_legacy_no_local_chat_switch_remains_accepted(self):
        self.assertIn("[switch]$NoLocalChat", self.launcher)


if __name__ == "__main__":
    unittest.main()
