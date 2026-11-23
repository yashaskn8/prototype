import re
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parent / "Start-ServyRealDemo.ps1"


class LauncherHealthPollingTests(unittest.TestCase):
    def setUp(self):
        self.text = LAUNCHER.read_text(encoding="utf-8-sig")

    def test_single_bounded_five_second_poll_that_stops_when_ready(self):
        loops = re.findall(r"for \(\$attempt = 0; \$attempt -lt (\d+) -and !\$modelsReady; \$attempt\+\+\) \{\s*Start-Sleep -Seconds (\d+)", self.text)
        self.assertEqual(len(loops), 1, "exactly one readiness polling loop")
        attempts, seconds = map(int, loops[0])
        self.assertEqual(seconds, 5)
        self.assertLessEqual(attempts * seconds, 150)
        self.assertEqual(self.text.count("Invoke-WebRequest -Uri 'http://127.0.0.1:5175/demo-api/health/'"), 2,
                         "initial probe plus the loop only")

    def test_viewer_shows_a_single_health_line(self):
        self.assertIn("$healthShown = $false", self.text)
        self.assertIn("if (-not $healthShown) { $healthShown = $true; $_ }", self.text)


if __name__ == "__main__":
    unittest.main()
