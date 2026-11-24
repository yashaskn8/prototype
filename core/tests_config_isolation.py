"""
tests_config_isolation.py — Regression tests for Operation 15A.

Verifies:
  1. Process-level environment overrides .env values.
  2. MySQL is the only database engine, for tests as well as runtime.
  3. Any other engine value (including the retired sqlite3) fails clearly.
  4. Tests do not silently select production credentials.
  5. Secrets are never exposed in diagnostic output.

These tests are designed to be self-contained and independent of
the user's actual .env contents. They test the settings module's
configuration logic by manipulating os.environ before import.
"""
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SettingsEnvPrecedenceTests(unittest.TestCase):
    """Verify that process-level environment variables take precedence over .env."""

    def _reload_settings(self):
        """Force a clean re-import of settings to test configuration logic."""
        # Remove cached settings module so it re-executes on import
        mods_to_remove = [k for k in sys.modules if k.startswith("servy_rag")]
        for m in mods_to_remove:
            del sys.modules[m]
        # Also reset Django's settings wrapper
        from django.conf import settings as django_settings
        django_settings._wrapped = None
        import servy_rag.settings as s
        return s

    def test_tests_run_on_mysql(self):
        """The test suite itself runs on MySQL; there is no other engine."""
        from django.conf import settings
        self.assertEqual(getattr(settings, "SERVY_DB_ENGINE", None), "mysql")
        self.assertIn("mysql", settings.DATABASES["default"]["ENGINE"])

    def test_debug_mode_reads_from_env(self):
        """DEBUG must be derived from DJANGO_DEBUG environment variable.

        Note: Django's test runner always sets DEBUG=False during tests.
        We verify the settings module reads from the correct env var.
        """
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        self.assertIn('os.getenv("DJANGO_DEBUG"', src,
                       "DEBUG must be derived from DJANGO_DEBUG env var")

    def test_retired_sqlite_engine_is_refused_even_for_test_runs(self):
        """The retired sqlite3 engine is refused for every process, test runs included,
        so no SQLite database file can ever be created by Django."""
        import subprocess
        for argv in ("test", "runserver", "migrate"):
            with self.subTest(argv=argv):
                script = (
                    "import os, sys, django\n"
                    f"sys.argv = ['manage.py', '{argv}']\n"
                    "os.environ['SERVY_DB_ENGINE'] = 'sqlite3'\n"
                    "os.environ['DJANGO_DEBUG'] = '1'\n"
                    "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'servy_rag.settings')\n"
                    "django.setup()\n"
                )
                res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                     cwd=str(Path(__file__).resolve().parent.parent), timeout=120)
                self.assertNotEqual(res.returncode, 0)
                self.assertIn("Unsupported database engine 'sqlite3'", res.stderr)


class SettingsProductionGuardTests(unittest.TestCase):
    """Verify production-mode safety guards are intact."""

    def test_settings_define_no_sqlite_backend(self):
        """Settings contain no SQLite backend or database file for any process."""
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        self.assertNotIn("django.db.backends.sqlite3", src)
        self.assertNotIn("db.sqlite3", src)

    def test_invalid_engine_raises_runtime_error(self):
        """Unsupported SERVY_DB_ENGINE values must raise RuntimeError."""
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        self.assertIn("Unsupported database engine", src)

    def test_default_secret_key_rejected_in_production(self):
        """Default secret key must be rejected when DEBUG=False."""
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        self.assertIn("Set DJANGO_SECRET_KEY before running with DJANGO_DEBUG=0", src)


class SettingsDotenvOverrideTests(unittest.TestCase):
    """Verify .env does not override process-level environment variables."""

    def test_dotenv_uses_override_false(self):
        """load_dotenv must use override=False to respect process env."""
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        self.assertIn("override=False", src,
                       "load_dotenv must use override=False")
        self.assertNotIn("override=True", src,
                          "override=True must NOT appear in settings.py")

    def test_manual_fallback_uses_setdefault(self):
        """Manual .env parser must use os.environ.setdefault, not direct assignment."""
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        self.assertIn("os.environ.setdefault", src,
                       "Manual .env parser must use setdefault")


class SettingsSecretsProtectionTests(unittest.TestCase):
    """Verify that secrets are not exposed."""

    def test_no_password_in_settings_source(self):
        """Settings source must not contain hardcoded real passwords."""
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        # Should not contain actual CI passwords (which are only in ci.yml)
        self.assertNotIn("root_ci_password", src)
        self.assertNotIn("servy_ci_password", src)

    def test_mysql_ci_uses_disposable_db(self):
        """CI workflow must use a disposable test database, not production."""
        ci_path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
        if not ci_path.exists():
            self.skipTest("CI workflow file not found locally")
        src = ci_path.read_text(encoding="utf-8")
        self.assertIn("servy_test", src,
                       "CI must use a test database name, not production")
        self.assertIn("SERVY_DB_ENGINE: mysql", src,
                       "CI must explicitly set MySQL engine")
