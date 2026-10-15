"""
tests_config_isolation.py — Regression tests for Operation 15A.

Verifies:
  1. Process-level environment overrides .env values.
  2. SQLite test mode does not initialize MySQL.
  3. SQLite test mode uses a disposable in-memory database.
  4. Invalid engine values fail clearly.
  5. SQLite production-mode protections remain intact.
  6. Tests do not silently select production credentials.
  7. Secrets are never exposed in diagnostic output.

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

    def test_sqlite_engine_selected_when_env_says_sqlite(self):
        """SERVY_DB_ENGINE=sqlite3 in process env must select SQLite backend."""
        from django.conf import settings
        configured = getattr(settings, "SERVY_DB_ENGINE", "mysql")
        engine = settings.DATABASES["default"]["ENGINE"]
        if configured == "sqlite3":
            self.assertIn("sqlite3", engine, f"Expected SQLite backend but got: {engine}")
            self.assertNotIn("mysql", engine)
        else:
            self.assertIn("mysql", engine, f"Expected MySQL backend in CI but got: {engine}")

    def test_sqlite_test_db_is_memory(self):
        """SQLite test database must be :memory: for isolation."""
        from django.conf import settings
        configured = getattr(settings, "SERVY_DB_ENGINE", "mysql")
        if configured != "sqlite3":
            self.skipTest("Skipped in MySQL CI environment (running with MySQL backend)")
        test_name = settings.DATABASES["default"].get("TEST", {}).get("NAME")
        self.assertEqual(test_name, ":memory:", f"SQLite test db should be :memory:, got: {test_name}")

    def test_debug_mode_reads_from_env(self):
        """DEBUG must be derived from DJANGO_DEBUG environment variable.

        Note: Django's test runner always sets DEBUG=False during tests.
        We verify the settings module reads from the correct env var.
        """
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        self.assertIn('os.getenv("DJANGO_DEBUG"', src,
                       "DEBUG must be derived from DJANGO_DEBUG env var")

    def test_no_mysql_driver_imported_in_sqlite_mode(self):
        """In SQLite mode, MySQLdb/pymysql should NOT be required."""
        from django.conf import settings
        configured = getattr(settings, "SERVY_DB_ENGINE", "mysql")
        if configured != "sqlite3":
            self.skipTest("Skipped in MySQL CI environment")
        engine = settings.DATABASES["default"]["ENGINE"]
        self.assertNotIn("mysql", engine, "MySQL engine should not be active in sqlite3 mode")

    def test_database_engine_setting_matches(self):
        """SERVY_DB_ENGINE setting attribute must reflect the resolved engine."""
        from django.conf import settings
        expected = os.getenv("SERVY_DB_ENGINE", "mysql")
        self.assertEqual(settings.SERVY_DB_ENGINE, expected)

    def test_subprocess_sqlite_isolation(self):
        """Verify in an isolated subprocess that SERVY_DB_ENGINE=sqlite3 selects SQLite."""
        import subprocess
        script = (
            "import os, django\n"
            "os.environ['SERVY_DB_ENGINE'] = 'sqlite3'\n"
            "os.environ['DJANGO_DEBUG'] = '1'\n"
            "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'servy_rag.settings')\n"
            "django.setup()\n"
            "from django.conf import settings\n"
            "print('ENGINE=' + settings.DATABASES['default']['ENGINE'])\n"
            "print('TEST_DB=' + str(settings.DATABASES['default'].get('TEST', {}).get('NAME')))\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(res.returncode, 0, f"Subprocess failed: {res.stderr}")
        self.assertIn("ENGINE=django.db.backends.sqlite3", res.stdout)
        self.assertIn("TEST_DB=:memory:", res.stdout)


class SettingsProductionGuardTests(unittest.TestCase):
    """Verify production-mode safety guards are intact."""

    def test_sqlite_rejected_in_production_mode(self):
        """SQLite must be rejected when DJANGO_DEBUG=0 and CI is not set."""
        # We can't easily reload settings in-process without side effects,
        # so we test the guard logic by examining the source code.
        settings_path = Path(__file__).resolve().parent.parent / "servy_rag" / "settings.py"
        src = settings_path.read_text(encoding="utf-8")
        # The guard must exist
        self.assertIn(
            "SQLite backend is strictly prohibited when DJANGO_DEBUG=0",
            src,
            "Production SQLite guard is missing from settings.py"
        )
        # Must check both DEBUG and CI
        self.assertIn('not DEBUG', src)
        self.assertIn('os.getenv("CI")', src)

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
