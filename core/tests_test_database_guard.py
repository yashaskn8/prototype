"""Database tests must never run against the configured (company) database."""
import os
import subprocess
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from django.db import connection
from django.test import SimpleTestCase, TestCase

from core.test_database_guard import check_test_databases, is_test_database

ROOT = Path(__file__).resolve().parent.parent


class TestDatabaseRecognitionTests(SimpleTestCase):
    def test_company_and_file_databases_are_not_test_databases(self):
        self.assertFalse(is_test_database({"NAME": "servy"}, "mysql"))
        self.assertFalse(is_test_database({"NAME": "servy_test"}, "mysql"))
        self.assertFalse(is_test_database({"NAME": str(ROOT / "db.sqlite3")}, "sqlite"))

    def test_throwaway_test_databases_are_recognised(self):
        self.assertTrue(is_test_database({"NAME": "test_servy"}, "mysql"))
        self.assertTrue(is_test_database({"NAME": "test_servy_test"}, "mysql"))
        self.assertTrue(is_test_database({"NAME": "file:memorydb_default?mode=memory&cache=shared"}, "sqlite"))
        self.assertTrue(is_test_database({"NAME": ":memory:"}, "sqlite"))

    def test_classes_without_databases_are_not_checked(self):
        class NoDatabase(SimpleTestCase):
            databases = set()
        check_test_databases(NoDatabase)


class GuardDuringTestRunTests(TestCase):
    def test_this_suite_is_running_on_a_test_database(self):
        self.assertTrue(is_test_database(connection.settings_dict, connection.vendor))


class GuardOutsideTestRunnerTests(SimpleTestCase):
    def test_testcase_outside_the_test_runner_is_refused_before_any_database_work(self):
        # Plain unittest (not manage.py test) would use the configured database itself.
        # The class performs no writes; the guard must stop it before setUpClass opens
        # a transaction.
        code = (
            "import os, unittest, django\n"
            "django.setup()\n"
            "from django.test import TestCase\n"
            "class Probe(TestCase):\n"
            "    def test_probe(self):\n"
            "        pass\n"
            "result = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Probe))\n"
            "print('ERRORS', len(result.errors))\n"
            "print('REFUSED', any('Refusing to run' in tb for _, tb in result.errors))\n"
        )
        env = dict(os.environ, DJANGO_SETTINGS_MODULE="servy_rag.settings", SERVY_DB_ENGINE="sqlite3", DJANGO_DEBUG="1")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        self.assertIn("ERRORS 1", out.stdout, out.stderr[-800:])
        self.assertIn("REFUSED True", out.stdout, out.stderr[-800:])

    def test_refusal_names_the_database_and_the_supported_command(self):
        class Probe(SimpleTestCase):
            databases = {"default"}
        with self.settings():
            original = dict(connection.settings_dict)
            try:
                connection.settings_dict["NAME"] = "servy"
                vendor = connection.vendor
                if vendor != "mysql":
                    connection.settings_dict["NAME"] = str(ROOT / "db.sqlite3")
                with self.assertRaisesRegex(ImproperlyConfigured, "Refusing to run .*manage.py test"):
                    check_test_databases(Probe)
            finally:
                connection.settings_dict.clear()
                connection.settings_dict.update(original)
