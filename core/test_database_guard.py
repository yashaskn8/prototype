"""Refuse to run database-backed Django tests against a non-test database.

Django's test runner (``manage.py test``) always switches connections to a throwaway
``test_*`` database. Running Django ``TestCase`` classes any other way (for example a
plain ``python -m unittest`` or a script that imports test classes) uses the configured
database directly. That is how test-fixture tenants were once written into the company
MySQL database. This guard runs before any test class touches a database and stops the
class if the connection is not a test database.
"""
from django.core.exceptions import ImproperlyConfigured

_installed = False


def is_test_database(settings_dict, vendor):
    name = str(settings_dict.get("NAME") or "")
    if vendor == "sqlite":
        # manage.py test uses an in-memory database; never a file such as db.sqlite3.
        return name in ("", ":memory:") or "mode=memory" in name or name.startswith("file:memorydb")
    test_name = (settings_dict.get("TEST") or {}).get("NAME")
    return name.startswith("test_") or (bool(test_name) and name == test_name)


def check_test_databases(test_class):
    from django.db import connections

    databases = getattr(test_class, "databases", None) or set()
    if not databases:
        return
    aliases = connections if databases == "__all__" else databases
    for alias in aliases:
        connection = connections[alias]
        if not is_test_database(connection.settings_dict, connection.vendor):
            raise ImproperlyConfigured(
                f"Refusing to run {test_class.__module__}.{test_class.__qualname__} against non-test "
                f"database {connection.settings_dict.get('NAME')!r} (alias {alias!r}). "
                "Run database tests with 'python manage.py test', which uses a throwaway test database."
            )


def install():
    """Wrap SimpleTestCase.setUpClass, which TestCase/TransactionTestCase call before any DB work."""
    global _installed
    if _installed:
        return
    from django.test import SimpleTestCase

    original = SimpleTestCase.__dict__["setUpClass"].__func__

    def setUpClass(cls):
        check_test_databases(cls)
        original(cls)

    SimpleTestCase.setUpClass = classmethod(setUpClass)
    _installed = True
