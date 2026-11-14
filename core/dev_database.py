"""Startup guard for the isolated local development database.

Writable local demo services must run only against servy_dev with the
servy_dev_app account. This inspects the live connection itself (not the
configuration) and refuses to continue on any mismatch. It never reads or
prints credentials.
"""
import re

from django.db import connection

PROTECTED_DATABASES = ("servy", "test_servy")


class DevDatabaseIsolationError(RuntimeError):
    """The connected database or account is not the isolated development one."""


def verify_isolated_database(expected_database="servy_dev", expected_user="servy_dev_app"):
    """Return a summary dict, or raise DevDatabaseIsolationError."""
    if connection.vendor != "mysql":
        raise DevDatabaseIsolationError("The development services require MySQL.")
    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE(), CURRENT_USER()")
        database, current_user = cursor.fetchone()
        if database != expected_database:
            raise DevDatabaseIsolationError(f"Connected database is {database!r}, expected {expected_database!r}.")
        if (current_user or "").split("@", 1)[0] != expected_user:
            raise DevDatabaseIsolationError(f"Connected account is not {expected_user!r}.")
        cursor.execute("SHOW GRANTS")
        grants = [row[0] for row in cursor.fetchall()]
        for grant in grants:
            if "WITH GRANT OPTION" in grant:
                raise DevDatabaseIsolationError("The development account has GRANT OPTION.")
            if " ON *.* " in grant and not grant.startswith("GRANT USAGE ON *.*"):
                raise DevDatabaseIsolationError("The development account has global privileges.")
            match = re.search(r" ON `?([^`.]+)`?\.", grant)
            if match and match.group(1) != "*" and match.group(1).replace("\\_", "_") != expected_database:
                raise DevDatabaseIsolationError("The development account has privileges on another database.")
        for protected in PROTECTED_DATABASES:
            try:
                cursor.execute(f"SELECT 1 FROM `{protected}`.`django_migrations` LIMIT 1")
            except Exception:
                continue
            raise DevDatabaseIsolationError(f"The development account can read {protected!r}.")
    return {"database": database, "user": expected_user, "grants": len(grants)}
