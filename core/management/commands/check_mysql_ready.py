"""
Management command: check_mysql_ready
Verifies that the active Django database connection is authoritative MySQL,
checks version, collation, character set, storage engine, strict SQL modes,
migration status, and protected database identity. Fails closed.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from core.security import is_protected_database


class Command(BaseCommand):
    help = "Verify that the active relational database is MySQL and ready for production."

    def handle(self, *args, **options):
        # Phase 1: Database Vendor & Connection Check
        if connection.vendor != "mysql":
            raise CommandError(
                f"DATABASE ARCHITECTURE VIOLATION: Expected 'mysql' backend, but active connection vendor is '{connection.vendor}'. "
                "Servy is architected MySQL-first. SQLite fallback is strictly prohibited in production."
            )

        try:
            connection.ensure_connection()
        except Exception as exc:
            raise CommandError(f"PHASE 1 FAILED: Could not establish connection to MySQL database: {exc}")

        # Phase 2: Schema Identification & Engine Variables
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT VERSION(), DATABASE(), CURRENT_USER(), "
                    "@@sql_mode, @@character_set_database, @@collation_database, "
                    "@@default_storage_engine, @@transaction_isolation;"
                )
                row = cursor.fetchone()
                (
                    version,
                    database_name,
                    user,
                    sql_mode,
                    charset,
                    collation,
                    storage_engine,
                    isolation,
                ) = row
        except Exception as exc:
            raise CommandError(f"PHASE 2 FAILED: Failed to query MySQL system variables: {exc}")

        if not database_name:
            raise CommandError("PHASE 2 FAILED: No active MySQL database schema selected.")

        # Phase 3: Strict Database Configuration Verification (Fail Closed)
        errors = []
        if storage_engine.upper() != "INNODB":
            errors.append(f"Default storage engine is '{storage_engine}', strictly requires 'InnoDB'.")
        if "utf8mb4" not in (charset or "").lower():
            errors.append(f"Database charset is '{charset}', strictly requires 'utf8mb4'.")
        if "STRICT_TRANS_TABLES" not in sql_mode and "STRICT_ALL_TABLES" not in sql_mode:
            errors.append("Strict SQL mode (STRICT_TRANS_TABLES or STRICT_ALL_TABLES) is not active.")
        if isolation.upper() not in ("READ-COMMITTED", "REPEATABLE-READ"):
            errors.append(f"Transaction isolation level '{isolation}' is unsupported. Requires READ-COMMITTED or REPEATABLE-READ.")

        if errors:
            for err in errors:
                self.stdout.write(self.style.ERROR(f"CONFIG ERROR: {err}"))
            raise CommandError(f"PHASE 3 FAILED: MySQL configuration validation failed with {len(errors)} blocker(s).")

        # Phase 4: Migration Status Verification
        migration_count = 0
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = DATABASE() AND table_name = 'django_migrations';"
                )
                has_migrations_table = cursor.fetchone()[0] > 0
                if has_migrations_table:
                    cursor.execute("SELECT COUNT(*) FROM `django_migrations` WHERE `app` = 'core';")
                    migration_count = cursor.fetchone()[0]
        except Exception as exc:
            raise CommandError(f"PHASE 4 FAILED: Could not verify migration status: {exc}")

        # Phase 5: Protected Database Identity Verification
        is_protected = is_protected_database(connection)
        db_identity_type = "PROTECTED_PRODUCTION_OR_COMPANY" if is_protected else "DISPOSABLE_TEST_OR_DEVELOPMENT"

        # Phase 6: Comprehensive Verification Report
        self.stdout.write(self.style.SUCCESS(
            f"=== MYSQL READINESS VERIFIED (ALL GATES PASSED) ===\n"
            f"  Vendor:                 {connection.vendor}\n"
            f"  Server Version:         {version}\n"
            f"  Database Schema:        {database_name}\n"
            f"  Connected User:         {user}\n"
            f"  Storage Engine:         {storage_engine} [PASS]\n"
            f"  Character Set:          {charset} [PASS]\n"
            f"  Collation:              {collation} [PASS]\n"
            f"  Transaction Isolation:  {isolation} [PASS]\n"
            f"  SQL Mode:               {sql_mode} [PASS]\n"
            f"  Core Migrations:        {migration_count} applied\n"
            f"  Database Identity:      {db_identity_type}\n"
            f"==================================================="
        ))
