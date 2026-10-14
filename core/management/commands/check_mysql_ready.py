"""
Management command: check_mysql_ready
Verifies that the active Django database connection is authoritative MySQL,
checks version, collation, character set, storage engine, and strict SQL modes.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import connection


class Command(BaseCommand):
    help = "Verify that the active relational database is MySQL and ready for production."

    def handle(self, *args, **options):
        if connection.vendor != "mysql":
            raise CommandError(
                f"DATABASE ARCHITECTURE VIOLATION: Expected 'mysql' backend, but active connection vendor is '{connection.vendor}'. "
                "Servy is architected MySQL-first. SQLite fallback is strictly prohibited in production."
            )

        try:
            connection.ensure_connection()
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

            self.stdout.write(self.style.SUCCESS(
                f"=== MYSQL READINESS VERIFIED ===\n"
                f"  Vendor:                 {connection.vendor}\n"
                f"  Server Version:         {version}\n"
                f"  Database Schema:        {database_name}\n"
                f"  Connected User:         {user}\n"
                f"  Storage Engine:         {storage_engine}\n"
                f"  Character Set:          {charset}\n"
                f"  Collation:              {collation}\n"
                f"  Transaction Isolation:  {isolation}\n"
                f"  SQL Mode:               {sql_mode}\n"
                f"================================="
            ))

            # Validate strict data guarantees
            warnings = []
            if storage_engine.upper() != "INNODB":
                warnings.append(f"Storage engine is '{storage_engine}', expected 'InnoDB'.")
            if "utf8mb4" not in charset.lower():
                warnings.append(f"Database charset is '{charset}', expected 'utf8mb4'.")
            if "STRICT_TRANS_TABLES" not in sql_mode and "STRICT_ALL_TABLES" not in sql_mode:
                warnings.append("Strict SQL mode (STRICT_TRANS_TABLES) is not active.")

            for w in warnings:
                self.stdout.write(self.style.WARNING(f"WARNING: {w}"))

        except Exception as exc:
            raise CommandError(f"Failed to connect to MySQL database: {exc}")
