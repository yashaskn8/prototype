"""Verify the live connection is the isolated servy_dev development database."""
from django.core.management.base import BaseCommand, CommandError

from core.dev_database import DevDatabaseIsolationError, verify_isolated_database


class Command(BaseCommand):
    help = "Refuse unless connected to servy_dev as servy_dev_app with database-only grants."

    def add_arguments(self, parser):
        parser.add_argument("--database", default="servy_dev")
        parser.add_argument("--user", default="servy_dev_app")

    def handle(self, *args, **options):
        try:
            summary = verify_isolated_database(options["database"], options["user"])
        except DevDatabaseIsolationError as exc:
            raise CommandError(f"Development database isolation failed: {exc}") from exc
        self.stdout.write(self.style.SUCCESS(
            f"Isolated development database verified: {summary['database']} as {summary['user']}; "
            "no global privileges, no GRANT OPTION, production servy unreadable."))
