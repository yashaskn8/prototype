"""Django management command to preview and execute Starlly real company data ingestion."""

from django.core.management.base import BaseCommand
from core.services.starlly_ingest import preview_starlly_ingestion, ingest_starlly_real_data


class Command(BaseCommand):
    help = "Ingest authentic assets and documents from Starlly into the dedicated real company tenant."

    def add_arguments(self, parser):
        parser.add_argument(
            "--preview",
            action="store_true",
            help="Perform read-only preview of available Starlly records without making any database changes.",
        )
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Execute the zero-deletion real company data ingestion.",
        )

    def handle(self, *args, **options):
        if not options["execute"]:
            # Default to preview mode for absolute safety
            self.stdout.write(self.style.NOTICE("Running Starlly Ingestion Preview (READ-ONLY)..."))
            preview = preview_starlly_ingestion()
            self.stdout.write(self.style.SUCCESS(f"Preview Status: {preview['status']}"))
            self.stdout.write(f"Target Tenant: {preview['target_tenant']}")
            self.stdout.write(f"Starlly Org ID: {preview['starlly_org_id']}")
            self.stdout.write("Discovered Counts:")
            for k, v in preview["counts"].items():
                self.stdout.write(f"  - {k}: {v}")
            self.stdout.write("\nSample Assets:")
            for a in preview["sample_assets"]:
                self.stdout.write(f"  * {a}")
            self.stdout.write(f"\nAttachments Found: {len(preview['attachments_preview'])}")
            for att in preview["attachments_preview"]:
                self.stdout.write(f"  * [{att['asset_code']}] {att['filename']}")
            self.stdout.write(self.style.WARNING("\nTo execute actual ingestion, pass --execute flag."))
            return

        self.stdout.write(self.style.NOTICE("Executing Starlly Real Company Data Ingestion..."))
        report = ingest_starlly_real_data()
        self.stdout.write(self.style.SUCCESS("Ingestion completed successfully:"))
        for k, v in report.items():
            self.stdout.write(f"  - {k}: {v}")
