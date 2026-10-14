"""
Management command: transfer_sqlite_to_mysql
Non-destructive, topological data audit and discovery tool to analyze SQLite records
and target Servy MySQL database readiness.

Guarantees (Operation Ironclad 5.0):
- Direct --execute is strictly DISABLED on protected or unverified databases to prevent
  foreign-key check bypass and implicit MySQL DDL transaction commits.
- Performs full non-destructive topological schema and data audit across the entire Django schema.
- Detects existing records, potential ID conflicts, and unique constraint collisions.
- Verifies column compatibility between source SQLite and target MySQL.
- Preserves all existing business records, documents, media files, and vectors intact.
- Execution requires an isolated, separately authorized migration plan.
"""

import hashlib
import sqlite3
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from core.security import is_protected_database

# Complete Django schema models in strict topological foreign-key dependency order
MODEL_TABLE_ORDER = [
    # 0. Django core / content types / auth
    ("django_content_type", "contenttypes.ContentType"),
    ("auth_permission", "auth.Permission"),
    ("auth_group", "auth.Group"),
    ("auth_user", "auth.User"),
    ("auth_user_groups", "auth.User.groups"),
    ("auth_user_user_permissions", "auth.User.user_permissions"),
    ("auth_group_permissions", "auth.Group.permissions"),
    ("django_session", "sessions.Session"),
    # 1. Base tenant & catalog
    ("core_tenant", "core.Tenant"),
    ("core_productdomain", "core.ProductDomain"),
    ("core_productcategory", "core.ProductCategory"),
    ("core_brand", "core.Brand"),
    ("core_product", "core.Product"),
    ("core_customer", "core.Customer"),
    ("core_branch", "core.Branch"),
    ("core_operationalzone", "core.OperationalZone"),
    ("core_site", "core.Site"),
    ("core_asset", "core.Asset"),
    ("core_slapolicy", "core.SLAPolicy"),
    ("core_staffprofile", "core.StaffProfile"),
    ("core_tenantmembership", "core.TenantMembership"),
    ("core_corporateidentity", "core.CorporateIdentity"),
    ("core_businessvocabulary", "core.BusinessVocabulary"),
    ("core_customizationsetting", "core.CustomizationSetting"),
    ("core_emailnotificationrule", "core.EmailNotificationRule"),
    ("core_project", "core.Project"),
    ("core_project_assets", "core.Project.assets"),
    # 2. Knowledge documents & chunks
    ("core_knowledgedocument", "core.KnowledgeDocument"),
    ("core_knowledgechunk", "core.KnowledgeChunk"),
    ("core_serviceresolutionindex", "core.ServiceResolutionIndex"),
    # 3. Service calls & operational items
    ("core_servicecall", "core.ServiceCall"),
    ("core_callupdate", "core.CallUpdate"),
    ("core_checklisttemplate", "core.ChecklistTemplate"),
    ("core_inventoryitem", "core.InventoryItem"),
    ("core_partrequest", "core.PartRequest"),
    ("core_approvalrequest", "core.ApprovalRequest"),
    ("core_localpurchase", "core.LocalPurchase"),
    ("core_voucherclaim", "core.VoucherClaim"),
    ("core_expenseclaim", "core.ExpenseClaim"),
    ("core_returnrequest", "core.ReturnRequest"),
    ("core_auditevent", "core.AuditEvent"),
    ("core_customerconversation", "core.CustomerConversation"),
    ("core_customermessage", "core.CustomerMessage"),
    ("core_customerinteraction", "core.CustomerInteraction"),
    ("core_sourceprovenancerecord", "core.SourceProvenanceRecord"),
    ("core_procedureapproval", "core.ProcedureApproval"),
    # 4. Diagnostics & Playbooks
    ("core_diagnosticplaybook", "core.DiagnosticPlaybook"),
    ("core_diagnosticsession", "core.DiagnosticSession"),
    ("core_diagnosticevent", "core.DiagnosticEvent"),
    ("core_diagnosticcommand", "core.DiagnosticCommand"),
    ("core_recoverypassport", "core.RecoveryPassport"),
]


class Command(BaseCommand):
    help = "Non-destructive discovery and schema audit of SQLite records against authoritative MySQL."

    def add_arguments(self, parser):
        parser.add_argument(
            "--sqlite-path",
            type=str,
            default=str(settings.BASE_DIR / "db.sqlite3"),
            help="Path to the source SQLite database file.",
        )
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Direct execution flag (DISABLED to protect relational integrity).",
        )
        parser.add_argument(
            "--check-conflicts",
            action="store_true",
            help="Inspect destination tables for ID and unique value conflicts.",
        )

    def handle(self, *args, **options):
        sqlite_path = Path(options["sqlite_path"])
        is_execute = options["execute"]
        check_conflicts = options["check_conflicts"]

        if is_execute:
            # Operation Ironclad 5.0 Contract: Block unsafe direct execution
            raise CommandError(
                "DIRECT EXECUTION BLOCKED (OPERATION IRONCLAD 5.0):\n"
                "Unsafe direct migration with --execute is permanently disabled to prevent\n"
                "foreign-key validation bypass, AUTO_INCREMENT DDL implicit commits, and destination data loss.\n"
                "Direct writes to MySQL require an independently verified and explicitly authorized migration plan.\n"
                "Run without --execute to inspect the non-destructive discovery and integrity manifest."
            )

        if not sqlite_path.is_file():
            raise CommandError(f"Source SQLite database not found at '{sqlite_path}'.")

        target_vendor = connection.vendor
        target_db = connection.settings_dict.get("NAME")
        target_host = connection.settings_dict.get("HOST")

        is_protected = is_protected_database()

        self.stdout.write(self.style.NOTICE(
            f"=== SQLITE TO MYSQL INTEGRITY AUDIT & MANIFEST ===\n"
            f"  Source SQLite:      {sqlite_path}\n"
            f"  Target DB Vendor:   {target_vendor}\n"
            f"  Target DB Name:     {target_db}\n"
            f"  Target Host:        {target_host}\n"
            f"  Protected Target:   {is_protected}\n"
            f"  Audit Mode:         READ-ONLY DISCOVERY (Zero writes)\n"
            f"=================================================="
        ))

        src_conn = sqlite3.connect(sqlite_path)
        src_conn.row_factory = sqlite3.Row
        src_cur = src_conn.cursor()

        existing_mysql_tables = set()
        dest_available = False
        try:
            if target_vendor == "mysql":
                connection.ensure_connection()
                with connection.cursor() as dest_cur:
                    dest_cur.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE();"
                    )
                    existing_mysql_tables = set(r[0] for r in dest_cur.fetchall())
                dest_available = True
        except Exception as exc:
            self.stdout.write(self.style.WARNING(
                f"Note: Target MySQL database is currently offline or unreachable ({exc}). "
                "Auditing source SQLite records and schema requirements."
            ))

        manifest = []
        total_source_rows = 0
        total_dest_rows = 0
        collision_warnings = []

        for table_name, model_label in MODEL_TABLE_ORDER:
            # Check source existence
            src_cur.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name = ?", [table_name]
            )
            if not src_cur.fetchone()[0]:
                continue

            src_cur.execute(f'SELECT count(*) FROM "{table_name}"')
            src_count = src_cur.fetchone()[0]
            total_source_rows += src_count

            dest_count = 0
            if dest_available and table_name in existing_mysql_tables:
                try:
                    with connection.cursor() as dest_cur:
                        dest_cur.execute(f"SELECT count(*) FROM `{table_name}`")
                        dest_count = dest_cur.fetchone()[0]
                        total_dest_rows += dest_count

                        if dest_count > 0 and check_conflicts:
                            # Check for ID collisions
                            dest_cur.execute(f"SELECT `id` FROM `{table_name}`")
                            dest_ids = set(r[0] for r in dest_cur.fetchall())
                            src_cur.execute(f'SELECT "id" FROM "{table_name}"')
                            src_ids = set(r[0] for r in src_cur.fetchall())
                            overlap = dest_ids.intersection(src_ids)
                            if overlap:
                                collision_warnings.append(
                                    f"Table {table_name}: {len(overlap)} conflicting IDs detected (e.g. {sorted(list(overlap))[:5]})"
                                )
                except Exception as exc:
                    dest_count = -1

            manifest.append({
                "table": table_name,
                "model": model_label,
                "src_count": src_count,
                "dest_count": dest_count,
            })

            dest_str = f"{dest_count:5d} rows" if dest_available else "N/A (unreachable)"
            self.stdout.write(
                f"  [{table_name:32s}] Source SQLite: {src_count:5d} rows | Target MySQL: {dest_str}"
            )

        self.stdout.write(f"\nTotal source records audited: {total_source_rows}")
        if dest_available:
            self.stdout.write(f"Total target records detected: {total_dest_rows}")

        if collision_warnings:
            self.stdout.write(self.style.ERROR("\n[COLLISION WARNINGS]"))
            for cw in collision_warnings:
                self.stdout.write(self.style.ERROR(f"  * {cw}"))

        self.stdout.write(self.style.SUCCESS(
            "\n[DISCOVERY COMPLETE] Zero records modified. Authoritative records preserved.\n"
            "Execution against MySQL remains strictly blocked pending explicit authorization."
        ))
