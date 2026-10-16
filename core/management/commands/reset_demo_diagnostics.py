"""
Management command: reset_demo_diagnostics
Prepares a clean, deterministic demo environment for the Milk Analyzer Zero-Repeat demo.
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from core.models import (
    Tenant, Customer, Asset, Product, KnowledgeDocument,
    DiagnosticPlaybook, DiagnosticSession, DiagnosticCommand,
    DiagnosticEvent, RecoveryPassport, ServiceCall
)
from core.seed_diagnostic_playbook import seed_milk_analyzer_playbook


class Command(BaseCommand):
    help = "Resets the Milk Analyzer demo asset to a clean, repeatable diagnostic state."

    def handle(self, *args, **options):
        tenant = Tenant.objects.filter(name="Starlly Tester").first() or Tenant.objects.first()
        if not tenant:
            self.stdout.write(self.style.ERROR("No tenant found."))
            return

        customer = Customer.objects.filter(tenant=tenant, name__icontains="FreshDairy").first()
        asset = Asset.objects.filter(tenant=tenant, name="Milk Analyzer Lab-01").first()

        if not asset:
            self.stdout.write(self.style.ERROR("Asset 'Milk Analyzer Lab-01' not found."))
            return

        from core.security import is_protected_database
        from django.core.management.base import CommandError

        if is_protected_database():
            raise CommandError(
                "DESTRUCTIVE OPERATION BLOCKED: reset_demo_diagnostics cannot operate on the protected database. "
                "Diagnostic event ledgers and session history are immutable."
            )

        with transaction.atomic():
            # 1. Clear any active or previous diagnostic sessions, commands, and passports for this asset
            sessions = DiagnosticSession.objects.filter(tenant=tenant, asset=asset)
            session_ids = list(sessions.values_list("id", flat=True))

            if session_ids:
                DiagnosticEvent.objects.filter(session_id__in=session_ids).delete()
                DiagnosticCommand.objects.filter(session_id__in=session_ids).delete()
                RecoveryPassport.objects.filter(session_id__in=session_ids).delete()
                sessions.delete()
            RecoveryPassport.objects.filter(asset=asset).delete()

            # 2. Reset service calls created for this asset from diagnostic escalation
            ServiceCall.objects.filter(
                tenant=tenant,
                asset=asset,
                complaint_type="Diagnostic Recovery Escalation"
            ).delete()

            # Ensure all remaining calls for this asset are closed history (keep up to 2 for history display)
            open_calls = ServiceCall.objects.filter(tenant=tenant, asset=asset, status__in=["open", "assigned", "in_progress"])
            for call in open_calls:
                call.status = "closed"
                call.save(update_fields=["status"])

            # 3. Ensure Milk Analyzer MA-100 playbook is seeded, published, and validly linked
            seed_milk_analyzer_playbook()

        self.stdout.write(self.style.SUCCESS(
            f"Successfully reset demo diagnostics for {asset.name} (Tenant: {tenant.name}, Customer: {customer.name if customer else 'None'}).\n"
            f"- Open Diagnostic Sessions: 0\n"
            f"- Recovery Passports: 0\n"
            f"- Open Service Calls for Asset: 0\n"
            f"- Playbook: Seeded and Published"
        ))
