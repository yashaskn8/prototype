"""
Seed Diagnostic Playbook for Milk Analyzer MA-100 with Canonical Evidence Anchoring.
"""

import os
import sys
from pathlib import Path
import django

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
django.setup()

from django.utils import timezone
from core.models import Tenant, Product, KnowledgeDocument, DiagnosticPlaybook
from core.diagnostics.playbooks import publish_playbook


def seed_milk_analyzer_playbook():
    tenants = Tenant.objects.all()
    created_count = 0

    for tenant in tenants:
        doc = KnowledgeDocument.objects.filter(
            tenant=tenant,
            title__icontains="Milk Analyzer MA-100"
        ).first()

        product = Product.objects.filter(tenant=tenant, name__icontains="Milk Analyzer").first()

        if not doc or not product:
            continue

        definition = {
            "start_node_id": "observe_error_code",
            "required_facts": ["error_code_present", "after_cleaning", "reading_stabilized"],
            "max_depth": 10,
            "nodes": {
                "observe_error_code": {
                    "node_type": "OBSERVE",
                    "question": "Is an error code displayed on the screen?",
                    "response_schema": "BOOLEAN",
                    "fact_key": "error_code_present",
                    "transitions": [
                        {"condition": {"equals": True}, "next_node": "observe_error_value"},
                        {"condition": {"equals": False}, "next_node": "observe_after_cleaning"},
                    ],
                },
                "observe_error_value": {
                    "node_type": "OBSERVE",
                    "question": "Which error code is shown (e.g. E04, E12)?",
                    "response_schema": "SHORT_TEXT",
                    "fact_key": "error_code_value",
                    "transitions": [
                        {"default": True, "next_node": "observe_after_cleaning"},
                    ],
                },
                "observe_after_cleaning": {
                    "node_type": "OBSERVE",
                    "question": "Did the unstable reading or issue begin immediately after a cleaning cycle?",
                    "response_schema": "BOOLEAN",
                    "fact_key": "after_cleaning",
                    "transitions": [
                        {"condition": {"equals": True}, "next_node": "action_rinse_and_seat"},
                        {"condition": {"equals": False}, "next_node": "action_check_tube_connection"},
                    ],
                },
                "action_rinse_and_seat": {
                    "node_type": "SAFE_ACTION",
                    "instruction": "Check sample mixing, sample temperature, sampling tube connection, thermostat cup cleanliness and seating. Run a rinse and the approved verification sample.",
                    "safety_class": "GREEN",
                    "evidence_anchor": {
                        "document_id": doc.id,
                        "checksum_sha256": doc.checksum_sha256,
                        "version": doc.version,
                        "heading": "Unstable reading",
                        "title": doc.title,
                    },
                    "next_verification_node": "verify_reading_stabilized",
                },
                "action_check_tube_connection": {
                    "node_type": "SAFE_ACTION",
                    "instruction": "Check the sampling tube for bends or blockage and ensure sample level is sufficient. Run the cleaning cycle.",
                    "safety_class": "GREEN",
                    "evidence_anchor": {
                        "document_id": doc.id,
                        "checksum_sha256": doc.checksum_sha256,
                        "version": doc.version,
                        "heading": "Does not draw sample",
                        "title": doc.title,
                    },
                    "next_verification_node": "verify_reading_stabilized",
                },
                "verify_reading_stabilized": {
                    "node_type": "VERIFY",
                    "question": "Did the verification sample return readings within the accepted tolerance range?",
                    "response_schema": "BOOLEAN",
                    "fact_key": "reading_stabilized",
                    "transitions": [
                        {"condition": {"equals": True}, "next_node": "terminal_resolve"},
                        {"condition": {"equals": False}, "next_node": "terminal_escalate"},
                    ],
                },
                "terminal_resolve": {
                    "node_type": "SAFE_ACTION",
                    "instruction": "Milk Analyzer recovered and verified within operating calibration.",
                    "safety_class": "GREEN",
                    "is_terminal": True,
                    "evidence_anchor": {
                        "document_id": doc.id,
                        "checksum_sha256": doc.checksum_sha256,
                        "version": doc.version,
                        "heading": "Verification after cleaning",
                        "title": doc.title,
                    },
                },
                "terminal_escalate": {
                    "node_type": "ESCALATE",
                    "reason": "Readings remain outside tolerance after rinse, temperature normalization, and cup reseating. Field service dispatch required.",
                    "is_terminal": True,
                },
            },
        }

        playbook, created = DiagnosticPlaybook.objects.get_or_create(
            tenant=tenant,
            name="Milk Analyzer MA-100 Recovery Playbook",
            version=1,
            defaults={
                "product": product,
                "status": "DRAFT",
                "applicability_tags": "unstable reading, reading unstable, cleaning, sensor, calibration, error code, draw sample",
                "definition": definition,
            }
        )

        if not created:
            DiagnosticPlaybook.objects.filter(id=playbook.id).update(
                definition=definition,
                product=product,
                applicability_tags="unstable reading, reading unstable, cleaning, sensor, calibration, error code, draw sample",
            )
            playbook.refresh_from_db()

        if playbook.status != "PUBLISHED":
            publish_playbook(playbook)
        print(f"Published playbook for tenant {tenant.name}: {playbook.name} v{playbook.version}")
        created_count += 1

    print(f"Total playbooks seeded/published: {created_count}")


if __name__ == "__main__":
    seed_milk_analyzer_playbook()
