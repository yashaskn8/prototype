"""
Unit tests for Industrial Safety Approval Authority segregation.
Verifies:
1. Training reviewers and annotators cannot approve physical industrial procedures.
2. Generic database staff users without explicit safety permissions are rejected.
3. Database users with core.can_approve_industrial_safety permission or superuser status are accepted.
4. Users with prohibited keywords in username are rejected even if superuser.
5. Standard authorized industrial safety approver titles are accepted.
6. Integration with validate_procedure_safety_approval fails closed on unauthorized approvers.
"""

from unittest.mock import MagicMock
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType
from django.utils import timezone

from core.services.safety_authority import (
    is_authorized_safety_approver,
    validate_procedure_safety_approval,
    PROHIBITED_REVIEWER_KEYWORDS,
)

User = get_user_model()


class IndustrialSafetyAuthorityTests(TestCase):
    def setUp(self):
        super().setUp()
        from core.models import ProcedureApproval
        self.perm_content_type = ContentType.objects.get_for_model(ProcedureApproval)
        self.safety_perm, _ = Permission.objects.get_or_create(
            codename="can_approve_industrial_safety",
            content_type=self.perm_content_type,
            defaults={"name": "Can approve industrial safety procedures"},
        )

    def test_prohibited_keywords_rejected_immediately(self):
        """Any identity matching PROHIBITED_REVIEWER_KEYWORDS must fail closed."""
        prohibited_samples = [
            "training_annotator_1",
            "Dataset Curator",
            "ML Engineer",
            "Synthetic Bot",
            "automated_pipeline",
            "test_runner",
            "dummy_reviewer",
            "placeholder_author",
            "self-approved",
            "unverified_actor",
            "unknown",
            "anonymous",
            "none",
        ]
        for ident in prohibited_samples:
            self.assertFalse(
                is_authorized_safety_approver(ident),
                f"Expected '{ident}' to be rejected due to prohibited keywords",
            )

    def test_generic_staff_user_is_rejected(self):
        """A generic Django staff user without safety permission must be rejected."""
        staff_user = User.objects.create_user(
            username="generic_staff_tech",
            password="testpassword123",
            is_staff=True,
            is_superuser=False,
        )
        self.assertFalse(
            is_authorized_safety_approver("generic_staff_tech"),
            "Generic staff user without explicit permission must NOT have safety approval authority",
        )

    def test_user_with_explicit_safety_permission_is_authorized(self):
        """A user with 'can_approve_industrial_safety' permission must be authorized."""
        safety_officer = User.objects.create_user(
            username="certified_safety_officer",
            password="testpassword123",
            is_staff=True,
            is_superuser=False,
        )
        safety_officer.user_permissions.add(self.safety_perm)
        safety_officer = User.objects.get(pk=safety_officer.pk)  # reload
        self.assertTrue(
            is_authorized_safety_approver("certified_safety_officer"),
            "User with explicit can_approve_industrial_safety permission should be authorized",
        )

    def test_superuser_is_authorized(self):
        """An active Django superuser is authorized by default."""
        super_admin = User.objects.create_superuser(
            username="root_safety_lead",
            email="root@example.com",
            password="testpassword123",
        )
        self.assertTrue(
            is_authorized_safety_approver("root_safety_lead"),
            "Active superuser without prohibited keywords should be authorized",
        )

    def test_superuser_with_prohibited_keyword_is_rejected(self):
        """Even a superuser with a prohibited keyword in username must be rejected."""
        training_admin = User.objects.create_superuser(
            username="training_root_admin",
            email="training@example.com",
            password="testpassword123",
        )
        self.assertFalse(
            is_authorized_safety_approver("training_root_admin"),
            "Superuser with 'training' keyword must fail closed",
        )

    def test_settings_authorized_safety_approvers(self):
        """Standard authorized safety titles from settings must pass."""
        self.assertTrue(is_authorized_safety_approver("Certified OEM Safety Engineer"))
        self.assertTrue(is_authorized_safety_approver("Director of Product Safety"))
        self.assertTrue(is_authorized_safety_approver("Senior Industrial Safety Auditor"))

    def test_unvetted_arbitrary_strings_rejected(self):
        """Arbitrary strings must be rejected."""
        self.assertFalse(is_authorized_safety_approver("random_external_tech"))
        self.assertFalse(is_authorized_safety_approver(""))
        self.assertFalse(is_authorized_safety_approver("   "))
        self.assertFalse(is_authorized_safety_approver(None))

    def test_validate_procedure_safety_approval_rejects_unauthorized_approver(self):
        """validate_procedure_safety_approval must reject procedures reviewed by unauthorized actors."""
        mock_doc = MagicMock()
        mock_doc.is_rag_enabled = True
        mock_doc.is_confidential = False
        mock_doc.index_version = "1.0"

        mock_app = MagicMock()
        mock_app.revoked_at = None
        mock_app.approval_status = "APPROVED"
        mock_app.effective_date = timezone.now() - timezone.timedelta(days=1)
        mock_app.expiration_date = timezone.now() + timezone.timedelta(days=365)
        mock_app.asset_scope = "*"
        mock_app.equipment_model = "*"
        mock_app.equipment_manufacturer = "*"
        mock_app.document = mock_doc
        mock_app.document_version = "1.0"
        mock_app.prerequisites = []
        mock_app.hazards = []
        mock_app.procedure_keywords = ["reset", "power"]
        mock_app.procedure_id = "PROC-PWR-001"

        # Case 1: Prohibited training annotator reviewer
        mock_app.reviewer_identity = "training_annotator_42"
        mock_app.approval_evidence = "Verified against OEM manual section 4.1"
        is_valid, classification, reason = validate_procedure_safety_approval(
            mock_app, "Turn off power switch and reset breaker"
        )
        self.assertFalse(is_valid)
        self.assertEqual(classification, "UNKNOWN_REQUIRES_REVIEW")
        self.assertIn("lacks authorized safety approval clearance", reason)

        # Case 2: Generic staff without safety authority
        mock_app.reviewer_identity = "generic_staff_tech"
        User.objects.create_user(
            username="generic_staff_tech",
            password="testpassword123",
            is_staff=True,
            is_superuser=False,
        )
        is_valid, classification, reason = validate_procedure_safety_approval(
            mock_app, "Turn off power switch and reset breaker"
        )
        self.assertFalse(is_valid)
        self.assertEqual(classification, "UNKNOWN_REQUIRES_REVIEW")

        # Case 3: Legitimate authorized safety approver
        mock_app.reviewer_identity = "Director of Product Safety"
        is_valid, classification, reason = validate_procedure_safety_approval(
            mock_app, "Turn off power switch and reset breaker"
        )
        self.assertTrue(is_valid)
        self.assertEqual(classification, "CUSTOMER_APPROVED")
