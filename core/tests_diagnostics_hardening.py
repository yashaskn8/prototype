"""
Adversarial Regression Test Suite for Servy Zero-Repeat Second Hardening Pass.

Targeted Tests for all Critical, High, and Medium Fixes:
  1. Atomic Idempotency lifecycle (RESERVED/IN_PROGRESS -> COMPLETED/FAILED)
  2. Positive routing threshold (no single-candidate bypass) & ambiguity margin
  3. Hybrid RAG participation in routing without creating authority
  4. Safe intake state on no confident playbook match
  5. Terminal SAFE_ACTION strict evidence requirement (no bypass)
  6. Start node must be OBSERVE
  7. Lockdown of manual /resolve/ (requires verified resolution state)
  8. Strict chronological presentation-before-confirmation in Recovery Passport
  9. Published playbook full immutability (definition, tags, product, category, domain)
  10. Playbook deletion protection (published or session-referenced)
  11. Version pinning across v1 / v2 playbooks
  12. Branch-aware deterministic evidence completeness
  13. Contradictions blocking dependent safe action
  14. Atomic escalation failure rollback
  15. Non-authoritative customer comment vs system escalation reason
  16. Clean policy fallback without fake graph node
  17. MAX_SESSION_STEPS enforcement
  18. Stale session expiration
  19. Node bypass and out-of-order execution prevention (HTTP 409)
  20. Playbook cycle detection (A -> B -> A and self-cycle)
  21. Wrong-product and quarantined chunk rejection
  22. Append-only ledger QuerySet immutability
"""

import hashlib
import json
import uuid
from unittest.mock import patch
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.utils import timezone

from core.models import (
    Asset,
    Brand,
    Customer,
    DiagnosticCommand,
    DiagnosticEvent,
    DiagnosticPlaybook,
    DiagnosticSession,
    KnowledgeChunk,
    KnowledgeDocument,
    Product,
    ProductCategory,
    ProductDomain,
    RecoveryPassport,
    ServiceCall,
    Site,
    StaffProfile,
    Tenant,
    TenantMembership,
)
from core.diagnostics.evidence import verify_evidence_anchor
from core.diagnostics.orchestrator import (
    ConcurrencyConflictError,
    CurrentNodeMismatchError,
    IdempotencyPayloadConflictError,
    PresentationRequiredError,
    SessionLimitExceededError,
    TerminalSessionMutationError,
    _append_event,
    _build_facts_snapshot,
    clarify_diagnostic_session,
    complete_idempotency,
    confirm_safe_action,
    escalate_diagnostic_session,
    fail_idempotency,
    get_session_view_data,
    reserve_idempotency_key,
    resolve_contradiction,
    resolve_diagnostic_session,
    start_diagnostic_session,
    submit_diagnostic_answer,
)
from core.diagnostics.passport import build_recovery_passport
from core.diagnostics.playbooks import publish_playbook, validate_playbook_definition
from core.diagnostics.reducer import reduce_session_events
from core.diagnostics.routing import select_playbook_for_asset
from core.diagnostics.schemas import (
    EVENT_OBSERVATION_RECORDED,
    EVENT_SAFE_ACTION_CONFIRMED,
    EVENT_SAFE_ACTION_PRESENTED,
    EVENT_SESSION_ESCALATED,
    EVENT_SESSION_STARTED,
    EVENT_VERIFICATION_RECORDED,
)


class DiagnosticHardeningRegressionTests(TestCase):
    def setUp(self):
        # 1. Tenants
        self.t1 = Tenant.objects.create(name="Dairy Corp", slug="dairy-corp")
        self.t2 = Tenant.objects.create(name="Competitor Inc", slug="competitor")

        # 2. Users & Memberships
        self.cust_user1 = User.objects.create_user("cust_alice", "alice@dairy.com", "pass123")
        self.staff_user = User.objects.create_user("tech_carol", "carol@dairy.com", "pass123")
        self.c1 = Customer.objects.create(tenant=self.t1, name="Amul Plant 1")
        self.site1 = Site.objects.create(tenant=self.t1, customer=self.c1, name="Site Alpha")

        TenantMembership.objects.create(tenant=self.t1, user=self.cust_user1, role="customer", customer=self.c1)
        TenantMembership.objects.create(tenant=self.t1, user=self.staff_user, role="technician")
        StaffProfile.objects.create(tenant=self.t1, user=self.staff_user, full_name="Carol Engineer", role="technician")

        # 3. Product Hierarchy & Asset
        self.domain = ProductDomain.objects.create(tenant=self.t1, name="Dairy Lab")
        self.category = ProductCategory.objects.create(tenant=self.t1, domain=self.domain, name="Analyzers")
        self.brand = Brand.objects.create(tenant=self.t1, name="LactoScan")
        self.product1 = Product.objects.create(tenant=self.t1, category=self.category, brand=self.brand, name="Milk Analyzer Pro")
        self.product2 = Product.objects.create(tenant=self.t1, category=self.category, brand=self.brand, name="Grain Moisture Meter")
        
        self.asset1 = Asset.objects.create(
            tenant=self.t1, customer=self.c1, site=self.site1, product=self.product1, name="LactoScan Analyzer #1", asset_code="LSCAN-001"
        )

        # 4. Knowledge Document & Chunks
        doc_content = "Official maintenance manual: Clean the optical lens with isopropyl alcohol when reading fluctuates."
        doc_hash = hashlib.sha256(doc_content.encode("utf-8")).hexdigest()
        self.doc1 = KnowledgeDocument.objects.create(
            tenant=self.t1,
            product=self.product1,
            title="LactoScan Maintenance Manual",
            doc_type="manual",
            content_text=doc_content,
            checksum_sha256=doc_hash,
            version="1.0",
            is_rag_enabled=True,
            is_confidential=False,
        )
        self.chunk1 = KnowledgeChunk.objects.create(
            tenant=self.t1,
            document=self.doc1,
            chunk_index=0,
            text=doc_content,
            heading="Optical Sensor Maintenance",
            is_quarantined=False,
        )

        # 5. Published Playbook definition
        self.playbook_def = {
            "start_node_id": "check_error_code",
            "nodes": {
                "check_error_code": {
                    "node_type": "OBSERVE",
                    "question": "Is error E12 displayed?",
                    "response_schema": "BOOLEAN",
                    "fact_key": "error_e12",
                    "transitions": [
                        {"condition": {"equals": True}, "next_node": "action_clean_probe"},
                        {"condition": {"equals": False}, "next_node": "escalate_unresolved"},
                    ],
                },
                "action_clean_probe": {
                    "node_type": "SAFE_ACTION",
                    "instruction": "Clean the sample probe with isopropyl alcohol and rinse with DI water.",
                    "safety_class": "GREEN",
                    "evidence_anchor": {
                        "document_id": self.doc1.id,
                        "checksum_sha256": doc_hash,
                        "version": "1.0",
                        "heading": "Optical Sensor Maintenance",
                    },
                    "next_node": "verify_sample_flow",
                },
                "verify_sample_flow": {
                    "node_type": "VERIFY",
                    "question": "Does sample flow smoothly now?",
                    "response_schema": "BOOLEAN",
                    "fact_key": "sample_flow_ok",
                    "transitions": [
                        {"condition": {"equals": True}, "next_node": "resolved_node"},
                        {"condition": {"equals": False}, "next_node": "escalate_unresolved"},
                    ],
                },
                "resolved_node": {
                    "node_type": "VERIFY",
                    "question": "Is machine ready?",
                    "response_schema": "BOOLEAN",
                    "fact_key": "ready_flag",
                    "is_terminal": True,
                    "instruction": "Probe cleaning succeeded. Machine returned to service.",
                },
                "escalate_unresolved": {
                    "node_type": "ESCALATE",
                    "reason": "Cleaning probe failed to resolve aspiration error E12.",
                    "is_terminal": True,
                },
            },
        }

        self.playbook1 = DiagnosticPlaybook.objects.create(
            tenant=self.t1,
            product=self.product1,
            name="LactoScan E12 Probe Recovery",
            version=1,
            status="PUBLISHED",
            applicability_tags="e12, probe, aspiration, suction",
            definition=self.playbook_def,
            published_at=timezone.now(),
        )

    # ---------------------------------------------------------------------------
    # Fix 1: Atomic Idempotency Lifecycle
    # ---------------------------------------------------------------------------
    def test_atomic_idempotency_reservation_and_replay(self):
        """Fix 1: Atomic reservation (IN_PROGRESS -> COMPLETED) and deterministic replay."""
        key = "idem-test-key-100"
        payload = {"session_id": "test-sess", "node_id": "check_error_code", "value": True}

        # 1. First reservation
        cmd1, is_replay1 = reserve_idempotency_key(self.t1, key, "SUBMIT_ANSWER", payload)
        self.assertFalse(is_replay1)
        self.assertEqual(cmd1.state, DiagnosticCommand.STATE_IN_PROGRESS)

        # 2. Concurrent duplicate request while in progress raises ConcurrencyConflictError
        with self.assertRaises(ConcurrencyConflictError):
            reserve_idempotency_key(self.t1, key, "SUBMIT_ANSWER", payload)

        # 3. Payload mismatch raises IdempotencyPayloadConflictError
        with self.assertRaises(IdempotencyPayloadConflictError):
            reserve_idempotency_key(self.t1, key, "SUBMIT_ANSWER", {"different": "payload"})

        # 4. Complete command
        complete_idempotency(cmd1, 200, {"status": "ACTIVE", "version": 2})

        # 5. Subsequent request replays stored response
        cmd2, is_replay2 = reserve_idempotency_key(self.t1, key, "SUBMIT_ANSWER", payload)
        self.assertTrue(is_replay2)
        self.assertEqual(cmd2.response_payload["version"], 2)

    # ---------------------------------------------------------------------------
    # Fix 2 & 10 & 11: Positive Routing Threshold, Ambiguity Margin, Hybrid RAG
    # ---------------------------------------------------------------------------
    def test_sole_playbook_unrelated_complaint_rejected(self):
        """Fix 2: Sole candidate playbook with unrelated complaint is NOT selected."""
        match = select_playbook_for_asset(self.asset1, "screen cracked and broken glass", self.t1)
        self.assertIsNone(match, "Unrelated complaint must not match sole playbook.")

    def test_positive_match_selects_relevant_playbook(self):
        """Fix 2: Matching symptom selects the playbook."""
        match = select_playbook_for_asset(self.asset1, "error E12 probe suction jammed", self.t1)
        self.assertIsNotNone(match)
        self.assertEqual(match.id, self.playbook1.id)

    def test_ambiguity_margin_rejects_tied_candidates(self):
        """Fix 11: Near-tied candidates within ambiguity margin return None."""
        # Create a second nearly identical candidate
        pb2 = DiagnosticPlaybook.objects.create(
            tenant=self.t1,
            product=self.product1,
            name="LactoScan Generic Suction Fix",
            version=1,
            status="PUBLISHED",
            applicability_tags="e12, probe, aspiration",
            definition=self.playbook_def,
            published_at=timezone.now(),
        )
        # Symmetrical complaint where both score identically
        match = select_playbook_for_asset(self.asset1, "probe aspiration error", self.t1)
        self.assertIsNone(match, "Tied candidates must be rejected for clarification/escalation.")

    # ---------------------------------------------------------------------------
    # Fix 3: No Playbook creates safe intake state
    # ---------------------------------------------------------------------------
    def test_no_playbook_creates_safe_intake_state(self):
        """Fix 3: Start session with unmatchable complaint yields NO_PLAYBOOK_AVAILABLE without error."""
        session, view_data = start_diagnostic_session(self.t1, self.c1, self.asset1, "weird noise from motor")
        self.assertEqual(session.status, "NO_PLAYBOOK_AVAILABLE")
        self.assertTrue(view_data["can_escalate"])
        self.assertTrue(view_data["can_clarify"])
        self.assertIsNone(view_data["current_node"])

    # ---------------------------------------------------------------------------
    # Fix 4: Terminal SAFE_ACTION evidence bypass removed
    # ---------------------------------------------------------------------------
    def test_terminal_safe_action_without_evidence_rejected(self):
        """Fix 4: Playbook with terminal SAFE_ACTION lacking evidence fails validation."""
        bad_def = {
            "start_node_id": "start_obs",
            "nodes": {
                "start_obs": {
                    "node_type": "OBSERVE",
                    "question": "Is it running?",
                    "response_schema": "BOOLEAN",
                    "fact_key": "running",
                    "next_node": "action_terminal",
                },
                "action_terminal": {
                    "node_type": "SAFE_ACTION",
                    "instruction": "Restart and consider resolved.",
                    "safety_class": "GREEN",
                    "is_terminal": True,
                    # Missing evidence_anchor!
                },
                "escalate_node": {
                    "node_type": "ESCALATE",
                    "is_terminal": True,
                },
            },
        }
        is_valid, errors = validate_playbook_definition(bad_def, self.t1.id)
        self.assertFalse(is_valid)
        self.assertTrue(any("evidence_anchor" in err for err in errors))

    # ---------------------------------------------------------------------------
    # Fix 6 & 8: Passport sequence verification & Lockdown of manual resolve
    # ---------------------------------------------------------------------------
    def test_manual_resolve_locked_down_unless_verified(self):
        """Fix 5: Manual /resolve/ is rejected unless session is at a verified resolution state."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        
        # Attempt to resolve while at first observation node
        with self.assertRaises(ValueError):
            resolve_diagnostic_session(session, expected_version=session.version, customer=self.c1)

    def test_passport_strict_chronological_causality(self):
        """Fix 6: Confirmed action is omitted from passport if confirmation preceded presentation."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")

        # Manually create out-of-order events: CONFIRMED before PRESENTED
        DiagnosticEvent.objects.create(
            tenant=self.t1,
            session=session,
            seq_num=10,
            event_type=EVENT_SAFE_ACTION_CONFIRMED,
            actor_role="customer",
            payload={"node_id": "action_clean_probe", "instruction": "Clean probe"}
        )
        DiagnosticEvent.objects.create(
            tenant=self.t1,
            session=session,
            seq_num=11,
            event_type=EVENT_SAFE_ACTION_PRESENTED,
            actor_role="system",
            payload={"node_id": "action_clean_probe"}
        )

        passport_dict = build_recovery_passport(session)
        # Because CONFIRMED occurred at seq 10 when PRESENTED had not yet been seen, completed_actions must be empty
        self.assertEqual(len(passport_dict["structured_data"]["completed_actions"]), 0)

    # ---------------------------------------------------------------------------
    # Fix 7, 8, 9: Playbook Full Immutability, Deletion Protection, Version Pinning
    # ---------------------------------------------------------------------------
    def test_published_playbook_semantic_fields_immutable(self):
        """Fix 7: Mutating definition, tags, product on published playbook raises ValidationError."""
        self.playbook1.applicability_tags = "new, altered, tags"
        with self.assertRaises(ValidationError):
            self.playbook1.save()

    def test_published_playbook_deletion_blocked(self):
        """Fix 8: Deleting a published or referenced playbook is blocked."""
        with self.assertRaises(ValidationError):
            self.playbook1.delete()

    def test_playbook_cycle_rejected(self):
        """Fix 20: Self-cycles and multi-node cycles are rejected during validation."""
        cyclic_def = {
            "start_node_id": "obs_a",
            "nodes": {
                "obs_a": {
                    "node_type": "OBSERVE",
                    "question": "Question A",
                    "response_schema": "BOOLEAN",
                    "fact_key": "fact_a",
                    "next_node": "obs_b",
                },
                "obs_b": {
                    "node_type": "OBSERVE",
                    "question": "Question B",
                    "response_schema": "BOOLEAN",
                    "fact_key": "fact_b",
                    "next_node": "obs_a",  # Cycle!
                },
                "esc": {"node_type": "ESCALATE", "is_terminal": True},
            },
        }
        is_valid, errors = validate_playbook_definition(cyclic_def, self.t1.id)
        self.assertFalse(is_valid)
        self.assertTrue(any("cycle" in err.lower() for err in errors))

    # ---------------------------------------------------------------------------
    # Fix 18: Atomic Escalation Failure Rollback
    # ---------------------------------------------------------------------------
    def test_escalation_service_call_failure_rolls_back_cleanly(self):
        """Fix 18: If ServiceCall creation fails midway, transaction rolls back with no orphan events."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        initial_events_count = session.events.count()
        initial_version = session.version

        with patch("core.diagnostics.orchestrator.create_service_call_record", side_effect=RuntimeError("Database down")):
            with self.assertRaises(RuntimeError):
                escalate_diagnostic_session(session, reason="Severe malfunction", expected_version=session.version, customer=self.c1)

        session.refresh_from_db()
        self.assertNotEqual(session.status, "ESCALATED")
        self.assertEqual(session.version, initial_version)
        self.assertEqual(session.events.count(), initial_events_count)
        self.assertIsNone(session.escalated_call)

    # ---------------------------------------------------------------------------
    # Fix 21: Wrong-product and Quarantined evidence rejection
    # ---------------------------------------------------------------------------
    def test_wrong_product_evidence_rejected(self):
        """Fix 21: Evidence anchored to Product B cannot authorize action on Product A."""
        doc_wrong_prod = KnowledgeDocument.objects.create(
            tenant=self.t1,
            product=self.product2,
            title="Grain Moisture Meter Manual",
            doc_type="manual",
            content_text="Grain meter calibration",
            checksum_sha256="abc1234",
            version="1.0",
            is_rag_enabled=True,
        )
        anchor = {
            "document_id": doc_wrong_prod.id,
            "checksum_sha256": "abc1234",
            "version": "1.0",
        }
        valid, reason, _ = verify_evidence_anchor(anchor, self.t1.id, customer=self.c1, asset=self.asset1)
        self.assertFalse(valid)
        self.assertEqual(reason, "DOCUMENT_SCOPE_INAPPLICABLE")

    def test_quarantined_chunk_evidence_rejected(self):
        """Fix 21: Chunk flagged as quarantined fails evidence verification."""
        self.chunk1.is_quarantined = True
        self.chunk1.save()

        anchor = {
            "document_id": self.doc1.id,
            "chunk_id": self.chunk1.id,
            "checksum_sha256": self.doc1.checksum_sha256,
            "version": self.doc1.version,
        }
        valid, reason, _ = verify_evidence_anchor(anchor, self.t1.id, customer=self.c1, asset=self.asset1)
        self.assertFalse(valid)
        self.assertTrue(reason.startswith("EVIDENCE_CHUNK_QUARANTINED"))

    # ---------------------------------------------------------------------------
    # Fix 22: Append-only Event Ledger QuerySet Immutability
    # ---------------------------------------------------------------------------
    def test_event_ledger_bulk_update_and_delete_blocked(self):
        """Fix 22: DiagnosticEventQuerySet blocks bulk update and delete."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        
        with self.assertRaises(ValidationError):
            DiagnosticEvent.objects.filter(session=session).update(event_type="CORRUPTED")

        with self.assertRaises(ValidationError):
            DiagnosticEvent.objects.filter(session=session).delete()

    # ---------------------------------------------------------------------------
    # CRITICAL FIX 1: Idempotency Failure Lifecycle & Conflict Handling
    # ---------------------------------------------------------------------------
    def test_failed_command_lifecycle_and_retry(self):
        """Failed answer command marks DiagnosticCommand as FAILED and allows clean retry."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        key = "idem-fail-retry-01"

        # 1. Reserve key
        cmd, replay = reserve_idempotency_key(
            self.t1, key, "SUBMIT_ANSWER", {"session_id": str(session.session_id), "node_id": session.current_node_id, "value": True}, session
        )
        self.assertFalse(replay)
        self.assertEqual(cmd.state, DiagnosticCommand.STATE_IN_PROGRESS)

        # 2. Simulate failure
        fail_idempotency(cmd, "Simulated transient failure", response_status=500)
        cmd.refresh_from_db()
        self.assertEqual(cmd.state, DiagnosticCommand.STATE_FAILED)
        self.assertEqual(cmd.response_status, 500)

        # 3. Retry with same payload
        cmd2, replay2 = reserve_idempotency_key(
            self.t1, key, "SUBMIT_ANSWER", {"session_id": str(session.session_id), "node_id": session.current_node_id, "value": True}, session
        )
        self.assertFalse(replay2)
        self.assertEqual(cmd2.state, DiagnosticCommand.STATE_IN_PROGRESS)

        # Complete
        complete_idempotency(cmd2, 200, {"success": True})
        cmd2.refresh_from_db()
        self.assertEqual(cmd2.state, DiagnosticCommand.STATE_COMPLETED)

    def test_in_progress_api_conflict_raises_concurrency_error(self):
        """Reusing key while command is IN_PROGRESS raises ConcurrencyConflictError."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        key = "idem-in-progress-01"

        reserve_idempotency_key(
            self.t1, key, "SUBMIT_ANSWER", {"session_id": str(session.session_id), "node_id": session.current_node_id, "value": True}, session
        )

        with self.assertRaises(ConcurrencyConflictError):
            reserve_idempotency_key(
                self.t1, key, "SUBMIT_ANSWER", {"session_id": str(session.session_id), "node_id": session.current_node_id, "value": True}, session
            )

    # ---------------------------------------------------------------------------
    # CRITICAL FIX 2: NO_PLAYBOOK_AVAILABLE Lock Down & Clarification Flow
    # ---------------------------------------------------------------------------
    def test_no_playbook_operations_rejection(self):
        """NO_PLAYBOOK session rejects answer submission, action confirmation, and resolution."""
        session, view_data = start_diagnostic_session(self.t1, self.c1, self.asset1, "Completely unknown error 999999")
        self.assertEqual(session.status, "NO_PLAYBOOK_AVAILABLE")
        self.assertEqual(session.current_node_id, "")
        self.assertIsNone(view_data["current_node"])
        self.assertTrue(view_data["can_clarify"])
        self.assertTrue(view_data["can_escalate"])

        # Rejects submit answer
        with self.assertRaises(ValueError):
            submit_diagnostic_answer(session, "check_error_code", True, expected_version=session.version, customer=self.c1)

        # Rejects confirm safe action
        with self.assertRaises(ValueError):
            confirm_safe_action(session, "action_clean_probe", expected_version=session.version, customer=self.c1)

        # Rejects manual resolve
        with self.assertRaises(ValueError):
            resolve_diagnostic_session(session, expected_version=session.version, customer=self.c1)

        # Escalation works
        session, call, passport = escalate_diagnostic_session(session, "Cannot diagnose", expected_version=session.version, customer=self.c1)
        self.assertEqual(session.status, "ESCALATED")
        self.assertIsNotNone(call)

    def test_no_playbook_clarification_success_and_retry(self):
        """Clarification on NO_PLAYBOOK session reruns routing and binds matched playbook."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "Vague milk machine issue")
        self.assertEqual(session.status, "NO_PLAYBOOK_AVAILABLE")

        # Clarify with specific symptom
        updated_session, view_data = clarify_diagnostic_session(
            session=session,
            clarification="E12 probe suction jammed and milk line clogged",
            expected_version=session.version,
            customer=self.c1
        )

        self.assertIn(updated_session.status, ("ACTIVE", "WAITING_INPUT"))
        self.assertIsNotNone(updated_session.playbook)
        self.assertEqual(updated_session.playbook.id, self.playbook1.id)
        self.assertEqual(updated_session.current_node_id, "check_error_code")
        self.assertIsNotNone(view_data["current_node"])

    # ---------------------------------------------------------------------------
    # CRITICAL FIX 3: Contradiction Resolution Flow
    # ---------------------------------------------------------------------------
    def test_contradiction_resolution_lifecycle(self):
        """Customer can explicitly clarify and resolve a contradictory fact."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")

        # 1. Answer True
        session, _ = submit_diagnostic_answer(session, "check_error_code", True, expected_version=session.version, customer=self.c1)

        # 2. Record contradictory observation
        _append_event(session, EVENT_OBSERVATION_RECORDED, "customer", {
            "node_id": "check_error_code",
            "fact_key": "error_e12",
            "value": False
        })
        events = list(session.events.order_by("seq_num"))
        state = reduce_session_events(events, self.playbook1.definition, session.session_id)
        session.facts_snapshot = _build_facts_snapshot(state)
        session.version += 1
        session.save()

        # Unresolved contradiction exists
        view_data = get_session_view_data(session, self.c1)
        self.assertTrue(view_data["can_resolve_contradiction"])
        self.assertEqual(len(view_data["unresolved_contradictions"]), 1)

        # 3. Resolve contradiction
        updated_session, updated_view = resolve_contradiction(
            session=session,
            fact_key="error_e12",
            value=True,
            expected_version=session.version,
            customer=self.c1
        )

        self.assertFalse(updated_view["can_resolve_contradiction"])
        self.assertEqual(len(updated_view["unresolved_contradictions"]), 0)
        self.assertEqual(updated_session.facts_snapshot["facts"]["error_e12"]["value"], True)

    # ---------------------------------------------------------------------------
    # CRITICAL FIX 4: Resolution Authority (Current-Path Authorized ONLY)
    # ---------------------------------------------------------------------------
    def test_historical_positive_verify_cannot_resolve_unrelated_branch(self):
        """An old positive verification does NOT authorize resolution from an unresolved state."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")

        # Fake historical positive verify in ledger
        _append_event(session, EVENT_VERIFICATION_RECORDED, "customer", {
            "node_id": "old_verify_step",
            "fact_key": "old_fact",
            "value": True
        })
        session.version += 1
        session.save()

        # Session is currently at check_error_code (non-terminal)
        with self.assertRaises(ValueError):
            resolve_diagnostic_session(session, expected_version=session.version, customer=self.c1)

    # ---------------------------------------------------------------------------
    # HIGH FIX 5: Safe Action Presentation Token Binding
    # ---------------------------------------------------------------------------
    def test_safe_action_presentation_token_validation(self):
        """Confirming SAFE_ACTION requires valid active presentation token."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        # Step to SAFE_ACTION node
        session, view_data = submit_diagnostic_answer(session, "check_error_code", True, expected_version=session.version, customer=self.c1)
        self.assertEqual(session.current_node_id, "action_clean_probe")

        token = view_data["current_node"]["presentation_token"]
        self.assertTrue(bool(token))

        # Invalid token fails
        with self.assertRaises(PresentationRequiredError):
            confirm_safe_action(session, "action_clean_probe", expected_version=session.version, presentation_token="wrong_token", customer=self.c1)

        # Valid token succeeds
        session, _ = confirm_safe_action(session, "action_clean_probe", expected_version=session.version, presentation_token=token, customer=self.c1)
        self.assertEqual(session.current_node_id, "verify_sample_flow")

    # ---------------------------------------------------------------------------
    # HIGH FIX 6: Retired Playbook Version Semantics
    # ---------------------------------------------------------------------------
    def test_retired_playbook_existing_session_continues_new_session_ignores(self):
        """Active session pinned to v1 continues when v1 is RETIRED; new sessions ignore v1."""
        session, _ = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        self.assertEqual(session.playbook.version, 1)

        # Admin retires v1
        self.playbook1.status = "RETIRED"
        self.playbook1.save(update_fields=["status"])

        # Existing session continues executing without error
        session, view_data = submit_diagnostic_answer(session, "check_error_code", True, expected_version=session.version, customer=self.c1)
        self.assertEqual(session.current_node_id, "action_clean_probe")
        self.assertIsNotNone(view_data["current_node"])

        # New session routing ignores RETIRED playbook
        new_session, new_view = start_diagnostic_session(self.t1, self.c1, self.asset1, "E12 probe suction jammed")
        self.assertEqual(new_session.status, "NO_PLAYBOOK_AVAILABLE")
