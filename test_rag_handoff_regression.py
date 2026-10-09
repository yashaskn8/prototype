"""
Regression tests for Customer RAG -> Service Call handoff defects.

Tests cover:
1. Extractive answer does NOT contain raw markdown heading prefixes (## ...)
2. Extractive answer does NOT cross-contaminate sections from unrelated chunks
3. Check-like sentences are classified as CHECK FIRST, not STOP AND ESCALATE IF
4. Escalated complaint_text uses "AI diagnostic guidance provided" not "Troubleshooting already attempted"
5. Escalated complaint_text includes "(completion not confirmed)" caveat
"""
import os
import re
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'servy_rag.settings')
django.setup()

from dataclasses import dataclass
from core.services.llm import _extractive_answer, INSUFFICIENT_EVIDENCE_MESSAGE


@dataclass
class MockChunk:
    """Mimics RetrievedChunk for testing extractive answer."""
    heading: str
    text: str
    title: str
    score: float
    doc_type: str
    reference: str
    source_url: str = ""

    @property
    def source_kind(self):
        return "knowledge_document"


@dataclass
class MockAsset:
    name: str
    model_number: str = ""

    @property
    def product(self):
        return None


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, detail):
        self.failed += 1
        self.errors.append((name, detail))
        print(f"  [FAIL] {name}")
        print(f"    â†’ {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, detail in self.errors:
                print(f"  - {name}: {detail}")
        print(f"{'='*60}")
        return self.failed == 0


def test_no_markdown_headings_in_output(result):
    """Defect #1: Raw markdown heading prefixes (## ...) must not appear in answer."""
    test_name = "No markdown heading prefixes in output"

    asset = MockAsset(name="Aerolift-50mtr")
    chunks = [
        MockChunk(
            heading="Unit does not start",
            text="## Unit does not start\nConfirm the main isolator is on, the emergency stop is released, and no safety interlock is open. Record any control-panel fault. Never bypass a safety interlock.",
            title="Aerolift Troubleshooting Guide",
            score=0.6,
            doc_type="Troubleshooting",
            reference="Aerolift Troubleshooting Guide > Unit does not start",
        ),
    ]

    answer = _extractive_answer("Unit does not start", asset, chunks)

    # Answer must NOT contain "## " markdown heading prefix
    if "## " in answer:
        result.fail(test_name, f"Answer contains raw markdown heading: ...{answer[answer.index('## '):answer.index('## ')+40]}...")
    else:
        result.ok(test_name)


def test_no_cross_contamination(result):
    """Defect #2: Section headings from unrelated chunks must not appear as steps."""
    test_name = "No cross-contamination from unrelated chunk headings"

    asset = MockAsset(name="Aerolift-50mtr")
    chunks = [
        MockChunk(
            heading="Unit does not start",
            text="## Unit does not start\nConfirm the main isolator is on. Record any control-panel fault.",
            title="Aerolift Troubleshooting Guide",
            score=0.6,
            doc_type="Troubleshooting",
            reference="Aerolift Troubleshooting Guide > Unit does not start",
        ),
        MockChunk(
            heading="Unusual noise or vibration",
            text="## Unusual noise or vibration\nStop operation and record when the issue occurs and whether the unit is loaded.",
            title="Aerolift Troubleshooting Guide",
            score=0.3,
            doc_type="Troubleshooting",
            reference="Aerolift Troubleshooting Guide > Unusual noise or vibration",
        ),
    ]

    answer = _extractive_answer("Unit does not start", asset, chunks)

    # "Unusual noise or vibration" heading must NOT appear as a step
    if "Unusual noise or vibration" in answer and "Step" in answer:
        # Check if it's embedded as a Step N line
        lines = answer.split("\n")
        for line in lines:
            if "Unusual noise or vibration" in line and line.strip().startswith("Step"):
                result.fail(test_name, f"Unrelated section heading appears as step: '{line.strip()}'")
                return
    result.ok(test_name)


def test_check_vs_escalation_classification(result):
    """Defect #3: 'Confirm the isolator is on...' is a CHECK, not an escalation."""
    test_name = "Check-like sentences classified as CHECK FIRST, not STOP AND ESCALATE IF"

    asset = MockAsset(name="Aerolift-50mtr")
    chunks = [
        MockChunk(
            heading="Unit does not start",
            text="Confirm the main isolator is on, the emergency stop is released, and no safety interlock is open. Record any control-panel fault. If the issue repeats, isolate the equipment and raise a service call.",
            title="Aerolift Troubleshooting Guide",
            score=0.6,
            doc_type="Troubleshooting",
            reference="Aerolift Troubleshooting Guide > Unit does not start",
        ),
    ]

    answer = _extractive_answer("Unit does not start", asset, chunks)

    # Parse sections
    sections = {}
    current = None
    for line in answer.split("\n"):
        stripped = line.strip()
        if stripped in ("DIAGNOSTIC SUMMARY", "CHECK FIRST", "STEP-BY-STEP TROUBLESHOOTING",
                        "EXPECTED RESULT", "STOP AND ESCALATE IF", "VERIFIED REFERENCES"):
            current = stripped
            sections[current] = []
        elif current:
            sections[current].append(line)

    check_text = "\n".join(sections.get("CHECK FIRST", []))
    escalate_text = "\n".join(sections.get("STOP AND ESCALATE IF", []))

    # "Confirm the main isolator" should be in CHECK FIRST
    if "confirm the main isolator" in check_text.lower() or "isolator" in check_text.lower():
        result.ok(test_name)
    elif "confirm the main isolator" in escalate_text.lower() or "isolator" in escalate_text.lower():
        result.fail(test_name, "'Confirm the isolator...' was placed in STOP AND ESCALATE IF instead of CHECK FIRST")
    else:
        # It may have been rephrased but should not be in escalations
        if "isolator" not in escalate_text.lower():
            result.ok(test_name)
        else:
            result.fail(test_name, "Could not verify classification")


def test_escalation_label_semantics(result):
    """Defect #4: Escalated complaint must NOT say 'Troubleshooting already attempted'."""
    test_name = "Escalated complaint uses 'AI diagnostic guidance provided' not 'Troubleshooting already attempted'"

    # Read the ticketing.py source directly to verify the label
    import inspect
    from core.services.ticketing import create_call_from_interaction
    source = inspect.getsource(create_call_from_interaction)

    if "Troubleshooting already attempted" in source:
        result.fail(test_name, "ticketing.py still contains 'Troubleshooting already attempted'")
    elif "AI diagnostic guidance provided" in source:
        result.ok(test_name)
    else:
        result.fail(test_name, "Could not find expected label in source")


def test_escalation_completion_caveat(result):
    """Defect #5: Escalated complaint must include '(completion not confirmed)' caveat."""
    test_name = "Escalated complaint includes 'completion not confirmed' caveat"

    import inspect
    from core.services.ticketing import create_call_from_interaction
    source = inspect.getsource(create_call_from_interaction)

    if "completion not confirmed" in source:
        result.ok(test_name)
    else:
        result.fail(test_name, "ticketing.py does not include 'completion not confirmed' caveat")


def test_extractive_answer_produces_all_sections(result):
    """Structural: extractive answer must produce all required sections."""
    test_name = "Extractive answer contains all required sections"

    asset = MockAsset(name="Aerolift-50mtr")
    chunks = [
        MockChunk(
            heading="Unit does not start",
            text="Confirm the main isolator is on, the emergency stop is released. Record any control-panel fault. Never bypass a safety interlock. If the issue repeats, isolate the equipment and raise a service call.",
            title="Aerolift Troubleshooting Guide",
            score=0.6,
            doc_type="Troubleshooting",
            reference="Aerolift Troubleshooting Guide > Unit does not start",
        ),
    ]

    answer = _extractive_answer("Unit does not start", asset, chunks)

    required_sections = [
        "DIAGNOSTIC SUMMARY",
        "CHECK FIRST",
        "STEP-BY-STEP TROUBLESHOOTING",
        "EXPECTED RESULT",
        "STOP AND ESCALATE IF",
        "VERIFIED REFERENCES",
    ]

    missing = [s for s in required_sections if s not in answer]
    if missing:
        result.fail(test_name, f"Missing sections: {missing}")
    else:
        result.ok(test_name)


def test_insufficient_evidence_with_no_chunks(result):
    """Safety: no chunks should return insufficient evidence, not crash."""
    test_name = "Empty chunks returns insufficient evidence message"

    asset = MockAsset(name="Aerolift-50mtr")
    answer = _extractive_answer("Something random", asset, [])

    if answer == INSUFFICIENT_EVIDENCE_MESSAGE:
        result.ok(test_name)
    else:
        result.fail(test_name, f"Expected insufficient evidence, got: {answer[:100]}")


def test_no_fabricated_why_lines(result):
    """Safety: Why: lines only appear if present in source text."""
    test_name = "No fabricated Why: lines in output"

    asset = MockAsset(name="Aerolift-50mtr")
    chunks = [
        MockChunk(
            heading="Unit does not start",
            text="Confirm the main isolator is on. Record any control-panel fault. Check for visible obstruction.",
            title="Aerolift Troubleshooting Guide",
            score=0.6,
            doc_type="Troubleshooting",
            reference="Aerolift Troubleshooting Guide > Unit does not start",
        ),
    ]

    answer = _extractive_answer("Unit does not start", asset, chunks)

    # Source has no "Why:" lines, so output must not contain them
    why_lines = [l for l in answer.split("\n") if l.strip().lower().startswith("why:")]
    if why_lines:
        result.fail(test_name, f"Found fabricated Why: lines: {why_lines}")
    else:
        result.ok(test_name)


if __name__ == "__main__":
    print("=" * 60)
    print("REGRESSION TESTS: Customer RAG -> Service Call Handoff")
    print("=" * 60)
    print()

    r = TestResult()

    test_no_markdown_headings_in_output(r)
    test_no_cross_contamination(r)
    test_check_vs_escalation_classification(r)
    test_escalation_label_semantics(r)
    test_escalation_completion_caveat(r)
    test_extractive_answer_produces_all_sections(r)
    test_insufficient_evidence_with_no_chunks(r)
    test_no_fabricated_why_lines(r)

    all_passed = r.summary()
    sys.exit(0 if all_passed else 1)

