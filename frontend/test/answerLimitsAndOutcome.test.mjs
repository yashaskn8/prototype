import test from 'node:test';
import assert from 'node:assert/strict';
import { MAX_SECTION_CLAIMS, validateCustomerPresentation } from '../src/components/customerAnswerContract.js';
import { outcomeLabel } from '../src/components/turnDiagnostics.js';
import { fixture } from './customerAnswerContract.test.mjs';

function withCauses(n) {
  const p = fixture();
  p.citations = Array.from({ length: n }, (_, i) => ({
    id: `s${i + 1}`, title: 'Split AC Unit 2T - Troubleshooting', heading: '2. SYMPTOM: UNIT DOES NOT START',
    url: `/api/customer-support/source/?citation=signed-${String(i + 1).padStart(4, '0')}`,
  }));
  p.sections[0].claims = p.citations.map((c, i) => ({
    text: `Possible documented cause: cause ${i + 1}.`, kind: 'possible_cause', evidence_ids: [c.id],
  }));
  return p;
}

test('a section at the shared limit of 12 cited causes is accepted', () => {
  assert.equal(MAX_SECTION_CLAIMS, 12);
  for (const n of [1, 12]) {
    const v = validateCustomerPresentation(withCauses(n));
    assert.ok(v, `${n} causes`);
    assert.equal(v.sections[0].claims.length, n);
    assert.equal(v.citations.length, n);
  }
});

test('more claims than the limit, or a claim citing a missing source, is rejected', () => {
  for (const n of [13, 20, 21]) assert.equal(validateCustomerPresentation(withCauses(n)), null, `${n} causes`);
  const bad = withCauses(3);
  bad.sections[0].claims[1].evidence_ids = ['s99'];
  assert.equal(validateCustomerPresentation(bad), null);
});

test('customer chat labels only unconfirmed answers; generation fallbacks stay internal', async () => {
  const { generationFallbackNote } = await import('../src/components/turnDiagnostics.js');
  assert.equal(outcomeLabel({}), '');
  for (const status of ['success', 'degraded']) {
    for (const response_mode of ['grounded_generated', 'grounded_extractive']) {
      const label = outcomeLabel({ outcome: { status, response_mode, generation_status: 'timeout' } });
      assert.equal(label, '');
    }
  }
  // The truthful fallback reason is still derivable for diagnostics views.
  assert.equal(generationFallbackNote({ outcome: { status: 'degraded', response_mode: 'grounded_extractive', generation_status: 'timeout' } }),
    'Source wording used because the wording step ran out of time.');
  assert.equal(generationFallbackNote({ outcome: { status: 'success', response_mode: 'grounded_generated' } }), '');
  assert.match(outcomeLabel({ outcome: { status: 'insufficient_evidence', response_mode: 'no_validated_evidence' } }),
    /No answer from the approved manuals could be confirmed/);
  assert.equal(outcomeLabel({ outcome: { status: 'safety_escalation' } }), '');
  assert.equal(outcomeLabel({ outcome: { status: 'conversational' } }), '');
});

test('a busy (503) refusal is shown as busy, other failures are not', async () => {
  const { busyMessage } = await import('../src/components/turnDiagnostics.js');
  assert.match(busyMessage({ status: 503 }), /still answering your previous question/);
  assert.equal(busyMessage({ status: 500 }), '');
  assert.equal(busyMessage(undefined), '');
});
