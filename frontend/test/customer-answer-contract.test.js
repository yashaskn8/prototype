import assert from 'node:assert/strict';
import test from 'node:test';

import { validateCustomerPresentation } from '../src/components/customerAnswerContract.js';

const mixedPresentation = () => ({
  schema_version: 'customer-answer-v1',
  response_type: 'DIAGNOSTIC_GUIDANCE',
  view_mode: 'focused',
  equipment: { id: 59 },
  citations: [{
    id: 's1',
    title: 'Synthetic approved manual',
    heading: 'Synthetic troubleshooting section',
    url: '/demo-api/source/?citation=synthetic-evidence-123',
  }],
  sections: [
    { number: 1, title: 'Documented possible causes', claims: [{
      text: 'Possible documented cause: synthetic source claim.',
      kind: 'possible_cause', evidence_ids: ['s1'],
    }] },
    { number: 2, title: 'What I could not verify', claims: [{
      text: 'No customer-safe check is verified.',
      kind: 'uncertainty', evidence_ids: [],
    }] },
  ],
  message: 'Supported claims and evidence limits are separated.',
  follow_up_question: null,
});

test('accepts multi-section focused guidance with a cited fact and uncited limitation', () => {
  const result = validateCustomerPresentation(mixedPresentation());
  assert.ok(result);
  assert.equal(result.focused, true);
  assert.equal(result.sections.length, 2);
  assert.equal(result.citations.length, 1);
});

test('rejects a focused technical claim without exactly one trusted citation', () => {
  const presentation = mixedPresentation();
  presentation.sections[0].claims[0].evidence_ids = [];
  assert.equal(validateCustomerPresentation(presentation), null);
});

test('rejects evidence citations attached to an uncertainty-only section', () => {
  const presentation = mixedPresentation();
  presentation.sections[1].claims[0].evidence_ids = ['s1'];
  assert.equal(validateCustomerPresentation(presentation), null);
});
