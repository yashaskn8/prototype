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


test('rejects informational claims placed under documented causes', () => {
  const presentation = mixedPresentation();
  presentation.sections[0].claims[0].kind = 'information';
  assert.equal(validateCustomerPresentation(presentation), null);
});

test('rejects causes, procedures, and observations placed under the wrong headings', () => {
  const cases = [
    ['Safe observations from the manual', 'possible_cause'],
    ['Approved customer checks', 'possible_cause'],
    ['Approved customer checks', 'next_step'],
    ['What I could not verify', 'customer_procedure'],
    ['Evidence limitation', 'information'],
  ];
  for (const [title, kind] of cases) {
    const presentation = mixedPresentation();
    presentation.sections = [{
      number: 1, title,
      claims: [{ text: 'Misclassified source claim.', kind, evidence_ids: ['s1'] }],
    }];
    assert.equal(validateCustomerPresentation(presentation), null, title);
  }
});

test('accepts only customer-safe claim kinds in observation and check sections', () => {
  for (const [title, kind] of [
    ['Safe observations from the manual', 'observation'],
    ['Safe observations from the manual', 'customer_procedure'],
    ['Approved customer checks', 'observation'],
    ['Approved customer checks', 'customer_procedure'],
  ]) {
    const presentation = mixedPresentation();
    presentation.sections = [{
      number: 1, title,
      claims: [{
        text: 'Synthetic customer-safe source statement.',
        kind, evidence_ids: ['s1'],
      }],
    }];
    assert.ok(validateCustomerPresentation(presentation), title + ' / ' + kind);
  }
});