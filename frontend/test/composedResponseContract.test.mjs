import test from 'node:test';
import assert from 'node:assert/strict';
import { validateCustomerPresentation } from '../src/components/customerAnswerContract.js';
import { fixture } from './customerAnswerContract.test.mjs';

const composed = () => ({
  segments: [
    { kind: 'text', text: 'For poor cooling on the Split AC Unit 2T, the manual lists this possible cause:', citation_ids: [] },
    { kind: 'list_item', text: 'Restricted airflow.', citation_ids: ['a'] },
    { kind: 'question', text: 'Do any of these match what you are seeing?', citation_ids: [] },
  ],
});

test('a well-formed composed reply is accepted alongside the sections', () => {
  const data = validateCustomerPresentation({ ...fixture(), response: composed() });
  assert.equal(data.response.segments.length, 3);
  assert.equal(data.sections.length, 4);
});

test('a technical segment must cite exactly one evidence id used by a validated claim', () => {
  for (const ids of [[], ['a', 'a'], ['unknown']]) {
    const response = composed();
    response.segments[1].citation_ids = ids;
    assert.equal(validateCustomerPresentation({ ...fixture(), response }).response, null, JSON.stringify(ids));
  }
});

test('connective text may not carry citations and unknown kinds are refused', () => {
  const cited = composed();
  cited.segments[0].citation_ids = ['a'];
  assert.equal(validateCustomerPresentation({ ...fixture(), response: cited }).response, null);
  const unknown = composed();
  unknown.segments[0].kind = 'diagnosis';
  assert.equal(validateCustomerPresentation({ ...fixture(), response: unknown }).response, null);
});

test('oversized or empty text falls back to the sections', () => {
  const long = composed();
  long.segments[0].text = 'x'.repeat(601);
  assert.equal(validateCustomerPresentation({ ...fixture(), response: long }).response, null);
  assert.equal(validateCustomerPresentation({ ...fixture(), response: { segments: [] } }).response, null);
  assert.equal(validateCustomerPresentation(fixture()).response, null);
});
