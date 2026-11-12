import test from 'node:test';
import assert from 'node:assert/strict';
import { validateCustomerPresentation } from '../src/components/customerAnswerContract.js';

function refusal(coverage) {
  return {
    schema_version: 'customer-answer-v1', response_type: 'INSUFFICIENT_EVIDENCE',
    message: 'No documented answer.', document_coverage: coverage,
  };
}

const zeros = {
  complete: false, documents_searched: 0, approved_documents: 0,
  documents_without_searchable_chunks: 0, passages_searched: 0, note: 'Scan did not finish.',
};

test('an unfinished document scan does not present its placeholder counts', () => {
  const coverage = validateCustomerPresentation(
    refusal({ ...zeros, scan_completed: false })).documentCoverage;
  assert.equal(coverage.countsAvailable, false);
  assert.equal(coverage.complete, false);
  assert.equal(coverage.note, 'Scan did not finish.');
});

test('a finished scan keeps its real counts, including a genuine zero', () => {
  const finished = validateCustomerPresentation(refusal(zeros)).documentCoverage;
  assert.equal(finished.countsAvailable, true);
  const full = validateCustomerPresentation(refusal({
    complete: true, documents_searched: 3, approved_documents: 3,
    documents_without_searchable_chunks: 0, passages_searched: 19, note: 'All searched.',
  })).documentCoverage;
  assert.equal(full.countsAvailable, true);
  assert.equal(full.documentsSearched, 3);
  assert.equal(full.complete, true);
});
