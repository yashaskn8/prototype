import test from 'node:test';
import assert from 'node:assert/strict';
import { validateCustomerPresentation } from '../src/components/customerAnswerContract.js';

function makePresentation(overrides = {}) {
  return {
    schema_version: 'customer-answer-v1',
    response_type: 'DIAGNOSTIC_GUIDANCE',
    view_mode: 'focused',
    equipment: { id: 60 },
    sections: [{
      number: 1,
      title: 'Documented possible causes',
      claims: [{
        id: 'c1', text: 'Possible documented cause: operating conditions.',
        kind: 'possible_cause', evidence_ids: ['s1'],
      }],
    }],
    citations: [{
      id: 's1', title: 'Approved manual',
      url: '/demo-api/source/?citation=abc123token',
    }],
    hazard: { severity: 'none', message: '' },
    online_research: {
      status: 'available',
      note: 'External excerpts are supplementary only.',
      sources: [{
        title: 'Manufacturer support',
        url: 'https://docs.example.test/support/model',
        hostname: 'docs.example.test',
        age: 'Today',
        match_basis: 'exact_model_and_fault_code',
        snippets: ['For model VRF-OUT-60, code E-04 indicates a reported state.'],
      }],
    },
    document_coverage: {
      complete: true, documents_searched: 3, approved_documents: 3,
      documents_without_searchable_chunks: 0, passages_searched: 42,
      documents_with_matches: 2, note: 'Three approved manuals were searched.',
    },
    ...overrides,
  };
}

test('keeps online evidence separate and validates document coverage', () => {
  const value = validateCustomerPresentation(makePresentation());
  assert.equal(value.onlineResearch.label, 'Supplementary online information — not company-approved');
  assert.equal(value.onlineResearch.sources.length, 1);
  assert.equal(value.onlineResearch.sources[0].matchBasis, 'exact_model_and_fault_code');
  assert.equal(value.documentCoverage.complete, true);
  assert.equal(value.documentCoverage.documentsSearched, 3);
});

test('downgrades coverage when an approved document has no searchable chunks', () => {
  const input = makePresentation({
    document_coverage: {
      complete: true, documents_searched: 2, approved_documents: 3,
      documents_without_searchable_chunks: 1, passages_searched: 20,
      documents_with_matches: 2, note: 'One manual has no searchable chunks.',
    },
  });
  assert.equal(validateCustomerPresentation(input).documentCoverage.complete, false);
});

test('rejects non-HTTPS URLs and hostname mismatches in external sources', () => {
  const input = makePresentation({
    online_research: {
      status: 'available', note: 'Only approved domains are accepted.',
      sources: [
        {
          title: 'Unsafe scheme', url: 'javascript:alert(1)', hostname: 'docs.example.test',
          match_basis: 'exact_model', snippets: ['This is a sufficiently long external source snippet.'],
        },
        {
          title: 'Hostname mismatch', url: 'https://evil.example.test/path',
          hostname: 'docs.example.test', match_basis: 'exact_model',
          snippets: ['This is a sufficiently long external source snippet.'],
        },
      ],
    },
  });
  assert.equal(validateCustomerPresentation(input).onlineResearch.sources.length, 0);
});

test('preserves labelled external excerpts on insufficient-evidence answers', () => {
  const input = makePresentation({
    response_type: 'INSUFFICIENT_EVIDENCE',
    view_mode: undefined,
    sections: [],
    citations: [],
    message: 'No company-manual answer was verified.',
  });
  const value = validateCustomerPresentation(input);
  assert.equal(value.message, 'No company-manual answer was verified.');
  assert.equal(value.onlineResearch.sources.length, 1);
});
