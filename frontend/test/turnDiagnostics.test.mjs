import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { diagnosticsSummary } from '../src/components/turnDiagnostics.js';

test('missing diagnostics are not presented as zero retrieval passes', () => {
  const text = diagnosticsSummary({ retrieval_backend: 'hybrid' });
  assert.equal(text, 'Execution details were not reported for this answer.');
  assert.doesNotMatch(text, /0 pass/);
});

test('a turn without retrieval says retrieval did not run', () => {
  const text = diagnosticsSummary({ diagnostics: {
    retrieval: { passes: 0, dense_candidates: null, sparse_candidates: null, retrieved: null },
    cited_evidence: [], rejection_reasons: [], qwen: { ran: false, status: 'skipped' },
  } });
  assert.match(text, /Retrieval: not run for this message\./);
  assert.match(text, /Qwen: skipped; exact source wording shown\./);
});

test('real counts, cited evidence, limitations and Qwen acceptance are shown', () => {
  const text = diagnosticsSummary({ diagnostics: {
    retrieval: { passes: 3, dense_candidates: 12, sparse_candidates: 9, retrieved: 8,
      backends: ['hybrid'], reranker_applied: true },
    cited_evidence: [{ id: 's1' }, { id: 's2' }],
    rejection_reasons: ['cause_without_explaining_evidence', 'cause_without_explaining_evidence'],
    qwen: { ran: true, status: 'applied', eligible_claims: 4, accepted_claims: 3 },
  } });
  assert.match(text, /Retrieval: 3 pass\(es\) via hybrid; vector candidates 12, text candidates 9, retrieved 8; reranking applied\./);
  assert.match(text, /Cited evidence: 2 passage\(s\)\./);
  assert.match(text, /Limitations: cause without explaining evidence\./);
  assert.match(text, /Qwen: applied; accepted 3 of 4 claim\(s\)\./);
});

test('the demo page no longer defaults retrieval passes to zero', () => {
  const page = readFileSync(new URL('../src/pages/RealDemoPage.jsx', import.meta.url), 'utf8');
  assert.equal(page.includes('retrieval_attempts ?? 0'), false);
  assert.equal(page.includes('diagnosticsSummary(message.data)'), true);
});
