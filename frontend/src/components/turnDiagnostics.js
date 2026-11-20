// Execution facts reported by the server for this turn. A stage that did not
// run is described as not run; no count is shown as zero by default.
export function diagnosticsSummary(data) {
  const d = data?.diagnostics;
  if (!d || typeof d !== 'object') return 'Execution details were not reported for this answer.';
  const r = d.retrieval || {};
  const count = value => (Number.isInteger(value) ? value : null);
  const parts = [];
  if (count(r.passes)) {
    parts.push(`Retrieval: ${r.passes} pass(es)${Array.isArray(r.backends) && r.backends.length ? ' via ' + r.backends.join(', ') : ''}`
      + `; vector candidates ${count(r.dense_candidates) ?? 'n/a'}, text candidates ${count(r.sparse_candidates) ?? 'n/a'}, `
      + `retrieved ${count(r.retrieved) ?? 'n/a'}; reranking ${r.reranker_applied ? 'applied' : 'not applied'}.`);
  } else {
    parts.push('Retrieval: not run for this message.');
  }
  const cited = Array.isArray(d.cited_evidence) ? d.cited_evidence : [];
  parts.push(`Cited evidence: ${cited.length} passage(s).`);
  const reasons = Array.isArray(d.rejection_reasons) ? d.rejection_reasons.filter(v => typeof v === 'string') : [];
  if (reasons.length) parts.push(`Limitations: ${[...new Set(reasons)].join(', ').replaceAll('_', ' ')}.`);
  const q = d.qwen || {};
  parts.push(q.ran
    ? `Qwen: ${q.status}; accepted ${count(q.accepted_claims) ?? 0} of ${count(q.eligible_claims) ?? 0} claim(s).`
    : `Qwen: ${q.status === 'skipped' ? 'skipped' : 'not run'}; exact source wording shown.`);
  return parts.join(' ');
}
