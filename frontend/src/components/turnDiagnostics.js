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

// One short, nontechnical line saying how this answer was produced. It reflects
// only the server-reported outcome; nothing is labelled generated or cited unless
// the server says so for this request.
const GENERATION_REASONS = {
  timeout: 'the wording step ran out of time',
  deadline_skipped: 'there was not enough time left for the wording step',
  ollama_unreachable: 'the local wording model was unavailable',
  request_failed: 'the local wording model was unavailable',
  rejected_by_source_validator: 'the reworded text did not match the manual closely enough',
  invalid_model_json: 'the local wording model returned an unusable reply',
  invalid_model_response: 'the local wording model returned an unusable reply',
};

export function outcomeLabel(data) {
  const o = data?.outcome;
  if (!o || typeof o !== 'object') return '';
  if (o.status === 'success' || o.status === 'degraded') {
    if (o.response_mode === 'grounded_generated') {
      return o.status === 'success'
        ? 'Answer worded by the local assistant from the cited manual sentences.'
        : 'Answer worded by the local assistant from the cited manual sentences (reduced search ranking was used to stay fast).';
    }
    const why = GENERATION_REASONS[o.generation_status];
    return why
      ? `Exact cited manual wording is shown because ${why}.`
      : 'Exact cited manual wording is shown.';
  }
  if (o.status === 'insufficient_evidence') {
    return 'No answer from the approved manuals could be confirmed for this question. Please describe the symptom in more detail or request a technician.';
  }
  return '';
}

// The server runs one answer at a time and refuses another with HTTP 503
// (code rag_busy). Show that plainly instead of a generic failure; the
// question stays in the input so the customer can resend it.
export function busyMessage(err) {
  return err?.status === 503
    ? 'Servy is still answering your previous question. Please wait for it to finish, then send this again.'
    : '';
}
