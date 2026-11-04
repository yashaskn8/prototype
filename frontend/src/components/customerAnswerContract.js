export const answerTitles = [
  'What could be wrong?', 'What can I safely check or report?',
  'What is still unknown?', 'What should I do next?',
];

export function validateCustomerPresentation(value) {
  if (!value || value.schema_version !== 'customer-answer-v1') return null;
  const hazard = value.hazard?.severity === 'immediate' && typeof value.hazard.message === 'string'
    ? { severity: 'immediate', message: value.hazard.message.slice(0, 1200) } : null;
  const invalid = () => hazard ? { sections: [], citations: [], hazard, message: 'Additional guidance is unavailable. Please seek technician assistance.' } : null;
  if (value.response_type !== 'DIAGNOSTIC_GUIDANCE') {
    const message = typeof value.message === 'string' ? value.message.slice(0, 1200) : 'No validated answer is available.';
    const warning = Array.isArray(value.supporting_claims) && value.supporting_claims.length === 1
      ? value.supporting_claims[0] : null;
    const citation = warning?.kind === 'safety_warning' && typeof warning.text === 'string'
      && message.includes(warning.text) && Array.isArray(warning.evidence_ids)
      && warning.evidence_ids.length === 1 && Array.isArray(value.citations)
      ? value.citations.find(s => s?.id === warning.evidence_ids[0]) : null;
    const safeCitation = citation && typeof citation.url === 'string'
      && (citation.url.startsWith('/api/customer-support/source/?citation=')
        || citation.url.startsWith('/demo-api/source/?citation='))
      && typeof citation.title === 'string' ? citation : null;
    return { sections: [], citations: safeCitation ? [safeCitation] : [], hazard, message };
  }
  if (!Array.isArray(value.sections) || value.sections.length !== 4 || !Array.isArray(value.citations) || value.citations.length > 30) return invalid();
  const citations = value.citations.filter(s => s && typeof s.id === 'string' && typeof s.url === 'string'
    && (s.url.startsWith('/api/customer-support/source/?citation=')
      || s.url.startsWith('/demo-api/source/?citation=')) && typeof s.title === 'string');
  if (new Set(citations.map(s => s.id)).size !== citations.length) return invalid();
  const ids = new Set(citations.map(s => s.id));
  const kinds = new Set(['possible_cause', 'information', 'document_fact', 'observation', 'customer_procedure', 'uncertainty', 'next_step']);
  for (const [i, section] of value.sections.entries()) {
    const validTitle = section?.title === answerTitles[i] || (i === 0 && section?.title === 'What the document says');
    if (!section || section.number !== i + 1 || !validTitle || !Array.isArray(section.claims)
      || !section.claims.length || section.claims.length > 4) return invalid();
    for (const claim of section.claims) {
      if (!claim || typeof claim.text !== 'string' || !claim.text.trim() || claim.text.length > 1000 || !kinds.has(claim.kind)
        || !Array.isArray(claim.evidence_ids) || claim.evidence_ids.some(id => !ids.has(id))
        || (!['uncertainty', 'next_step'].includes(claim.kind) && !claim.evidence_ids.length)) return invalid();
    }
  }
  const used = new Set(value.sections.flatMap(s => s.claims.flatMap(c => c.evidence_ids)));
  return { sections: value.sections, citations: citations.filter(s => used.has(s.id)), hazard, message: '' };
}
