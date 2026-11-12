export const answerTitles = [
  'What could be wrong?', 'What can I safely check or report?',
  'What is still unknown?', 'What should I do next?',
];

export const focusedAnswerTitles = [
  'Documented possible causes', 'Safe observations from the manual',
  'Approved customer checks', 'What the manual says',
  'What the document says', 'What the manual supports',
  'What I could not verify', 'Evidence limitation',
];

// URL shape is client-side defense in depth. The server still verifies the
// signed citation, source hash, permissions, tenant and selected asset.
export function isTrustedSourceUrl(url) {
  return typeof url === 'string'
    && /^\/(?:api\/customer-support|demo-api)\/source\/\?citation=[A-Za-z0-9._:-]{8,8192}$/.test(url);
}

function validatedOnlineResearch(value) {
  if (!value || typeof value !== 'object') return null;
  const statuses = new Set(['available', 'not_configured', 'model_identity_incomplete',
    'asset_model_mismatch', 'not_applicable', 'no_exact_match', 'unavailable']);
  if (!statuses.has(value.status)) return null;
  const sources = [];
  if (value.status === 'available' && Array.isArray(value.sources)) {
    for (const source of value.sources.slice(0, 5)) {
      if (!source || typeof source.url !== 'string' || source.url.length > 2048) continue;
      let parsed;
      try { parsed = new URL(source.url); } catch { continue; }
      if (parsed.protocol !== 'https:' || parsed.username || parsed.password
        || !parsed.hostname || source.hostname !== parsed.hostname.toLowerCase()) continue;
      if (!['exact_model', 'exact_model_and_fault_code'].includes(source.match_basis)
        || !Array.isArray(source.snippets)) continue;
      const snippets = source.snippets.slice(0, 3)
        .filter(item => typeof item === 'string' && item.trim().length > 20 && item.length <= 1200);
      if (!snippets.length) continue;
      sources.push({
        title: typeof source.title === 'string' ? source.title.slice(0, 220) : parsed.hostname,
        url: source.url,
        hostname: parsed.hostname.toLowerCase(),
        age: typeof source.age === 'string' ? source.age.slice(0, 100) : '',
        matchBasis: source.match_basis,
        snippets,
      });
    }
  }
  return {
    status: value.status,
    label: 'Supplementary online information — not company-approved',
    note: typeof value.note === 'string' ? value.note.slice(0, 700) : '',
    sources,
  };
}

function validatedDocumentCoverage(value) {
  if (!value || typeof value !== 'object') return null;
  const count = key => Number.isInteger(value[key]) && value[key] >= 0 ? value[key] : null;
  const searched = count('documents_searched');
  const approved = count('approved_documents');
  const missing = count('documents_without_searchable_chunks');
  const passages = count('passages_searched');
  if (searched === null || approved === null || missing === null || passages === null
    || typeof value.complete !== 'boolean') return null;
  const complete = value.complete && searched === approved && missing === 0;
  return {
    complete,
    documentsSearched: searched,
    approvedDocuments: approved,
    documentsWithoutSearchableChunks: missing,
    passagesSearched: passages,
    documentsWithMatches: count('documents_with_matches') ?? 0,
    // false when the scan did not run to completion: its counts are then
    // placeholders and must not be shown as a finding.
    countsAvailable: value.scan_completed !== false,
    note: typeof value.note === 'string' ? value.note.slice(0, 700) : '',
  };
}

export function validateCustomerPresentation(value) {
  if (!value || value.schema_version !== 'customer-answer-v1') return null;
  const onlineResearch = validatedOnlineResearch(value.online_research);
  const documentCoverage = validatedDocumentCoverage(value.document_coverage);
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
      && isTrustedSourceUrl(citation.url)
      && typeof citation.title === 'string' ? citation : null;
    return { sections: [], citations: safeCitation ? [safeCitation] : [], hazard, message, onlineResearch, documentCoverage };
  }
  const focused = value.view_mode === 'focused';
  if (!Array.isArray(value.sections) ||
    (focused ? value.sections.length < 1 || value.sections.length > 6 : value.sections.length !== 4) ||
    !Array.isArray(value.citations) || value.citations.length > 30) return invalid();
  const citations = value.citations.filter(s => s && typeof s.id === 'string' && typeof s.url === 'string'
    && isTrustedSourceUrl(s.url) && typeof s.title === 'string');
  if (new Set(citations.map(s => s.id)).size !== citations.length) return invalid();
  const ids = new Set(citations.map(s => s.id));
  const kinds = new Set(['possible_cause', 'information', 'document_fact', 'observation', 'customer_procedure', 'uncertainty', 'next_step']);
  // Defense in depth: even a server-side classification regression must not
  // render a fact as a cause, an observation as a customer procedure, or a
  // cause under an observation/limitation heading.
  const kindsBySectionTitle = {
    'What could be wrong?': new Set(['possible_cause', 'uncertainty']),
    'What can I safely check or report?': new Set(['observation', 'customer_procedure', 'uncertainty']),
    'What is still unknown?': new Set(['uncertainty']),
    'What should I do next?': new Set(['next_step', 'uncertainty']),
    'Documented possible causes': new Set(['possible_cause']),
    'Safe observations from the manual': new Set(['observation']),
    'Approved customer checks': new Set(['customer_procedure']),
    'What I could not verify': new Set(['uncertainty']),
    'Evidence limitation': new Set(['uncertainty']),
    'What the manual says': new Set(['possible_cause', 'information', 'document_fact']),
    'What the document says': new Set(['possible_cause', 'information', 'document_fact']),
    'What the manual supports': new Set(['information', 'document_fact']),
  };
  for (const [i, section] of value.sections.entries()) {
    const validTitle = focused
      ? focusedAnswerTitles.includes(section?.title)
      : (section?.title === answerTitles[i] || (i === 0 && section?.title === 'What the document says'));
    if (!section || section.number !== i + 1 || !validTitle || !Array.isArray(section.claims)
      || !section.claims.length || section.claims.length > 12) return invalid();
    const allowedKinds = kindsBySectionTitle[section.title];
    for (const claim of section.claims) {
      if (!claim || (allowedKinds && !allowedKinds.has(claim.kind))
        || typeof claim.text !== 'string' || !claim.text.trim() || claim.text.length > 1000 || !kinds.has(claim.kind)
        || (claim.explanation !== undefined && (typeof claim.explanation !== 'string'
          || claim.explanation.trim().length < 35 || claim.explanation.length > 600
          || /[\r\n]/.test(claim.explanation)))
        || !Array.isArray(claim.evidence_ids) || claim.evidence_ids.some(id => !ids.has(id))
        || (focused && (!['possible_cause', 'information', 'document_fact', 'observation', 'customer_procedure', 'uncertainty', 'next_step'].includes(claim.kind)
          || (['uncertainty', 'next_step'].includes(claim.kind)
            ? claim.evidence_ids.length !== 0
            : claim.evidence_ids.length !== 1)))
        || (!['uncertainty', 'next_step'].includes(claim.kind) && !claim.evidence_ids.length)) return invalid();
    }
  }
  const used = new Set(value.sections.flatMap(s => s.claims.flatMap(c => c.evidence_ids)));
  return {
    sections: value.sections,
    citations: citations.filter(s => used.has(s.id)),
    hazard,
    focused,
    message: focused && typeof value.message === 'string' ? value.message.slice(0, 450) : '',
    followUpQuestion: focused && typeof value.follow_up_question === 'string'
      ? value.follow_up_question.slice(0, 240) : null,
    onlineResearch,
    documentCoverage,
  };
}
