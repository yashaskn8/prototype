import React, { useEffect, useId, useRef, useState } from 'react';
import { api } from '../api';
import { validateCustomerPresentation } from './customerAnswerContract';
import './CustomerAnswer.css';

export function CustomerAnswer(props) {
  return <AnswerContent key={JSON.stringify([props.assetId, props.presentation])} {...props} />;
}

function AnswerContent({ presentation, assetId, loadSource = url => api.get(url) }) {
  const data = validateCustomerPresentation(presentation);
  const [source, setSource] = useState(null);
  const [unable, setUnable] = useState(false);
  const dialog = useRef(null);
  const trigger = useRef(null);
  const revision = useRef(0);
  const titleId = useId();
  useEffect(() => () => { revision.current += 1; }, []);
  useEffect(() => {
    if (source && dialog.current && !dialog.current.open) dialog.current.showModal();
  }, [source]);
  function close() {
    revision.current += 1;
    dialog.current?.close();
    setSource(null);
    trigger.current?.focus();
  }
  function containFocus(event) {
    if (event.key !== 'Tab') return;
    const controls = [...dialog.current.querySelectorAll('button, a[href], [tabindex="0"]')]
      .filter(control => !control.disabled);
    const first = controls[0], last = controls.at(-1);
    if (!first) return;
    if ((event.shiftKey && document.activeElement === first)
      || (!event.shiftKey && document.activeElement === last)) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    }
  }
  async function open(event, citation) {
    trigger.current = event.currentTarget;
    const current = ++revision.current;
    setSource({ loading: true, citation });
    try {
      // Always re-fetch. Never cache an excerpt across permission changes.
      const result = await loadSource(citation.url);
      if (revision.current !== current) return;
      if (typeof result?.excerpt !== 'string' || typeof result.title !== 'string') throw new Error('Invalid source');
      setSource({ ...result, loading: false, citation });
    } catch {
      if (revision.current === current) setSource({ error: 'This source is unavailable or access has changed. Ask the question again.', citation });
    }
  }
  if (!data || (presentation.equipment?.id && String(presentation.equipment.id) !== String(assetId))) {
    return <p role="status">No validated answer is available for the selected equipment. Please ask again.</p>;
  }
  return <section className="customer-answer" aria-label="Equipment guidance">
    {data.hazard && <div className="customer-answer-hazard" role="alert"><strong>Immediate safety warning</strong><p>{data.hazard.message}</p></div>}
    {data.message && <p role="status">{data.message}{!data.sections.length && data.citations.map((citation, index) =>
      <button type="button" className="customer-answer-citation" key={citation.id}
        title={`Open source ${index + 1}`}
        aria-label={`Source ${index + 1}: ${citation.title}${citation.heading ? `, ${citation.heading}` : ''}`}
        onClick={event => open(event, citation)}>[{index + 1}]</button>)}</p>}
    {!!data.sections.length && <div className="customer-answer-points">
      {data.sections.map(section => <section className="customer-answer-point" key={section.number}>
        {!data.focused && <h3><span aria-hidden="true">{section.number}</span>{section.title}</h3>}
        {section.claims.map((claim, index) => <div className="customer-answer-claim" key={claim.id || index}>
          <p className="customer-answer-source-claim">
            {claim.text}
          {[...new Set(claim.evidence_ids)].map(id => {
            const citation = data.citations.find(s => s.id === id);
            const number = data.citations.indexOf(citation) + 1;
            return <button type="button" className="customer-answer-citation" key={id}
              title={`Open source ${number}`}
              aria-label={`Source ${number}: ${citation.title}${citation.heading ? `, ${citation.heading}` : ''}`}
              onClick={event => open(event, citation)}>[{number}]</button>;
          })}
          </p>
        </div>)}
      </section>)}
    </div>}

    {data.onlineResearch && <section className="customer-answer-online" aria-label="Supplementary online information">
      <h3>{data.onlineResearch.label}</h3>
      <p className="customer-answer-online-note">{data.onlineResearch.note}</p>
      {data.onlineResearch.sources.map(source => <article className="customer-answer-online-source" key={source.url}>
        <div><a href={source.url} target="_blank" rel="noopener noreferrer">{source.title}</a>
          <span>{source.hostname}{source.age ? ' · ' + source.age : ''}</span></div>
        {source.snippets.map((snippet, index) => <blockquote key={index}>{snippet}</blockquote>)}
      </article>)}
    </section>}
    {data.followUpQuestion && <p className="customer-answer-followup">{data.followUpQuestion}</p>}
    {!!data.sections.length && Array.isArray(presentation.customer_authorized_actions)
      && presentation.customer_authorized_actions.length > 0
      && <button type="button" className="btn btn-secondary" onClick={() => setUnable(true)}>I cannot perform a check</button>}
    {unable && <p role="status">You can report that limitation to a technician. No service ticket has been created.</p>}
    {source && <dialog ref={dialog} className="customer-answer-source" aria-labelledby={titleId} onKeyDown={containFocus} onCancel={event => { event.preventDefault(); close(); }}>
      <h3 id={titleId}>Supporting source</h3>
      <button type="button" className="btn btn-secondary" autoFocus onClick={close}>Close source</button>
      {source.loading ? <p role="status">Loading the authorized passage…</p> : source.error ? <p role="alert">{source.error}</p> : <>
        <h4>{source.title}</h4>
        {source.heading && <p>{source.heading}</p>}
        <blockquote>{source.excerpt}</blockquote>
      </>}
    </dialog>}
  </section>;
}
