import React from 'react';
import './RAGExecutionLog.css';

const STATUS_LABELS = {
  started: 'Running',
  completed: 'Complete',
  failed: 'Failed',
  skipped: 'Skipped',
  info: 'Info',
};

function metricLabels(metrics = {}) {
  const labels = [];
  if (Number.isInteger(metrics.candidate_count)) labels.push(`Candidates: ${metrics.candidate_count}`);
  if (Number.isInteger(metrics.dense_count)) labels.push(`Vector hits: ${metrics.dense_count}`);
  if (Number.isInteger(metrics.sparse_count)) labels.push(`Text hits: ${metrics.sparse_count}`);
  if (Number.isInteger(metrics.retrieved_count)) labels.push(`Retrieved: ${metrics.retrieved_count}`);
  if (Number.isInteger(metrics.supported_count)) labels.push(`Supported: ${metrics.supported_count}`);
  if (Number.isInteger(metrics.past_case_count)) labels.push(`Past cases: ${metrics.past_case_count}`);
  if (metrics.retrieval_backend) labels.push(`Search: ${metrics.retrieval_backend}`);
  if (metrics.reranker_applied === true) labels.push('Reranker applied');
  if (metrics.reranker_applied === false) labels.push('Reranker not applied');
  if (metrics.reranker_model) labels.push(`Reranker: ${metrics.reranker_model}`);
  if (metrics.confidential_scope) labels.push(
    metrics.confidential_scope === 'permitted' ? 'Confidential scope permitted' : 'Confidential scope restricted'
  );
  if (metrics.answer_engine) labels.push(`Answer engine: ${metrics.answer_engine}`);
  if (metrics.model_provider) {
    labels.push(`Generation: ${metrics.model_provider}`);
    if (metrics.generation_status) labels.push(`Outcome: ${metrics.generation_status}`);
  }
  return labels;
}

export function RAGExecutionLog({ events = [], running = false, title = 'Live RAG execution' }) {
  if (!running && events.length === 0) return null;
  return (
    <section className="rag-execution-log" aria-label={title}>
      <div className="rag-execution-log-header">
        <strong>{title}</strong>
        <span className={running ? 'rag-execution-log-live' : 'rag-execution-log-finished'}>
          {running ? 'LIVE' : 'FINISHED'}
        </span>
      </div>
      <ol className="rag-execution-log-list" role="log" aria-live="polite" aria-relevant="additions text">
        {events.map((event, index) => {
          const status = event.status || 'info';
          const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';
          const metrics = metricLabels(event.metrics);
          return (
            <li className={`rag-execution-log-item is-${status}`} key={event.event_id || `${event.stage}-${index}`}>
              <span className="rag-execution-log-status" aria-hidden="true" />
              <div className="rag-execution-log-content">
                <div className="rag-execution-log-line">
                  <strong>{event.label || event.stage}</strong>
                  <span>{STATUS_LABELS[status] || status}</span>
                </div>
                <div className="rag-execution-log-meta">
                  {time && <time dateTime={event.timestamp}>{time}</time>}
                  {Number.isFinite(event.duration_ms) && <span>{event.duration_ms.toFixed(1)} ms</span>}
                  {event.error_type && <span>Failure: {event.error_type}</span>}
                </div>
                {metrics.length > 0 && <div className="rag-execution-log-metrics">{metrics.join(' · ')}</div>}
              </div>
            </li>
          );
        })}
        {running && <li className="rag-execution-log-waiting"><span className="rag-execution-log-pulse" />Waiting for the next backend event…</li>}
      </ol>
      <p className="rag-execution-log-note">Stages and timings are emitted by the running backend request. Source text and private prompts are not included.</p>
    </section>
  );
}
