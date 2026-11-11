import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, AlertTriangle, Check, ChevronRight, Clock3, FileText, Inbox, RefreshCw, ShieldCheck, Ticket, UserRound, Wrench } from 'lucide-react';
import { api } from '../api';
import { handoffBucket, handoffLabel } from './technicianHandoffState';
import './TechnicianInboxPage.css';

const POLL_INTERVAL_MS = 10000;

const BUCKETS = [
  { id: 'incoming', label: 'Incoming', icon: Inbox },
  { id: 'acknowledged', label: 'Acknowledged', icon: Check },
  { id: 'in_progress', label: 'In Progress', icon: Wrench },
  { id: 'resolved', label: 'Resolved', icon: ShieldCheck },
];

function formatConfirmedStep(step) {
  if (typeof step === 'string') return step;
  if (!step || typeof step !== 'object') return '';
  return step.text || step.action || step.label || step.description || '';
}

export function TechnicianInboxPage() {
  const [tickets, setTickets] = useState([]);
  const [activeBucket, setActiveBucket] = useState('incoming');
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [acknowledging, setAcknowledging] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const listRequestInFlight = useRef(false);

  const loadInbox = useCallback(async () => {
    if (listRequestInFlight.current) return;
    listRequestInFlight.current = true;
    try {
      const data = await api.get('/api/technician/inbox/');
      setTickets(Array.isArray(data.tickets) ? data.tickets : []);
      setLastUpdated(data.polled_at || new Date().toISOString());
      setError(null);
    } catch (loadError) {
      setError(loadError.detail || 'Could not load the technician inbox. Retry in a moment.');
    } finally {
      listRequestInFlight.current = false;
      setLoading(false);
    }
  }, []);

  const loadTicketDetail = useCallback(async (id) => {
    setDetailLoading(true);
    try {
      const data = await api.get(`/api/technician/inbox/${id}/`);
      setSelectedTicket(data.ticket || null);
    } catch (loadError) {
      setError(loadError.detail || 'Could not open this ticket.');
      setSelectedTicket(null);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    const poll = () => {
      if (document.visibilityState === 'visible') loadInbox();
    };
    const initialLoad = window.setTimeout(poll, 0);
    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    document.addEventListener('visibilitychange', poll);
    return () => {
      window.clearTimeout(initialLoad);
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', poll);
    };
  }, [loadInbox]);

  useEffect(() => {
    if (!selectedTicket?.id) return undefined;
    const timer = window.setInterval(async () => {
      if (document.visibilityState !== 'visible') return;
      try {
        const data = await api.get(`/api/technician/inbox/${selectedTicket.id}/`);
        setSelectedTicket(data.ticket || null);
      } catch {
        // Keep the last authorized detail visible while the list poll reports errors.
      }
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [selectedTicket?.id]);

  const groupedTickets = useMemo(() => tickets.filter((ticket) => handoffBucket(ticket.handoff_state) === activeBucket), [tickets, activeBucket]);
  const counts = useMemo(() => Object.fromEntries(BUCKETS.map((bucket) => [
    bucket.id,
    tickets.filter((ticket) => handoffBucket(ticket.handoff_state) === bucket.id).length,
  ])), [tickets]);

  async function acknowledgeTicket() {
    if (!selectedTicket?.id || acknowledging || !selectedTicket.can_acknowledge) return;
    setAcknowledging(true);
    setError(null);
    try {
      await api.post(`/api/technician/inbox/${selectedTicket.id}/acknowledge/`, {});
      await Promise.all([loadInbox(), loadTicketDetail(selectedTicket.id)]);
    } catch (ackError) {
      setError(ackError.detail || 'Ticket acknowledgement did not complete. Retry to check its status.');
    } finally {
      setAcknowledging(false);
    }
  }

  return (
    <div className="content-body technician-inbox-page">
      <section className="technician-inbox-heading">
        <div>
          <div className="technician-eyebrow"><ShieldCheck size={14} /> AUTHORISED SERVICE WORKSPACE</div>
          <h2>Technician Inbox</h2>
          <p>Review incoming customer reports and acknowledge requests you can take.</p>
        </div>
        <div className="technician-sync-state" aria-live="polite">
          <span className="technician-live-dot" /> Live updates every 10 seconds
          <button className="btn btn-sm btn-outline-secondary" type="button" onClick={loadInbox} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'technician-spin' : ''} /> Refresh
          </button>
        </div>
      </section>

      {error && <div className="alert alert-warning" role="alert"><AlertTriangle size={16} /> {error}</div>}

      <div className="technician-inbox-layout">
        <section className="technician-queue-panel" aria-label="Ticket queues">
          <div className="technician-queue-title">
            <div><h3>Service requests</h3><span>{tickets.length} visible in your authorised scope</span></div>
            <Inbox size={19} />
          </div>
          <div className="technician-queue-tabs" role="tablist" aria-label="Ticket status">
            {BUCKETS.map((bucket) => {
              const Icon = bucket.icon;
              return (
                <button
                  key={bucket.id}
                  role="tab"
                  aria-selected={activeBucket === bucket.id}
                  className={activeBucket === bucket.id ? 'active' : ''}
                  type="button"
                  onClick={() => setActiveBucket(bucket.id)}
                >
                  <Icon size={15} /> <span>{bucket.label}</span><b>{counts[bucket.id]}</b>
                </button>
              );
            })}
          </div>

          <div className="technician-ticket-list" aria-live="polite">
            {loading && tickets.length === 0 && <div className="technician-empty"><RefreshCw className="technician-spin" size={20} /> Loading authorized tickets…</div>}
            {!loading && groupedTickets.length === 0 && (
              <div className="technician-empty">
                <Inbox size={25} />
                <strong>No {BUCKETS.find((bucket) => bucket.id === activeBucket)?.label.toLowerCase()} tickets</strong>
                <span>New requests in your assignment scope will appear here automatically.</span>
              </div>
            )}
            {groupedTickets.map((ticket) => (
              <button
                className={`technician-ticket-card ${ticket.is_new ? 'is-new' : ''} ${selectedTicket?.id === ticket.id ? 'selected' : ''}`}
                type="button"
                key={ticket.id}
                onClick={() => loadTicketDetail(ticket.id)}
              >
                <div className="technician-ticket-card-top">
                <strong>{ticket.ticket_ref}</strong>
                  <span className={`technician-state-chip state-${ticket.handoff_state}`}>{handoffLabel(ticket.handoff_state)}</span>
                </div>
                {ticket.is_new && <span className="technician-new-ticket"><i /> NEW TICKET</span>}
                <div className="technician-ticket-issue">{ticket.complaint_text || ticket.complaint_type || 'Service request'}</div>
                <div className="technician-ticket-meta">
                  <span><Wrench size={13} /> {ticket.asset?.name || 'Equipment not recorded'}</span>
                  <span><UserRound size={13} /> {ticket.customer?.name || 'Customer'}</span>
                </div>
                <div className="technician-ticket-card-bottom">
                  <span><Clock3 size={13} /> {ticket.created_at ? new Date(ticket.created_at).toLocaleString() : 'Time not recorded'}</span>
                  {ticket.priority && <span className="technician-priority">{ticket.priority}</span>}
                </div>
                <span className="technician-open-ticket">Open ticket <ChevronRight size={14} /></span>
              </button>
            ))}
          </div>
          <div className="technician-last-updated">{lastUpdated ? `Updated ${new Date(lastUpdated).toLocaleTimeString()}` : 'Waiting for first update'}</div>
        </section>

        <section className="technician-detail-panel" aria-label="Ticket details">
          {!selectedTicket && !detailLoading && (
            <div className="technician-detail-empty">
              <div className="technician-detail-empty-icon"><Ticket size={24} /></div>
              <h3>Ticket details</h3>
              <p>Open a request to review the customer report, equipment, conversation and confirmed work.</p>
            </div>
          )}
          {detailLoading && <div className="technician-detail-loading"><RefreshCw className="technician-spin" size={20} /> Loading ticket…</div>}
          {selectedTicket && !detailLoading && (
            <article className="technician-detail-content">
              <header className="technician-detail-header">
                <div>
                  <span className="technician-eyebrow">SERVICE REQUEST</span>
                  <h3>{selectedTicket.ticket_ref}</h3>
                  <span className={`technician-state-chip state-${selectedTicket.handoff_state}`}>{handoffLabel(selectedTicket.handoff_state)}</span>
                </div>
                {selectedTicket.can_acknowledge && (
                  <button type="button" className="btn btn-primary" onClick={acknowledgeTicket} disabled={acknowledging}>
                    <Check size={16} /> {acknowledging ? 'Acknowledging…' : 'Acknowledge ticket'}
                  </button>
                )}
              </header>

              <div className="technician-overview-grid">
                <div><small>Customer</small><strong>{selectedTicket.customer?.name || 'Not recorded'}</strong></div>
                <div><small>Organisation</small><strong>{selectedTicket.tenant?.name || 'Current tenant'}</strong></div>
                <div><small>Asset</small><strong>{selectedTicket.asset?.name || 'Not recorded'}{selectedTicket.asset?.asset_code ? ` · ${selectedTicket.asset.asset_code}` : ''}</strong></div>
                <div><small>Product / model</small><strong>{selectedTicket.asset?.product_name || selectedTicket.asset?.model_number || 'Not recorded'}</strong></div>
                <div><small>Facility</small><strong>{selectedTicket.site?.name || 'Not recorded'}</strong></div>
                <div><small>Priority</small><strong>{selectedTicket.priority || 'Not recorded'}</strong></div>
                <div><small>Received</small><strong>{selectedTicket.created_at ? new Date(selectedTicket.created_at).toLocaleString() : 'Not recorded'}</strong></div>
                <div><small>Assignment</small><strong>{selectedTicket.technician_name || 'Unassigned'}</strong></div>
              </div>

              <section className="technician-detail-section">
                <h4><Activity size={16} /> Reported problem</h4>
                <p className="technician-complaint">{selectedTicket.complaint_text || selectedTicket.complaint_type || 'No problem description was recorded.'}</p>
              </section>

              <section className="technician-detail-section">
                <h4><FileText size={16} /> Customer conversation</h4>
                {selectedTicket.conversation?.length ? (
                  <div className="technician-transcript">
                    {selectedTicket.conversation.map((message) => (
                      <div className={`technician-transcript-message ${message.sender_role === 'customer' ? 'customer' : 'assistant'}`} key={message.id}>
                        <div className="technician-transcript-label">
                          <strong>{message.sender_role === 'customer' ? 'Customer reported' : 'Servy guidance'}</strong>
                          <span>{message.confirmation}</span>
                        </div>
                        <p>{message.message_text}</p>
                        {message.created_at && <time>{new Date(message.created_at).toLocaleString()}</time>}
                      </div>
                    ))}
                  </div>
                ) : <p className="technician-muted">No saved conversation is linked to this service request.</p>}
              </section>

              <section className="technician-detail-section">
                <h4><Check size={16} /> Confirmed troubleshooting</h4>
                {selectedTicket.confirmed_troubleshooting?.length ? (
                  <ul>{selectedTicket.confirmed_troubleshooting.map((step, index) => <li key={index}>{formatConfirmedStep(step) || 'Confirmed step recorded'}</li>)}</ul>
                ) : <p className="technician-muted">No troubleshooting step is recorded as completed. Servy guidance above is not treated as customer-confirmed work.</p>}
              </section>

              <section className="technician-detail-section">
                <h4><FileText size={16} /> Verified supporting references</h4>
                {selectedTicket.supporting_documents?.length ? (
                  <ul className="technician-reference-list">
                    {selectedTicket.supporting_documents.map((reference) => (
                      <li key={`${reference.document_id}-${reference.chunk_id}`}>
                        <strong>{reference.title}</strong>{reference.heading ? <span> · {reference.heading}</span> : null}
                        <small className="technician-reference-id">Doc {reference.document_id} · Chunk {reference.chunk_id}</small>
                      </li>
                    ))}
                  </ul>
                ) : <p className="technician-muted">No current, authorised document reference was attached to this conversation.</p>}
              </section>

              <section className="technician-detail-section technician-outstanding-section">
                <h4><AlertTriangle size={16} /> Outstanding information</h4>
                {selectedTicket.outstanding_information?.length ? (
                  <ul>{selectedTicket.outstanding_information.map((item, index) => <li key={index}>{item}</li>)}</ul>
                ) : <p className="technician-muted">No unanswered clarification is recorded. Investigate any remaining details during service.</p>}
              </section>
            </article>
          )}
        </section>
      </div>
    </div>
  );
}
