import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle, Check, ClipboardList, Clock3, Download, FileText,
  Inbox, KeyRound, LogOut, RefreshCw, ShieldCheck, Ticket, Wrench,
} from 'lucide-react';
import './TechnicianSecretPage.css';

async function technicianCall(path, token, options = {}) {
  const headers = {
    Authorization: `Technician ${token}`,
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
  };
  const response = await fetch(path, {
    ...options,
    headers,
    cache: 'no-store',
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || 'Technician request failed.');
  }
  return response;
}

const STATUS_OPTIONS = [
  ['open', 'Open'],
  ['assigned', 'Assigned'],
  ['in_progress', 'In progress'],
  ['marked_for_closure', 'Marked for closure'],
  ['closed', 'Closed'],
];

function formatDate(value) {
  if (!value) return 'Not recorded';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Not recorded' : date.toLocaleString();
}

export function TechnicianSecretPage() {
  const [entry, setEntry] = useState('');
  const [token, setToken] = useState('');
  const [tickets, setTickets] = useState([]);
  const [selected, setSelected] = useState(null);
  const [status, setStatus] = useState('open');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [filter, setFilter] = useState('all');

  // Keep an unlocked technician queue current when the customer creates a
  // ticket in the separate RAG tab. Reads are safe when the workspace is absent.
  useEffect(() => {
    if (!token) return undefined;
    let stopped = false;
    let inFlight = false;
    const refreshQueue = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const response = await technicianCall('/tech-api/tickets/', token);
        const data = await response.json();
        if (!stopped) setTickets(Array.isArray(data.tickets) ? data.tickets : []);
      } catch {
        // Keep the last successful queue visible; the manual refresh reports errors.
      } finally {
        inFlight = false;
      }
    };
    const timer = window.setInterval(refreshQueue, 8000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [token]);

  async function loadQueue(accessCode = token) {
    if (!accessCode) return;
    setLoading(true);
    setError('');
    try {
      const response = await technicianCall('/tech-api/tickets/', accessCode);
      const data = await response.json();
      setTickets(Array.isArray(data.tickets) ? data.tickets : []);
    } catch (requestError) {
      setError(requestError.message || 'Could not load the technician queue.');
      throw requestError;
    } finally {
      setLoading(false);
    }
  }

  async function unlock(event) {
    event.preventDefault();
    const code = entry.trim();
    if (!code || loading) return;
    setLoading(true);
    setError('');
    try {
      const response = await technicianCall('/tech-api/tickets/', code);
      const data = await response.json();
      setToken(code);
      setEntry('');
      setTickets(Array.isArray(data.tickets) ? data.tickets : []);
      setSelected(null);
      setNotice('Technician access verified.');
    } catch (requestError) {
      setError(requestError.message || 'The technician access code is invalid.');
    } finally {
      setLoading(false);
    }
  }

  async function openTicket(ticket) {
    setError('');
    setNotice('');
    try {
      const response = await technicianCall(
        `/tech-api/tickets/${encodeURIComponent(ticket.id)}/`, token,
      );
      const data = await response.json();
      setSelected(data.ticket || null);
      setStatus(data.ticket?.status || 'open');
      setNotes(data.ticket?.technicianNotes || '');
    } catch (requestError) {
      setError(requestError.message || 'Could not open this ticket.');
    }
  }

  async function saveUpdate(event) {
    event.preventDefault();
    if (!selected || saving) return;
    setSaving(true);
    setError('');
    setNotice('');
    try {
      const response = await technicianCall(
        `/tech-api/tickets/${encodeURIComponent(selected.id)}/`,
        token,
        {
          method: 'PATCH',
          body: JSON.stringify({ status, technicianNotes: notes }),
        },
      );
      const data = await response.json();
      setSelected(data.ticket || null);
      setNotice('Ticket update saved to the local technician workspace.');
      await loadQueue();
    } catch (requestError) {
      setError(requestError.message || 'Could not save the ticket update.');
    } finally {
      setSaving(false);
    }
  }

  async function downloadAttachment(attachment) {
    if (!selected) return;
    setError('');
    try {
      const response = await technicianCall(
        `/tech-api/tickets/${encodeURIComponent(selected.id)}/attachments/${encodeURIComponent(attachment.id)}/`,
        token,
      );
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = attachment.name || 'ticket-attachment';
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (requestError) {
      setError(requestError.message || 'The attachment could not be downloaded.');
    }
  }

  function lockWorkspace() {
    setToken('');
    setEntry('');
    setTickets([]);
    setSelected(null);
    setNotes('');
    setStatus('open');
    setError('');
    setNotice('');
  }

  const counts = useMemo(() => ({
    all: tickets.length,
    open: tickets.filter(item => item.status === 'open').length,
    active: tickets.filter(item => ['assigned', 'in_progress'].includes(item.status)).length,
    closed: tickets.filter(item => ['marked_for_closure', 'closed'].includes(item.status)).length,
  }), [tickets]);

  const visibleTickets = useMemo(() => tickets.filter((item) => {
    if (filter === 'all') return true;
    if (filter === 'open') return item.status === 'open';
    if (filter === 'active') return ['assigned', 'in_progress'].includes(item.status);
    return ['marked_for_closure', 'closed'].includes(item.status);
  }), [tickets, filter]);

  if (!token) {
    return (
      <div className="technician-secret-shell">
        <main className="technician-secret-login">
          <div className="technician-secret-mark"><Wrench size={25} /></div>
          <span className="technician-secret-eyebrow">PRIVATE TECHNICIAN INSTANCE</span>
          <h1>Authorized Technician Workspace</h1>
          <p>Enter the separate technician access code to view service requests raised through the authorized company RAG.</p>
          <form onSubmit={unlock}>
            <label htmlFor="technician-access-code">Technician access code</label>
            <div className="technician-secret-input-wrap">
              <KeyRound size={18} />
              <input
                id="technician-access-code"
                type="password"
                autoComplete="off"
                value={entry}
                onChange={event => setEntry(event.target.value)}
                placeholder="Enter technician code"
                required
              />
            </div>
            {error && <div className="technician-secret-error" role="alert"><AlertCircle size={16} />{error}</div>}
            <button className="technician-secret-primary" type="submit" disabled={loading || !entry.trim()}>
              {loading ? <RefreshCw size={17} className="technician-secret-spin" /> : <ShieldCheck size={17} />}
              {loading ? 'Verifying access…' : 'Unlock technician workspace'}
            </button>
          </form>
          <div className="technician-secret-footnote">
            <ShieldCheck size={15} />
            <span>Separate code · localhost only · scoped to authorized RAG tickets</span>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="technician-secret-shell">
      <header className="technician-secret-topbar">
        <div className="technician-secret-brand">
          <span className="technician-secret-brand-icon"><Wrench size={21} /></span>
          <span><strong>Servy Technician</strong><small>Authorized RAG workspace</small></span>
        </div>
        <div className="technician-secret-top-actions">
          <span className="technician-secret-protected"><ShieldCheck size={14} /> Private session</span>
          <button type="button" onClick={lockWorkspace}><LogOut size={15} /> Lock</button>
        </div>
      </header>

      <main className="technician-secret-main">
        <section className="technician-secret-heading">
          <div>
            <span className="technician-secret-eyebrow">FIELD SERVICE · CUSTOMER HANDOFF</span>
            <h1>Technician ticket queue</h1>
            <p>Review customer reports submitted through the separate authorized RAG instance.</p>
          </div>
          <button type="button" className="technician-secret-refresh" onClick={() => loadQueue()} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'technician-secret-spin' : ''} /> Refresh queue
          </button>
        </section>

        <section className="technician-secret-stats" aria-label="Ticket summary">
          <div><span>All requests</span><strong>{counts.all}</strong></div>
          <div><span>New / open</span><strong>{counts.open}</strong></div>
          <div><span>Active</span><strong>{counts.active}</strong></div>
          <div><span>Completed / closure</span><strong>{counts.closed}</strong></div>
        </section>

        {error && <div className="technician-secret-banner error" role="alert"><AlertCircle size={16} />{error}</div>}
        {notice && <div className="technician-secret-banner success" role="status"><Check size={16} />{notice}</div>}

        <div className="technician-secret-layout">
          <section className="technician-secret-queue">
            <div className="technician-secret-queue-heading">
              <div><ClipboardList size={19} /><strong>RAG service requests</strong></div>
              <span>{visibleTickets.length} shown</span>
            </div>
            <div className="technician-secret-filters" role="group" aria-label="Filter tickets">
              {[
                ['all', 'All'],
                ['open', 'Open'],
                ['active', 'Active'],
                ['closed', 'Completed'],
              ].map(([value, label]) => (
                <button key={value} type="button" className={filter === value ? 'active' : ''}
                  onClick={() => setFilter(value)}>{label}</button>
              ))}
            </div>
            <div className="technician-secret-ticket-list">
              {loading && tickets.length === 0
                ? <div className="technician-secret-empty"><RefreshCw size={20} className="technician-secret-spin" /> Loading requests…</div>
                : visibleTickets.length === 0
                  ? <div className="technician-secret-empty"><Inbox size={24} /><strong>No requests in this view</strong><span>Create and confirm a ticket in Customer RAG. This queue refreshes automatically.</span></div>
                  : visibleTickets.map(ticket => (
                    <button type="button" key={ticket.id}
                      className={`technician-secret-ticket ${selected?.id === ticket.id ? 'selected' : ''}`}
                      onClick={() => openTicket(ticket)}>
                      <span className="technician-secret-ticket-top">
                        <strong>{ticket.id}</strong>
                        <em className={`status-${ticket.status}`}>{ticket.status.replaceAll('_', ' ')}</em>
                      </span>
                      <b>{ticket.complaintType || ticket.callType}</b>
                      <span>{ticket.assetName} · {ticket.customerName}</span>
                      <small><Clock3 size={13} /> {formatDate(ticket.createdAt)}</small>
                    </button>
                  ))}
            </div>
            <div className="technician-secret-queue-foot">Updates are stored only in the local RAG ticket workspace.</div>
          </section>

          <section className="technician-secret-detail">
            {!selected ? (
              <div className="technician-secret-detail-empty">
                <span><Ticket size={23} /></span>
                <h2>Request details</h2>
                <p>Select a request to read the complete customer conversation, inspect attachments and record the service update.</p>
              </div>
            ) : (
              <article>
                <header className="technician-secret-detail-heading">
                  <div><span className="technician-secret-eyebrow">CUSTOMER HANDOFF</span><h2>{selected.id}</h2></div>
                  <em className={`status-${selected.status}`}>{selected.status.replaceAll('_', ' ')}</em>
                </header>
                <div className="technician-secret-metadata">
                  <div><span>Customer</span><strong>{selected.customerName || 'Not recorded'}</strong></div>
                  <div><span>Facility</span><strong>{selected.siteName || 'Not recorded'}</strong></div>
                  <div><span>Equipment</span><strong>{selected.assetName || 'Not recorded'}{selected.assetCode ? ` · ${selected.assetCode}` : ''}</strong></div>
                  <div><span>Priority</span><strong>{selected.priority || 'Normal'}</strong></div>
                  <div><span>Severity</span><strong>{selected.severity || 'Not recorded'}</strong></div>
                  <div><span>Machine status</span><strong>{selected.machineStatus || 'Not recorded'}</strong></div>
                  <div><span>Received</span><strong>{formatDate(selected.createdAt)}</strong></div>
                </div>
                <section className="technician-secret-report">
                  <h3><FileText size={17} /> Customer report and RAG conversation</h3>
                  <pre>{selected.complaintText || 'No description was provided.'}</pre>
                </section>
                {selected.attachments?.length > 0 && (
                  <section className="technician-secret-attachments">
                    <h3>Issue documentation</h3>
                    {selected.attachments.map(item => (
                      <button key={item.id} type="button" onClick={() => downloadAttachment(item)}>
                        <FileText size={15} /><span>{item.name}</span><Download size={15} />
                      </button>
                    ))}
                  </section>
                )}
                <form className="technician-secret-update" onSubmit={saveUpdate}>
                  <h3>Technician update</h3>
                  <label>Status
                    <select value={status} onChange={event => setStatus(event.target.value)}>
                      {STATUS_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                    </select>
                  </label>
                  <label>Internal technician notes
                    <textarea value={notes} onChange={event => setNotes(event.target.value)}
                      maxLength={8000} rows={4} placeholder="Record inspection findings or the next service action…" />
                  </label>
                  <button type="submit" className="technician-secret-primary" disabled={saving}>
                    {saving ? <RefreshCw size={16} className="technician-secret-spin" /> : <Check size={16} />}
                    {saving ? 'Saving update…' : 'Save technician update'}
                  </button>
                </form>
              </article>
            )}
          </section>
        </div>
        <footer className="technician-secret-footer">
          <ShieldCheck size={14} /> Local code-gated instance · separate from operational MySQL service calls
        </footer>
      </main>
    </div>
  );
}
