import React, { useState } from 'react';
import { Bot, BookOpen, Building2, FileText, LockKeyhole, LogOut, Package, ShieldCheck, Ticket } from 'lucide-react';
import { CustomerAnswer } from '../components/CustomerAnswer';
import './RealDemoPage.css';

async function call(path, token, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { 'Authorization': `Demo ${token}`, 'Content-Type': 'application/json' },
    cache: 'no-store',
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Demo request failed');
  return data;
}

export function RealDemoPage() {
  const [token, setToken] = useState('');
  const [entry, setEntry] = useState('');
  const [assets, setAssets] = useState(null);
  const [assetId, setAssetId] = useState('');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [answeredQuestion, setAnsweredQuestion] = useState('');
  const [ticketDraft, setTicketDraft] = useState(null);
  const [ticketPreview, setTicketPreview] = useState(null);
  const [ticketConfirmation, setTicketConfirmation] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [view, setView] = useState('chat');

  async function enter(event) {
    event.preventDefault();
    setError('');
    try {
      const value = entry.trim();
      const data = await call('/demo-api/inventory/', value);
      setToken(value);
      setEntry('');
      setAssets(data.assets);
      setAssetId(String(data.assets.find(item => item.id === 60)?.id || data.assets[0]?.id || ''));
    } catch (err) {
      setError(err.message === 'Demo identity required'
        ? 'Use the separate code in real_demo_access.txt. The servy_ui_tester password is only for the synthetic test site.'
        : err.message);
    }
  }

  async function ask(event) {
    event.preventDefault();
    setError('');
    setResult(null);
    setTicketDraft(null);
    setTicketPreview(null);
    setTicketConfirmation(null);
    setBusy(true);
    try {
      const submittedQuestion = question.trim();
      const data = await call('/demo-api/query/', token, {
        method: 'POST', body: JSON.stringify({ asset_id: Number(assetId), question: submittedQuestion }),
      });
      setResult(data);
      setAnsweredQuestion(submittedQuestion);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  function startTicket() {
    setTicketDraft({ complaint_type: answeredQuestion.slice(0, 180), complaint_text: answeredQuestion });
    setTicketPreview(null);
    setTicketConfirmation(null);
    setError('');
  }

  async function reviewTicket(event) {
    event.preventDefault();
    setError('');
    setBusy(true);
    try {
      const data = await call('/demo-api/tickets/preview/', token, {
        method: 'POST', body: JSON.stringify({ ...ticketDraft, question: answeredQuestion,
          ticket_context: result.ticket_context }),
      });
      setTicketPreview(data.ticket);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  function confirmTicketDetails() {
    if (!ticketPreview) return;
    setTicketConfirmation({ ticket: ticketPreview,
      detail: 'Details reviewed locally. No Servy ticket or database record was created.' });
    setTicketPreview(null);
    setTicketDraft(null);
  }

  const asset = assets?.find(item => String(item.id) === assetId);
  return <div className="app-container real-demo-shell">
    <aside className="sidebar real-demo-sidebar">
      <div className="sidebar-header"><div className="brand-icon">S</div>
        <div className="brand-text"><h1>Servy RAG</h1><span>Field Service AI</span></div></div>
      <nav className="sidebar-nav" aria-label="Demo navigation"><div className="nav-section-title">Company Showcase</div>
        <button type="button" className={`nav-item ${view === 'chat' ? 'active' : ''}`} onClick={() => setView('chat')}>
          <Bot className="icon" /><span>AI Self-Service</span></button>
        <button type="button" className={`nav-item ${view === 'assets' ? 'active' : ''}`} onClick={() => setView('assets')} disabled={!assets}>
          <Package className="icon" /><span>Equipment</span></button>
      </nav>
      <div className="sidebar-footer"><div className="user-profile-badge"><div className="avatar">RD</div>
        <div className="user-meta"><div className="user-name">Real-data demo</div><div className="user-role">Local · read-only</div></div></div>
        {assets && <button type="button" className="btn btn-secondary btn-sm real-demo-lock" onClick={() => {
          setToken(''); setAssets(null); setResult(null); setQuestion(''); setTicketDraft(null);
          setTicketPreview(null); setTicketConfirmation(null); setView('chat'); }}><LogOut size={14} /> Lock Demo</button>}</div>
    </aside>
    <main className="main-area">
      <header className="top-bar"><div className="top-bar-title"><h2>{view === 'assets' && assets ? 'Company Equipment' : 'Customer AI Self-Service'}</h2>
        <p>Tenant: <strong>Skydream</strong> · Role: <strong>Authorized local tester</strong></p></div>
        </header>
      <div className="content-body real-demo-content">
        <div className="real-demo-intro"><span className="real-demo-kicker">CUSTOMER AI SUPPORT PLATFORM</span>
          <h1>{view === 'assets' && assets ? 'Equipment & Original Documentation' : 'Equipment Troubleshooting & Service Portal'}</h1>
          <p>Ask about real company equipment. Servy checks approved original manuals and shows the source behind each cited claim.</p></div>
        {!assets ? <div className="real-demo-login-layout">
          <section className="card real-demo-login-card"><span className="real-demo-icon"><LockKeyhole size={23} /></span>
            <span className="real-demo-kicker">AUTHORIZED ACCESS</span><h2>Unlock your approved demo scope</h2>
            <p>The synthetic tester password does not unlock company data. Enter the separate code from <strong>real_demo_access.txt</strong>.</p>
            <form onSubmit={enter}><div className="form-group"><label className="form-label" htmlFor="demo-identity">Real-data demo access code</label>
              <input className="form-control" id="demo-identity" type="password" autoComplete="off" value={entry}
                onChange={event => setEntry(event.target.value)} required /></div>
              <button className="btn btn-primary real-demo-full" type="submit">Unlock approved scope</button></form>
            <small><ShieldCheck size={15} /> Customer documents stay restricted by equipment and customer.</small></section>
          <aside className="real-demo-login-aside"><BookOpen size={29} />
            <h2>Original manuals.<br />Real equipment.<br />Clear limits.</h2>
            <p>SELECT-only database access, a private company index copy, and local BGE-small + V5 retrieval.</p>
            <span>✓ No company tickets or dispatches created</span><span>✓ Source access checked when opened</span></aside>
        </div> : view === 'assets' ? <>
          <div className="real-demo-stats"><div className="kpi-card"><div className="kpi-icon purple"><Package size={22} /></div>
            <div><div className="kpi-value">{assets.length}</div><div className="kpi-label">Verified assets</div></div></div>
            <div className="kpi-card"><div className="kpi-icon green"><BookOpen size={22} /></div>
              <div><div className="kpi-value">{assets.filter(item => item.documents.length).length}</div><div className="kpi-label">Within document scope</div></div></div></div>
          <section className="card"><div className="card-header"><h2 className="card-title"><Package size={20} /> Company Equipment</h2>
            <span className="badge badge-indexed">Read-only inventory</span></div>
            <div className="real-demo-inventory">{assets.map(item => <button className="real-demo-inventory-item" type="button" key={item.id}
              onClick={() => { setAssetId(String(item.id)); setResult(null); setTicketDraft(null);
                setTicketPreview(null); setTicketConfirmation(null); setView('chat'); }}>
              <span className="real-demo-icon"><Package size={18} /></span><span><strong>{item.name}</strong><small>{item.code} · {item.model || 'Model not listed'}</small></span>
              <em>{item.documents.length ? `${item.documents.length} permitted manuals` : 'No demo document access'}</em></button>)}</div></section>
        </> : <>
          <div className="real-demo-columns"><section className="card real-demo-equipment">
            <h2 className="card-title"><Building2 size={20} /> Equipment Selection</h2>
            <div className="form-group"><label className="form-label" htmlFor="demo-asset">Equipment / Asset</label>
              <select className="form-control" id="demo-asset" value={assetId}
                onChange={event => { setAssetId(event.target.value); setResult(null); setTicketDraft(null);
                  setTicketPreview(null); setTicketConfirmation(null); }}>
                {assets.map(item => <option key={item.id} value={item.id}>{item.name} ({item.code})</option>)}</select></div>
            {asset && <div className="real-demo-asset-detail"><h3>{asset.name}</h3>
              <dl><div><dt>Asset code</dt><dd>{asset.code}</dd></div>
                <div><dt>Product</dt><dd>{asset.product || 'Not listed'}</dd></div>
                {asset.site && <div><dt>Site</dt><dd>{asset.site}</dd></div>}</dl>
              {!asset.documents.length && <p className="real-demo-doc-status"><FileText size={17} />
                Technical-document access unavailable in this demo</p>}</div>}</section>
            <section className="card real-demo-assistant"><div className="real-demo-assistant-header"><Bot size={25} />
              <div><h2>AI Troubleshooting Assistant</h2><p>Original company documents · Read-only Q&A</p></div></div>
              <form onSubmit={ask}><label className="form-label" htmlFor="demo-question">Describe the problem or symptom you are observing</label>
                <textarea className="form-control" id="demo-question" maxLength={4000} value={question}
                  onChange={event => setQuestion(event.target.value)} rows={5} required
                  placeholder="e.g. What does the manual say about a low-pressure fault?" />
                <div className="real-demo-char-count"><span>Unsupported guidance will be refused.</span><span>{question.length}/4000</span></div>
                <button className="btn btn-primary real-demo-full" type="submit" disabled={busy}>
                  {busy ? 'Checking original manuals…' : 'Ask Servy'}</button></form></section></div>
          {result && <section className="card real-demo-result"><div className="real-demo-result-header">
            <h2><span className="real-demo-result-icon"><Bot size={20} /></span> Servy answer</h2>
            <div className={`real-demo-result-status ${result.response_type === 'DIAGNOSTIC_GUIDANCE' ? 'is-guidance' : 'is-limited'}`}>
              {result.response_type.replaceAll('_', ' ')}</div>
          </div>
            {result.presentation ? <CustomerAnswer assetId={Number(assetId)} presentation={result.presentation}
              loadSource={url => call(url, token)} /> : <p>{result.message}</p>}
            {result.ticket_context && <div className="real-demo-ticket-action">
              <p>If this answer did not resolve the issue, prepare a service ticket for review.</p>
              <button className="btn btn-secondary" type="button" onClick={startTicket}>
                <Ticket size={17} /> Create Ticket</button>
            </div>}
          </section>}
          {ticketDraft && <section className="card real-demo-ticket" aria-label="Mock ticket form">
            <div className="real-demo-ticket-heading"><div><span className="real-demo-kicker">REVIEW ONLY</span>
              <h2>Review service ticket details</h2></div>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => {
                setTicketDraft(null); setTicketPreview(null); }}>Cancel</button></div>
            <p className="real-demo-ticket-note">Ticket creation is not authorized for this demo. You can review the details, but nothing will be submitted or saved.</p>
            <dl className="real-demo-ticket-context"><div><dt>Equipment</dt><dd>{asset?.name} ({asset?.code})</dd></div>
              {asset?.site && <div><dt>Site</dt><dd>{asset.site}</dd></div>}</dl>
            <form onSubmit={reviewTicket}>
              <label className="form-label" htmlFor="ticket-complaint-type">Issue summary</label>
              <input className="form-control" id="ticket-complaint-type" maxLength={180} required
                value={ticketDraft.complaint_type} onChange={event => { setTicketDraft({ ...ticketDraft,
                  complaint_type: event.target.value }); setTicketPreview(null); }} />
              <label className="form-label real-demo-ticket-label" htmlFor="ticket-complaint-text">Issue description</label>
              <textarea className="form-control" id="ticket-complaint-text" maxLength={4000} rows={5} required
                value={ticketDraft.complaint_text} onChange={event => { setTicketDraft({ ...ticketDraft,
                  complaint_text: event.target.value }); setTicketPreview(null); }} />
              <button className="btn btn-primary" type="submit" disabled={busy}>Review ticket details</button>
            </form>
            {ticketPreview && <div className="real-demo-ticket-preview" aria-live="polite">
              <h3>Confirm details for review</h3>
              <p><strong>Type:</strong> {ticketPreview.call_type} · <strong>Priority:</strong> {ticketPreview.priority}
                {' · '}<strong>Status:</strong> {ticketPreview.status}</p>
              <p><strong>Issue summary:</strong> {ticketPreview.complaint_type}</p>
              <p><strong>Issue description:</strong> {ticketPreview.complaint_text}</p>
              <button type="button" className="btn btn-primary" disabled={busy} onClick={confirmTicketDetails}>
                Confirm reviewed details</button>
            </div>}
          </section>}
          {ticketConfirmation && <div className="real-demo-ticket-confirmation" role="status">
            <strong>Ticket details confirmed for review</strong><span>{ticketConfirmation.detail}</span>
            <dl><div><dt>Equipment</dt><dd>{asset?.name} ({asset?.code})</dd></div>
              <div><dt>Issue summary</dt><dd>{ticketConfirmation.ticket.complaint_type}</dd></div>
              <div><dt>Issue description</dt><dd>{ticketConfirmation.ticket.complaint_text}</dd></div></dl>
          </div>}
        </>}
        {error && <p className="real-demo-error" role="alert">{error}</p>}
      </div>
    </main>
  </div>;
}
