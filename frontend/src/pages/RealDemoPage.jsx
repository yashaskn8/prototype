import React, { useEffect, useRef, useState } from 'react';
import { Bot, BookOpen, Building2, FileText, LockKeyhole, LogOut, Package, Plus, ShieldCheck, Send, Ticket } from 'lucide-react';
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
  const [messages, setMessages] = useState([]);
  const [continuation, setContinuation] = useState(null);
  const chatEndRef = useRef(null);
  const requestRevision = useRef(0);
  const [result, setResult] = useState(null);
  const [answeredQuestion, setAnsweredQuestion] = useState('');
  const [ticketDraft, setTicketDraft] = useState(null);
  const [ticketPreview, setTicketPreview] = useState(null);
  const [ticketConfirmation, setTicketConfirmation] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [view, setView] = useState('chat');


  // Chat and its signed retrieval context are deliberately browser-memory only.
  // No customer conversation is saved to the company's operational database.
  function resetConversation() {
    requestRevision.current += 1;
    setMessages([]);
    setContinuation(null);
    setQuestion('');
    setResult(null);
    setAnsweredQuestion('');
    setTicketDraft(null);
    setTicketPreview(null);
    setTicketConfirmation(null);
    setBusy(false);
    setError('');
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages]);

  async function enter(event) {
    event.preventDefault();
    setError('');
    try {
      const value = entry.trim();
      const data = await call('/demo-api/inventory/', value);
      resetConversation();
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


  async function ask(event, suggestedQuestion = null) {
    event?.preventDefault?.();
    const submittedQuestion = (suggestedQuestion ?? question).trim();
    if (!submittedQuestion || busy || !token || !assetId) return;
    const activeRevision = requestRevision.current;
    const activeAsset = assetId;
    const messageId = `user-${Date.now()}-${Math.random()}`;
    const responseId = `assistant-${Date.now()}-${Math.random()}`;
    setError('');
    setBusy(true);
    setTicketDraft(null);
    setTicketPreview(null);
    setTicketConfirmation(null);
    setMessages(previous => [...previous,
      { id: messageId, role: 'user', text: submittedQuestion },
      { id: responseId, role: 'assistant', pending: true },
    ]);
    setQuestion('');
    try {
      const data = await call('/demo-api/query/', token, {
        method: 'POST',
        body: JSON.stringify({
          asset_id: Number(activeAsset),
          question: submittedQuestion,
          continuation,
        }),
      });
      if (requestRevision.current !== activeRevision || activeAsset !== assetId) return;
      setMessages(previous => previous.map(message => message.id === responseId
        ? { id: responseId, role: 'assistant', data }
        : message));
      setContinuation(data.continuation || null);
      setResult(data);
      setAnsweredQuestion(submittedQuestion);
    } catch (err) {
      if (requestRevision.current !== activeRevision || activeAsset !== assetId) return;
      setMessages(previous => previous.filter(message =>
        message.id !== responseId && message.id !== messageId));
      setQuestion(submittedQuestion);
      setError(err.message);
    } finally {
      if (requestRevision.current === activeRevision && activeAsset === assetId) {
        setBusy(false);
      }
    }
  }

  function startTicket() {
    const customerTurns = messages.filter(message => message.role === 'user');
    const originalComplaint = customerTurns[0]?.text || answeredQuestion;
    const customerHistory = customerTurns.map((message, index) =>
      `Turn ${index + 1}: ${message.text}`).join('\n');
    setTicketDraft({
      complaint_type: originalComplaint.slice(0, 180),
      complaint_text: customerHistory.slice(0, 4000) || originalComplaint,
    });
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
          resetConversation(); setToken(''); setAssets(null); setAssetId(''); setView('chat'); }}><LogOut size={14} /> Lock Demo</button>}</div>
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
              onClick={() => { resetConversation(); setAssetId(String(item.id)); setView('chat'); }}>
              <span className="real-demo-icon"><Package size={18} /></span><span><strong>{item.name}</strong><small>{item.code} · {item.model || 'Model not listed'}</small></span>
              <em>{item.documents.length ? `${item.documents.length} permitted manuals` : 'No demo document access'}</em></button>)}</div></section>
        </> : <>
          <div className="real-demo-columns real-demo-conversation-layout"><section className="card real-demo-equipment">
            <h2 className="card-title"><Building2 size={20} /> Equipment Selection</h2>
            <div className="form-group"><label className="form-label" htmlFor="demo-asset">Equipment / Asset</label>
              <select className="form-control" id="demo-asset" value={assetId}
                onChange={event => { resetConversation(); setAssetId(event.target.value); }}>
                {assets.map(item => <option key={item.id} value={item.id}>{item.name} ({item.code})</option>)}</select></div>
            {asset && <div className="real-demo-asset-detail"><h3>{asset.name}</h3>
              <dl><div><dt>Asset code</dt><dd>{asset.code}</dd></div>
                <div><dt>Product</dt><dd>{asset.product || 'Not listed'}</dd></div>
                {asset.site && <div><dt>Site</dt><dd>{asset.site}</dd></div>}</dl>
              {!asset.documents.length && <p className="real-demo-doc-status"><FileText size={17} />
                Technical-document access unavailable in this demo</p>}</div>}</section>

            <section className="card real-demo-assistant real-demo-chat-panel" aria-label="Ask Servy conversation">
              <div className="real-demo-assistant-header">
                <Bot size={25} />
                <div className="real-demo-chat-heading">
                  <h2>Ask Servy</h2>
                  <p>Continuous conversation · Approved original manuals · Cited answers</p>
                </div>
                <button type="button" className="real-demo-new-chat" onClick={resetConversation}>
                  <Plus size={16} /> New chat
                </button>
              </div>
              <div className="real-demo-chat-thread" role="log" aria-label="Conversation messages" aria-live="polite">
                {messages.length === 0 && <div className="real-demo-chat-empty">
                  <Bot size={28} />
                  <h3>Ask about this equipment</h3>
                  <p>Describe a symptom or fault code. Continue asking follow-up questions;
                    Servy will use the earlier customer reports to retrieve relevant approved documentation.</p>
                  <small>This is a read-only local demo. Conversations exist only in this browser tab and are not saved as company records.</small>
                </div>}
                {messages.map((message, index) => (
                  <div key={message.id} className={`real-demo-chat-message ${message.role === 'user' ? 'is-user' : 'is-assistant'}`}>
                    <div className="real-demo-chat-avatar">{message.role === 'user' ? 'You' : <Bot size={19} />}</div>
                    <div className="real-demo-chat-bubble">
                      {message.role === 'user' ? <div className="real-demo-chat-user-text">{message.text}</div>
                        : message.pending ? <p role="status">Checking the approved manuals and earlier messages…</p>
                          : <>
                            <div className="real-demo-chat-answer-heading">
                              <strong>Servy</strong>
                              <span className={message.data?.response_type === 'DIAGNOSTIC_GUIDANCE'
                                ? 'real-demo-chat-guidance' : 'real-demo-chat-limited'}>
                                {({
                                  DIAGNOSTIC_GUIDANCE: 'Cited answer',
                                  CLARIFICATION: 'Quick question',
                                  INSUFFICIENT_EVIDENCE: message.data?.reason_code === 'OUT_OF_SCOPE_QUERY'
                                    ? 'Outside manual scope' : 'More evidence needed',
                                  ESCALATION_REQUIRED: 'Safety notice',
                                  POLICY_VIOLATION: 'Not permitted',
                                })[message.data?.response_type] || 'Response'}
                              </span>
                            </div>
                            {message.data?.presentation ? <CustomerAnswer
                              assetId={Number(assetId)}
                              presentation={message.data.presentation}
                              loadSource={url => call(url, token)}
                            /> : <p>{message.data?.message || 'No response is available.'}</p>}
                            {message.data?.response_type === 'DIAGNOSTIC_GUIDANCE' && (
                              <details className="real-demo-dialogue-meta">
                                <summary>How this answer was prepared</summary>
                                <p>The technical statements are drawn from approved company sources.
                                  {message.data.dialogue_planner === 'local-ollama-structured'
                                    ? ' A local language model helped organize the conversation.'
                                    : ' The response was organized without a generative language model.'}
                                </p>
                              </details>
                            )}
                            {Array.isArray(message.data?.clarification_options)
                              && message.data.clarification_options.length > 0
                              && index === messages.length - 1 && (
                                <div className="real-demo-chat-options" aria-label="Clarification suggestions">
                                  {message.data.clarification_options.map(option => (
                                    <button key={option} type="button" disabled={busy}
                                      onClick={() => ask(null, option)}>{option}</button>
                                  ))}
                                </div>
                              )}
                            {index === messages.length - 1
                              && Array.isArray(message.data?.suggested_follow_ups)
                              && message.data.suggested_follow_ups.length > 0 && (
                                <div className="real-demo-chat-options" aria-label="Suggested follow-up questions">
                                  {message.data.suggested_follow_ups.map(option => (
                                    <button key={option} type="button" disabled={busy}
                                      onClick={() => ask(null, option)}>{option}</button>
                                  ))}
                                </div>
                              )}
                            {index === messages.length - 1 && message.data?.ticket_context
                              && (message.data.response_type !== 'DIAGNOSTIC_GUIDANCE'
                                || messages.filter(item => item.role === 'user').length >= 3) && (
                              <div className="real-demo-ticket-action">
                                <p>Still need assistance? Prepare a service ticket draft for review only.</p>
                                <button className="btn btn-secondary btn-sm" type="button" onClick={startTicket}>
                                  <Ticket size={17} /> Review ticket draft
                                </button>
                              </div>
                            )}
                          </>}
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
              <form className="real-demo-chat-composer" onSubmit={ask}>
                <label className="form-label" htmlFor="demo-question">
                  {messages.length ? 'Continue this conversation' : 'Describe the equipment issue'}
                </label>
                <textarea className="form-control" id="demo-question" maxLength={4000}
                  value={question} rows={3} required disabled={busy || !asset?.can_query}
                  onChange={event => setQuestion(event.target.value)}
                  placeholder={messages.length ? 'Ask a follow-up, share an observation, or report a code…'
                    : 'e.g. My VRF module is not cooling. There is an error on the display…'}
                  onKeyDown={event => {
                    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      if (question.trim()) ask(null);
                    }
                  }}
                />
                <div className="real-demo-chat-composer-footer">
                  <span>{question.length}/4000 · Answers must be supported by permitted documents</span>
                  <button className="btn btn-primary" type="submit"
                    disabled={busy || !question.trim() || !asset?.can_query}>
                    <Send size={16} /> {busy ? 'Searching…' : 'Send'}
                  </button>
                </div>
              </form>
            </section>
          </div>
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
