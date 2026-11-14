import React, { useEffect, useRef, useState } from 'react';
import { Bot, BookOpen, Building2, FilePlus2, FileText, LockKeyhole, LogOut, Package, Plus, ShieldCheck, Send, Ticket, Trash2, X } from 'lucide-react';
import { CustomerAnswer } from '../components/CustomerAnswer';
import { AssistantTypingIndicator } from '../components/AssistantTypingIndicator';
import { consumeRagEventStream } from '../components/ragExecutionStream';
import {
  EMPTY_ASSISTANT_RESPONSE,
  createAssistantTypingMessage,
  finishRequest,
  hasAssistantResponse,
  replaceMessageById,
  tryBeginRequest,
} from '../components/assistantRequestLifecycle';
import '../components/AssistantTypingIndicator.css';
import {
  appendDemoThread, canOpenDemoThread, chatTitle, isDemoAssetThreadBound,
} from './demoThreadState';
import './RealDemoPage.css';
import {
  CONVERSATIONS_PATH, conversationPath, gatewaySourceUrl, messagesFromServer,
} from './demoGatewayClient';

async function call(path, token, options = {}) {
  const headers = { 'Authorization': `Demo ${token}` };
  if (typeof FormData === 'undefined' || !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  const response = await fetch(path, {
    ...options,
    headers,
    cache: 'no-store',
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Demo request failed');
  return data;
}

async function callRagStream(path, token, payload, onLog, signal) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {
      Authorization: `Demo ${token}`,
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(payload),
    cache: 'no-store',
    signal,
  });
  return consumeRagEventStream(response, onLog);
}

const TICKET_TYPES = ['Breakdown Service', 'Installation / Demo', 'Service', 'Preventive Maintenance (PM)', 'AMC'];
const SEVERITIES = ['Low', 'Medium', 'High', 'Critical'];
const PRIORITIES = [
  { value: 'low', label: 'Low' }, { value: 'normal', label: 'Normal' },
  { value: 'high', label: 'High' }, { value: 'critical', label: 'Critical' },
];
const MACHINE_STATUSES = ['Running', 'Intermittent', 'Stopped', 'Unknown'];
const COMPLAINT_TYPES = [
  'Cooling / performance', 'Electrical', 'Noise / vibration', 'Leakage',
  'Installation / commissioning', 'Preventive maintenance', 'Other',
];
const MAX_TICKET_FILES = 10;
const MAX_TICKET_UPLOAD_BYTES = 25 * 1024 * 1024;


function assignmentNotice(data) {
  const value = data?.issue_assignment;
  return value?.required === true && typeof value.notice === 'string'
    ? value.notice.slice(0, 500) : '';
}

// Inventory status from the server's per-manual readiness. An asset is
// ready only when every required manual is; a count of database records is
// never shown as readiness.
function manualStatus(item) {
  const required = Array.isArray(item?.required_manuals) ? item.required_manuals : [];
  if (item?.manuals_ready === true) return `${required.length} required manuals ready`;
  if (item?.manuals_ready === false) {
    const incomplete = required.filter(doc => doc && doc.ready !== true)
      .map(doc => `${doc.doc_type || 'manual'} (${doc.id})`);
    return incomplete.length
      ? `Not ready: ${incomplete.join(', ')} unavailable or incomplete`
      : 'Not ready: required manuals could not be verified';
  }
  return item?.documents?.length ? `${item.documents.length} permitted manuals` : 'No demo document access';
}



export function RealDemoPage() {
  const [token, setToken] = useState('');
  const [entry, setEntry] = useState('');
  const [assets, setAssets] = useState(null);
  const [assetId, setAssetId] = useState('');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [continuation, setContinuation] = useState(null);
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState('');
  const threadCache = useRef(new Map());
  const threadSequence = useRef(0);
  const chatEndRef = useRef(null);
  const requestRevision = useRef(0);
  const requestInFlight = useRef(null);
  const requestStreamController = useRef(null);
  const [result, setResult] = useState(null);
  const [answeredQuestion, setAnsweredQuestion] = useState('');
  const [ticketDraft, setTicketDraft] = useState(null);
  const [ticketFiles, setTicketFiles] = useState([]);
  const [ticketConfirmation, setTicketConfirmation] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [view, setView] = useState('chat');

  useEffect(() => () => requestStreamController.current?.abort(), []);

  // Conversations are saved in the isolated local development database
  // (servy_dev) and reload after a browser refresh. Nothing is written to the
  // company's operational database.
  function resetConversation() {
    requestRevision.current += 1;
    requestStreamController.current?.abort();
    requestStreamController.current = null;
    requestInFlight.current = null;
    setMessages([]);
    setContinuation(null);
    setQuestion('');
    setResult(null);
    setAnsweredQuestion('');
    setTicketDraft(null);
    setTicketFiles([]);
    setTicketConfirmation(null);
    setBusy(false);
    setError('');
  }

  // Each thread has separate signed state. Browser memory only: no disk writes.
  function snapshotThread() {
    return { assetId, messages, continuation, question, result, answeredQuestion,
      ticketDraft, ticketFiles, ticketConfirmation };
  }

  function hydrateThread(snapshot) {
    requestRevision.current += 1;
    requestStreamController.current?.abort();
    requestStreamController.current = null;
    setMessages(snapshot?.messages || []);
    setContinuation(snapshot?.continuation || null);
    setQuestion(snapshot?.question || '');
    setResult(snapshot?.result || null);
    setAnsweredQuestion(snapshot?.answeredQuestion || '');
    setTicketDraft(snapshot?.ticketDraft || null);
    setTicketFiles(snapshot?.ticketFiles || []);
    setTicketConfirmation(snapshot?.ticketConfirmation || null);
    setError('');
  }

  function initializeThreads(nextAssetId, accessToken = token) {
    threadCache.current.clear();
    threadSequence.current = nextAssetId ? 1 : 0;
    const initialId = nextAssetId ? 'thread-' + crypto.randomUUID() : '';
    setActiveThreadId(initialId);
    setThreads(nextAssetId
      ? [{ id: initialId, assetId: String(nextAssetId), title: 'New chat' }]
      : []);
    resetConversation();
    if (nextAssetId && accessToken) loadSavedThreads(String(nextAssetId), accessToken);
  }

  // Previously saved conversations for this equipment (survive a refresh).
  async function loadSavedThreads(forAssetId, accessToken) {
    try {
      const data = await call(`${CONVERSATIONS_PATH}?asset_id=${encodeURIComponent(forAssetId)}&limit=50`, accessToken);
      const saved = (data.conversations || [])
        .filter(item => String(item.asset_id ?? item.asset?.id ?? forAssetId) === forAssetId)
        .map(item => ({
          id: 'conv-' + item.conversation_uuid, assetId: forAssetId,
          conversationUuid: item.conversation_uuid, title: item.title || 'Saved chat',
        }));
      setThreads(previous => [...previous.filter(item => !item.conversationUuid
        || !saved.some(entry => entry.conversationUuid === item.conversationUuid)), ...saved]);
    } catch {
      setError('Saved conversations could not be loaded. New questions still work.');
    }
  }

  function startNewChat() {
    if (busy || !assetId) return;
    if (activeThreadId) threadCache.current.set(activeThreadId, snapshotThread());
    const id = 'thread-' + crypto.randomUUID();
    const next = appendDemoThread(threads, {
      id, assetId: String(assetId), title: 'New chat',
    });
    for (const saved of threads) {
      if (!next.some(item => item.id === saved.id)) threadCache.current.delete(saved.id);
    }
    setThreads(next);
    setActiveThreadId(id);
    resetConversation();
  }

  async function openThread(id) {
    if (busy || id === activeThreadId) return;
    const thread = threads.find(item => item.id === id);
    if (!canOpenDemoThread(thread, assetId)) {
      setError('That conversation is not available for the selected equipment.');
      return;
    }
    let snapshot = threadCache.current.get(id);
    if (!snapshot && thread.conversationUuid) {
      try {
        const data = await call(conversationPath(thread.conversationUuid, 'messages/'), token);
        snapshot = { assetId: String(assetId), messages: messagesFromServer(data.messages),
          continuation: thread.conversationUuid };
      } catch {
        setError('That saved conversation could not be loaded.');
        return;
      }
    }
    if (!snapshot || String(snapshot.assetId) !== String(assetId)) {
      setError('That conversation was cleared or its equipment scope changed.');
      return;
    }
    if (activeThreadId) threadCache.current.set(activeThreadId, snapshotThread());
    setActiveThreadId(id);
    hydrateThread(snapshot);
  }

  function deleteThread(id) {
    if (busy) return;
    const remaining = threads.filter(item => item.id !== id);
    threadCache.current.delete(id);
    if (id !== activeThreadId) {
      setThreads(remaining);
      return;
    }
    if (!remaining.length) {
      initializeThreads(assetId);
      return;
    }
    const next = remaining.at(-1);
    setThreads(remaining);
    setActiveThreadId(next.id);
    hydrateThread(threadCache.current.get(next.id));
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages, activeThreadId]);

  async function enter(event) {
    event.preventDefault();
    setError('');
    try {
      const value = entry.trim();
      const data = await call('/demo-api/inventory/', value);
      // Never choose equipment silently. Every conversation must begin with
      // the customer explicitly selecting the asset they are asking about.
      const selected = '';
      initializeThreads(selected);
      setToken(value);
      setEntry('');
      setAssets(data.assets);
      setAssetId(selected);
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
    if (!isDemoAssetThreadBound(threads, activeThreadId, assetId)) {
      setError('Select the intended equipment and start a new chat before asking a question.');
      return;
    }
    const messageId = `user-${Date.now()}-${Math.random()}`;
    const responseId = `assistant-${Date.now()}-${Math.random()}`;
    if (!tryBeginRequest(requestInFlight, responseId)) return;
    const activeRevision = requestRevision.current;
    const activeAsset = assetId;
    requestStreamController.current?.abort();
    const streamController = new AbortController();
    requestStreamController.current = streamController;
    setError('');
    setBusy(true);
    setTicketDraft(null);
    setTicketFiles([]);
    setTicketConfirmation(null);
    setMessages(previous => [...previous,
      { id: messageId, role: 'user', text: submittedQuestion },
      createAssistantTypingMessage(responseId),
    ]);
    setThreads(previous => previous.map(item =>
      item.id === activeThreadId && item.title === 'New chat'
        ? { ...item, title: chatTitle(submittedQuestion) } : item
    ));
    setQuestion('');
    try {
      // The first question of a new chat creates the saved conversation; every
      // question then continues it through the official message stream.
      let conversationUuid = continuation;
      if (!conversationUuid) {
        const created = await call(CONVERSATIONS_PATH, token, {
          method: 'POST', body: JSON.stringify({ asset_id: Number(activeAsset) }),
        });
        conversationUuid = created.conversation_uuid;
        if (!conversationUuid) throw new Error('The conversation could not be started.');
        setContinuation(conversationUuid);
        setThreads(previous => previous.map(item => item.id === activeThreadId
          ? { ...item, conversationUuid } : item));
      }
      const turn = await callRagStream(
        conversationPath(conversationUuid, 'messages-stream/'),
        token,
        { message: submittedQuestion },
        undefined,
        streamController.signal,
      );
      const data = { ...(turn?.contract || {}), continuation: conversationUuid };
      if (requestRevision.current !== activeRevision || activeAsset !== assetId) return;
      const responseAssetId = data?.presentation?.equipment?.id ?? data?.asset?.id;
      if (responseAssetId != null && String(responseAssetId) !== String(activeAsset)) {
        throw new Error('Servy response equipment scope mismatch');
      }
      if (hasAssistantResponse(data)) {
        setMessages(previous => replaceMessageById(previous, responseId,
          { id: responseId, role: 'assistant', data }));
      } else {
        setMessages(previous => replaceMessageById(previous, responseId,
          { id: responseId, role: 'assistant',
            errorMessage: EMPTY_ASSISTANT_RESPONSE }));
        setQuestion(submittedQuestion);
      }
      setContinuation(data.continuation || null);
      setResult(hasAssistantResponse(data) ? data : null);
      if (hasAssistantResponse(data)) setAnsweredQuestion(submittedQuestion);
    } catch (err) {
      if (requestRevision.current !== activeRevision || activeAsset !== assetId) return;
      const errorMessage = err?.message === 'Servy response equipment scope mismatch'
        ? 'Servy returned information for different equipment. The response was blocked. Select the intended asset and start a new chat.'
        : 'Servy could not complete that response. Please try again.';
      setMessages(previous => replaceMessageById(previous, responseId,
        { id: responseId, role: 'assistant', errorMessage }));
      setQuestion(submittedQuestion);
      setError('');
    } finally {
      if (requestStreamController.current === streamController) {
        requestStreamController.current = null;
      }
      finishRequest(requestInFlight, responseId);
      if (requestRevision.current === activeRevision && activeAsset === assetId) {
        setBusy(false);
      }
    }
  }

  // The ticket description is built by the server from the complete saved
  // conversation; the customer reviews it here and confirms explicitly.
  async function startTicket() {
    const latest = messages.at(-1);
    if (latest?.role !== 'assistant' || !continuation || busy) return;
    setBusy(true);
    setError('');
    try {
      const data = await call(conversationPath(continuation, 'ticket/'), token);
      if (data.ticket) {
        setTicketConfirmation({ ticket: data.ticket, duplicate: true,
          detail: 'A service call already exists for this conversation.' });
        return;
      }
      setTicketDraft({
        call_type: TICKET_TYPES[0], complaint_type: '', severity: '', priority: 'normal', machine_status: '',
        preview: data.preview || null,
        previewDetail: data.preview_detail || (data.preview ? '' : 'The service call description could not be prepared.'),
      });
      setTicketFiles([]);
      setTicketConfirmation(null);
    } catch (err) {
      setError(err.message || 'The service call preview could not be loaded.');
    } finally {
      setBusy(false);
    }
  }

  function closeTicketPanel() {
    if (busy) return;
    setTicketDraft(null);
    setTicketFiles([]);
    setError('');
  }

  function updateTicketDraft(field, value) {
    setTicketDraft(current => current ? { ...current, [field]: value } : current);
    setError('');
  }

  function addTicketFiles(incoming) {
    const selected = [...ticketFiles, ...Array.from(incoming || [])]
      .filter((file, index, all) => all.findIndex(item =>
        `${item.name}:${item.size}:${item.lastModified}` === `${file.name}:${file.size}:${file.lastModified}`) === index);
    if (selected.length > MAX_TICKET_FILES) {
      setError(`Choose no more than ${MAX_TICKET_FILES} attachments.`);
      return;
    }
    if (selected.some(file => file.size > MAX_TICKET_UPLOAD_BYTES)
      || selected.reduce((total, file) => total + file.size, 0) > MAX_TICKET_UPLOAD_BYTES) {
      setError('Attachments must total 25 MB or less.');
      return;
    }
    setTicketFiles(selected);
    setError('');
  }

  async function createTicket(event) {
    event.preventDefault();
    setError('');
    setBusy(true);
    try {
      const payload = new FormData();
      payload.append('confirm_ticket_creation', 'true');
      ['call_type', 'complaint_type', 'severity', 'priority', 'machine_status'].forEach(field => {
        if (ticketDraft[field]) payload.append(field, ticketDraft[field]);
      });
      ticketFiles.forEach(file => payload.append('attachments', file, file.name));
      const data = await call(conversationPath(continuation, 'ticket/'), token, {
        method: 'POST', body: payload,
      });
      if (data.mode !== 'SERVY_DEV_MYSQL' || !data.ticket?.id) {
        throw new Error('The service call was not confirmed.');
      }
      setTicketConfirmation({ ticket: data.ticket, duplicate: data.duplicate === true,
        detail: data.duplicate === true
          ? 'This issue was already recorded. No duplicate service call was created.'
          : 'Service call created in the local development database.' });
      setTicketDraft(null);
      setTicketFiles([]);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  // Ticket creation is deliberately user-initiated. An answer that needs
  // technician help may show the optional action, but never opens the form.

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
        <div className="user-meta"><div className="user-name">Real-data demo</div><div className="user-role">Local · development data</div></div></div>
        {assets && <button type="button" className="btn btn-secondary btn-sm real-demo-lock" onClick={() => {
          initializeThreads(''); setToken(''); setAssets(null); setAssetId(''); setView('chat'); }}><LogOut size={14} /> Lock Demo</button>}</div>
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
            <p>Approved manuals in an isolated local index, local BGE-small + V5 retrieval, and a separate development database.</p>
            <span>✓ Service calls stay in the local development database</span><span>✓ Source access checked when opened</span></aside>
        </div> : view === 'assets' ? <>
          <div className="real-demo-stats"><div className="kpi-card"><div className="kpi-icon purple"><Package size={22} /></div>
            <div><div className="kpi-value">{assets.length}</div><div className="kpi-label">Verified assets</div></div></div>
            <div className="kpi-card"><div className="kpi-icon green"><BookOpen size={22} /></div>
              <div><div className="kpi-value">{assets.filter(item => item.documents.length).length}</div><div className="kpi-label">Within document scope</div></div></div></div>
          <section className="card"><div className="card-header"><h2 className="card-title"><Package size={20} /> Company Equipment</h2>
            <span className="badge badge-indexed">Read-only inventory</span></div>
            <div className="real-demo-inventory">{assets.map(item => <button className="real-demo-inventory-item" type="button" key={item.id}
              onClick={() => { initializeThreads(String(item.id)); setAssetId(String(item.id)); setView('chat'); }}>
              <span className="real-demo-icon"><Package size={18} /></span><span><strong>{item.name}</strong><small>{item.code} · {item.model || 'Model not listed'}</small></span>
              <em>{manualStatus(item)}</em></button>)}</div></section>
        </> : <>
          <div className="real-demo-columns real-demo-conversation-layout"><section className="card real-demo-equipment">
            <h2 className="card-title"><Building2 size={20} /> Equipment Selection</h2>
            <div className="form-group"><label className="form-label" htmlFor="demo-asset">Equipment / Asset</label>
              <select className="form-control" id="demo-asset" value={assetId}
                onChange={event => { initializeThreads(event.target.value); setAssetId(event.target.value); }}>
                <option value="">Select equipment before asking questions</option>
                {assets.map(item => <option key={item.id} value={item.id}>{item.name} ({item.code})</option>)}</select></div>
            {!asset && <p className="real-demo-doc-status" role="status">
              Select the equipment you want to ask about. Servy will not choose an asset automatically.
            </p>}
            {asset && <div className="real-demo-asset-detail"><h3>{asset.name}</h3>
              <dl><div><dt>Asset code</dt><dd>{asset.code}</dd></div>
                <div><dt>Product</dt><dd>{asset.product || 'Not listed'}</dd></div>
                {asset.site && <div><dt>Site</dt><dd>{asset.site}</dd></div>}</dl>
              {!asset.documents.length && <p className="real-demo-doc-status"><FileText size={17} />
                Technical-document access unavailable in this demo</p>}
              {asset.manuals_ready === false && <p className="real-demo-doc-status" role="status"><FileText size={17} />
                {manualStatus(asset)}. Questions are disabled until all required manuals are available.</p>}</div>}</section>

            <section className="card real-demo-assistant real-demo-chat-panel" aria-label="Ask Servy conversation">
              <div className="real-demo-assistant-header">
                <Bot size={25} />
                <div className="real-demo-chat-heading">
                  <h2>Ask Servy</h2>
                  <p>Continuous conversation · Approved original manuals · Cited answers</p>
                </div>
                <button type="button" className="real-demo-new-chat" onClick={startNewChat} disabled={busy}>
                  <Plus size={16} /> New chat
                </button>
              </div>
              <div className="real-demo-thread-index" aria-label="Conversations in this tab">
                <span className="real-demo-thread-label">Chats · saved locally</span>
                <div className="real-demo-thread-items">
                  {threads.map(item => <div className={[
                      'real-demo-thread-item', item.id === activeThreadId ? 'is-active' : ''
                    ].join(' ')} key={item.id}>
                    <button type="button" className="real-demo-thread-switch"
                      title={item.title} aria-current={item.id === activeThreadId ? 'true' : undefined}
                      disabled={busy} onClick={() => openThread(item.id)}>{item.title}</button>
                    <button type="button" className="real-demo-thread-delete" disabled={busy}
                      title={'Delete ' + item.title} aria-label={'Delete conversation: ' + item.title}
                      onClick={() => deleteThread(item.id)}><Trash2 size={13} /></button>
                  </div>)}
                </div>
              </div>
              <div className="real-demo-chat-thread" role="log" aria-label="Conversation messages" aria-live="polite" aria-busy={busy}>
                {messages.length === 0 && <div className="real-demo-chat-empty">
                  <Bot size={28} />
                  <h3>{asset ? 'Ask about this equipment' : 'Select equipment first'}</h3>
                  <p>{asset
                    ? 'Describe a symptom or fault code. Continue asking follow-up questions; Servy will use the earlier customer reports to retrieve relevant approved documentation.'
                    : 'Choose the exact equipment in Equipment Selection before starting a conversation. Each chat stays bound to that asset and its approved manuals.'}</p>
                  <small>Local demo: conversations are saved in the isolated development database and reload after a refresh. Nothing is written to company records.</small>
                </div>}
                {messages.map((message, index) => (
                  <div key={message.id} className={`real-demo-chat-message ${message.role === 'user' ? 'is-user' : 'is-assistant'}`}>
                    <div className="real-demo-chat-avatar">{message.role === 'user' ? 'You' : <Bot size={19} />}</div>
                    <div className="real-demo-chat-bubble">
                      {message.role === 'user' ? <div className="real-demo-chat-user-text">{message.text}</div>
                        : message.pending ? <>
                            <AssistantTypingIndicator />
                          </>
                          : message.errorMessage ? <>
                            <p className="servy-chat-error" role="alert">{message.errorMessage}</p>
                          </> : <>
                            <div className="real-demo-chat-answer-heading">
                              <strong>Servy</strong>
                              <span className={message.data?.response_type === 'DIAGNOSTIC_GUIDANCE'
                                ? 'real-demo-chat-guidance' : 'real-demo-chat-limited'}>
                                {({
                                  DIAGNOSTIC_GUIDANCE: 'Cited answer',
                                  TECHNICIAN_HANDOFF: 'Technician handoff',
                                  CLARIFICATION: message.data?.reason_code === 'CUSTOMER_REPORT_CAPTURED'
                                    ? 'Customer detail captured' : 'Clarification',
                                  INSUFFICIENT_EVIDENCE: message.data?.reason_code === 'OUT_OF_SCOPE_QUERY'
                                    ? 'Outside manual scope' : 'More evidence needed',
                                  ESCALATION_REQUIRED: 'Safety notice',
                                  POLICY_VIOLATION: 'Not permitted',
                                })[message.data?.response_type] || 'Response'}
                              </span>
                            </div>
                            {message.data?.response_type === 'TECHNICIAN_HANDOFF'
                              && message.data?.handoff ? (
                                <section className="real-demo-handoff" aria-label="Technician handoff checklist">
                                  {message.data.handoff.reported_details?.length > 0 && (
                                    <div>
                                      <strong>{message.data.handoff.reported_details_label}</strong>
                                      <ul>
                                        {message.data.handoff.reported_details.map((detail, detailIndex) => (
                                          <li key={detailIndex}>{detail.text}</li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}
                                  {message.data.handoff.checklist?.length > 0 && (
                                    <div>
                                      <strong>{message.data.handoff.checklist_label}</strong>
                                      <ul>
                                        {message.data.handoff.checklist.map((item, itemIndex) => (
                                          <li key={itemIndex}>{item}</li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}
                                  {message.data.handoff.follow_up_question && (
                                    <p className="real-demo-handoff-followup">
                                      {message.data.handoff.follow_up_question}
                                    </p>
                                  )}
                                  <p className="real-demo-handoff-note">
                                    This checklist prepares a handoff; it is not a diagnosis or repair procedure.
                                  </p>
                                </section>
                              ) : message.data?.presentation ? <CustomerAnswer
                              assetId={Number(assetId)}
                              presentation={message.data.presentation}
                              loadSource={url => call(gatewaySourceUrl(url), token)}
                            /> : <p>{message.data?.message || 'No response is available.'}</p>}
                            {message.data?.reason_code === 'CUSTOMER_REPORT_CAPTURED' && (
                              <p className="real-demo-dialogue-meta">
                                This is a conversation-state acknowledgement, not a new knowledge-base answer or a diagnosis. The next follow-up question is shown above.
                              </p>
                            )}
                            {message.data?.response_type === 'DIAGNOSTIC_GUIDANCE' && (
                              <details className="real-demo-dialogue-meta">
                                <summary>How this answer was prepared</summary>
                                <p>The technical statements are drawn from approved company sources.
                                  {message.data.engine === 'source-grounded-qwen'
                                    ? ' Local Qwen generated plain-language explanations using complete, reauthorized approved-manual passages. Original claim sentences remain intact, and each generated explanation links to both its sentence citation and the full source-passage citation. Output that fails source validation falls back to the extractive answer.'
                                    : message.data.engine === 'source-structured-qwen'
                                      ? ' Local Qwen helped phrase already-approved factual and possible-cause claims. Their exact source sentences and citations are retained; customer procedures and observations remain source-exact.'
                                      : message.data.engine === 'source-structured-extractive'
                                        ? ' The displayed technical claims are extracted from authorized passages and linked to exact citations; a generative language model did not rewrite them.'
                                        : message.data.dialogue_planner === 'local-ollama-structured'
                                          ? ' A local language model helped select the order of source-backed claims and a pre-approved follow-up.'
                                          : ' No generative language model rewrote the technical claims.'}
                                </p>
                                {message.data.answer_generation && (
                                  <p>
                                    Qwen answer generation: {message.data.answer_generation.status}.
                                    {message.data.answer_generation.model
                                      ? ' Model: ' + message.data.answer_generation.model + '.'
                                      : ''}
                                    {' '}Explanations applied: {message.data.answer_generation.applied_claims ?? 0}.
                                    {message.data.answer_generation.error_type
                                      ? ' Error type: ' + message.data.answer_generation.error_type + '.'
                                      : ''}
                                  </p>
                                )}
                                <p>
                                  Retrieval: {message.data.retrieval_attempts ?? 0} pass(es)
                                  {message.data.retrieval_backend ? ' via ' + message.data.retrieval_backend : ''}.
                                  {' '}Reranking: {message.data.reranker_applied ? 'applied' : 'not applied'}.
                                  {' '}Citation-backed passages shown: {message.data.presentation?.citations?.length ?? message.data.citations?.length ?? 0}.
                                </p>
                                {message.data.presentation?.document_coverage && (
                                  <p>
                                    Approved-document scan: {message.data.presentation.document_coverage.scan_completed === false
                                      ? 'not completed; no complete-coverage claim is made.'
                                      : (message.data.presentation.document_coverage.documents_searched ?? 0) + ' of ' +
                                        (message.data.presentation.document_coverage.approved_documents ?? 0) +
                                        ' approved documents; ' +
                                        (message.data.presentation.document_coverage.passages_searched ?? 0) +
                                        ' passages searched.'}
                                  </p>
                                )}
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
                  <button className="btn btn-primary" type="submit"
                    disabled={busy || !question.trim() || !asset?.can_query}>
                    <Send size={16} /> Send
                  </button>
                </div>
              </form>
            </section>
          </div>
          {messages.at(-1)?.role === 'assistant' && assignmentNotice(messages.at(-1)?.data)
            && !ticketConfirmation && (
            <section className="card real-demo-ticket-action" role="status"
              aria-label="Issue assignment status">
              <div><strong>{messages.at(-1).data.issue_assignment.status === 'ALREADY_ASSIGNED'
                ? 'Issue already assigned' : 'Issue needs the service team'}</strong>
                <p>{assignmentNotice(messages.at(-1).data)}</p></div>
            </section>
          )}
          {messages.at(-1)?.role === 'assistant' && continuation
            && messages.at(-1)?.data?.issue_assignment?.status !== 'ALREADY_ASSIGNED'
            && !ticketDraft && !ticketConfirmation && (
            <section className="card real-demo-ticket-action" aria-label="Ticket escalation options">
              <div><strong>Still need assistance?</strong>
                <p>Raise a ticket with the conversation attached for the technician.</p></div>
              <button className="btn btn-secondary" type="button" onClick={startTicket} disabled={busy}>
                <Ticket size={17} /> Create Ticket
              </button>
            </section>
          )}
          {ticketDraft && <div className="real-demo-ticket-overlay" role="presentation"
            onMouseDown={event => { if (event.target === event.currentTarget) closeTicketPanel(); }}>
            <section className="real-demo-ticket-panel" role="dialog" aria-modal="true"
              aria-labelledby="real-demo-ticket-title">
              <header className="real-demo-ticket-modal-heading">
                <div><h2 id="real-demo-ticket-title">Create New Ticket</h2>
                  <p>Fill in the details below to raise a service ticket.</p></div>
                <button type="button" className="real-demo-ticket-close" aria-label="Close ticket form"
                  onClick={closeTicketPanel} disabled={busy}><X size={19} /></button>
              </header>
              <div className="real-demo-ticket-panel-body">
                <section className="real-demo-ticket-details" aria-labelledby="real-demo-ticket-details-title">
                  <h3 id="real-demo-ticket-details-title">TICKET DETAILS</h3>
                  <form onSubmit={createTicket}>
                    <div className="real-demo-ticket-field real-demo-ticket-types">
                      <span className="real-demo-ticket-label">Ticket Type <b>*</b></span>
                      <div className="real-demo-ticket-type-list" role="group" aria-label="Ticket Type">
                        {TICKET_TYPES.map(type => <button key={type} type="button"
                          className={ticketDraft.call_type === type ? 'is-selected' : ''}
                          aria-pressed={ticketDraft.call_type === type}
                          onClick={() => updateTicketDraft('call_type', type)}>{type}</button>)}
                      </div>
                    </div>

                    <div className="real-demo-ticket-grid real-demo-ticket-grid-three">
                      <label className="real-demo-ticket-field"><span>Customer</span>
                        <input className="form-control" value={asset?.customer_name || `Customer ${asset?.customer_id || ''}`}
                          readOnly aria-readonly="true" /></label>
                      <label className="real-demo-ticket-field"><span>Facility <b>*</b></span>
                        <input className="form-control" value={asset?.site || ''} readOnly aria-readonly="true" /></label>
                      <label className="real-demo-ticket-field"><span>Contact Person</span>
                        <input className="form-control" value="From your customer record" readOnly
                          aria-readonly="true" /></label>
                    </div>

                    <label className="real-demo-ticket-field real-demo-ticket-equipment"><span>Facility Equipment <b>*</b></span>
                      <input className="form-control" value={`${asset?.name || ''} (${asset?.code || ''})`}
                        readOnly aria-readonly="true" /></label>

                    <div className="real-demo-ticket-grid real-demo-ticket-grid-two">
                      <label className="real-demo-ticket-field"><span>Severity Level <b>*</b></span>
                        <select className="form-control" value={ticketDraft.severity} required
                          onChange={event => updateTicketDraft('severity', event.target.value)}>
                          <option value="">Select Severity Level</option>
                          {SEVERITIES.map(value => <option key={value} value={value}>{value}</option>)}
                        </select></label>
                      <label className="real-demo-ticket-field"><span>Priority Level</span>
                        <select className="form-control" value={ticketDraft.priority}
                          onChange={event => updateTicketDraft('priority', event.target.value)}>
                          <option value="">Select Priority Level</option>
                          {PRIORITIES.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
                        </select></label>
                    </div>

                    <div className="real-demo-ticket-grid real-demo-ticket-grid-two">
                      <label className="real-demo-ticket-field"><span>Machine Status <b>*</b></span>
                        <select className="form-control" value={ticketDraft.machine_status} required
                          onChange={event => updateTicketDraft('machine_status', event.target.value)}>
                          <option value="">Select Machine Status</option>
                          {MACHINE_STATUSES.map(value => <option key={value} value={value}>{value}</option>)}
                        </select></label>
                      <label className="real-demo-ticket-field"><span>Complaint Type <b>*</b></span>
                        <select className="form-control" value={ticketDraft.complaint_type} required
                          onChange={event => updateTicketDraft('complaint_type', event.target.value)}>
                          <option value="">Select Complaint</option>
                          {COMPLAINT_TYPES.map(value => <option key={value} value={value}>{value}</option>)}
                        </select></label>
                    </div>

                    <div className="real-demo-ticket-field real-demo-ticket-description">
                      <span>Description (prepared by Servy from this whole conversation)</span>
                      {ticketDraft.previewDetail && <p className="real-demo-error" role="alert">{ticketDraft.previewDetail}</p>}
                      {ticketDraft.preview?.missing_fields?.length > 0 && <p className="real-demo-error" role="alert">
                        Contact details needed before a technician can be dispatched:{' '}
                        {ticketDraft.preview.missing_fields.map(field => ({
                          contact_name: 'contact name', phone_or_email: 'a phone number or email address',
                          asset: 'the selected equipment' })[field] || field).join(', ')}.
                        Update your customer record; Servy does not fill these in.</p>}
                      {ticketDraft.preview?.sections && <div className="form-control real-demo-ticket-preview"
                        data-testid="ticket-description-preview" style={{ height: 'auto', maxHeight: 320, overflowY: 'auto' }}>
                        <small>{ticketDraft.preview.message_count} saved messages</small>
                        {Object.entries(ticketDraft.preview.sections).map(([title, items]) => <div key={title}>
                          <strong>{title}</strong>
                          <ul>{(items || []).map((item, itemIndex) => <li key={itemIndex}>{item}</li>)}</ul>
                        </div>)}
                      </div>}
                    </div>

                    <div className="real-demo-ticket-field">
                      <span className="real-demo-ticket-label">Issue Documentation</span>
                      <label className="real-demo-ticket-upload" htmlFor="real-demo-ticket-files"
                        onDragOver={event => event.preventDefault()}
                        onDrop={event => { event.preventDefault(); addTicketFiles(event.dataTransfer.files); }}>
                        <FilePlus2 size={25} />
                        <strong>Click or drag to upload</strong>
                        <span>Images, PDF or text · up to 10 files, 25 MB total · stored with the service call</span>
                        <input id="real-demo-ticket-files" type="file" multiple
                          onChange={event => { addTicketFiles(event.target.files); event.target.value = ''; }} />
                      </label>
                      {ticketFiles.length > 0 && <ul className="real-demo-ticket-files">
                        {ticketFiles.map((file, index) => <li key={`${file.name}:${file.size}:${file.lastModified}`}>
                          <span>{file.name} · {(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                          <button type="button" aria-label={`Remove ${file.name}`} disabled={busy}
                            onClick={() => setTicketFiles(current => current.filter((_, i) => i !== index))}>
                            <X size={15} /></button></li>)}
                      </ul>}
                    </div>
                    {error && <p className="real-demo-error" role="alert">{error}</p>}
                    <div className="real-demo-ticket-modal-actions">
                      <button type="button" className="btn btn-secondary" onClick={closeTicketPanel} disabled={busy}>Cancel</button>
                      <button type="submit" className="btn btn-primary" disabled={busy || !ticketDraft.complaint_type
                        || !ticketDraft.preview?.can_create}>
                        <Ticket size={16} /> {busy ? 'Creating…' : 'Create Ticket'}
                      </button>
                    </div>
                  </form>
                </section>
              </div>
            </section>
          </div>}
          {ticketConfirmation && <div className="real-demo-ticket-confirmation" role="status">
            <strong>{ticketConfirmation.duplicate ? 'Ticket already exists' : 'Ticket created'}</strong>
            <span>{ticketConfirmation.ticket.ticket_ref || `SRV-${ticketConfirmation.ticket.servy_id}`}</span>
            {ticketConfirmation.detail && <small>{ticketConfirmation.detail}</small>}
          </div>}
        </>}
        {error && !ticketDraft && <p className="real-demo-error" role="alert">{error}</p>}
      </div>
    </main>
  </div>;
}
