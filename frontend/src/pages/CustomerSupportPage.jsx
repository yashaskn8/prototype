import { CustomerAnswer } from '../components/CustomerAnswer';
import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import { useAuth } from '../AuthContext';
import {
  Bot,
  Send,
  CheckCircle2,
  AlertTriangle,
  FileText,
  ShieldCheck,
  RefreshCw,
  Building,
  Ticket,
  HelpCircle,
  ChevronRight,
  MessageSquare,
  ArrowRight,
  History,
} from 'lucide-react';

/**
 * Canonical array normalizer to prevent .map() runtime crashes
 * when backend returns { conversations: [...] }, { tickets: [...] }, or { messages: [...] }
 */
export function normalizeArray(data, primaryKey) {
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (primaryKey && Array.isArray(data[primaryKey])) return data[primaryKey];
  if (Array.isArray(data.results)) return data.results;
  if (Array.isArray(data.data)) return data.data;
  return [];
}


export function CustomerSupportPage() {
  const { user } = useAuth();
  const [contextData, setContextData] = useState(null);
  const [loadingContext, setLoadingContext] = useState(true);

  // Tab Navigation: 'diagnostics' | 'conversations' | 'tickets'
  const [activeTab, setActiveTab] = useState('diagnostics');

  // Diagnostic State
  const [selectedSite, setSelectedSite] = useState('');
  const [siteAssets, setSiteAssets] = useState([]);
  const [loadingAssets, setLoadingAssets] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [query, setQuery] = useState('');
  const [customClarification, setCustomClarification] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Durable Session & Conversation Context
  const [conversationUuid, setConversationUuid] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const chatEndRef = useRef(null);
  const [activeSessionNotice, setActiveSessionNotice] = useState(null);
  const [resolving, setResolving] = useState(false);
  const [escalating, setEscalating] = useState(false);
  const [resolutionNotice, setResolutionNotice] = useState(null);
  const [escalationResult, setEscalationResult] = useState(null);

  // Persistent History / Tracking State
  const [conversations, setConversations] = useState([]);
  const [conversationsTotal, setConversationsTotal] = useState(0);
  const [conversationsOffset, setConversationsOffset] = useState(0);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [conversationsError, setConversationsError] = useState(null);

  const [tickets, setTickets] = useState([]);
  const [ticketsTotal, setTicketsTotal] = useState(0);
  const [ticketsOffset, setTicketsOffset] = useState(0);
  const [loadingTickets, setLoadingTickets] = useState(false);
  const [ticketsError, setTicketsError] = useState(null);

  const [activeTranscript, setActiveTranscript] = useState(null);
  const [transcriptMessages, setTranscriptMessages] = useState([]);
  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const [transcriptError, setTranscriptError] = useState(null);
  const [replyText, setReplyText] = useState('');
  const [sendingReply, setSendingReply] = useState(false);

  // Request race condition tracking: ensure late responses for previous assets are discarded
  const activeAssetRef = useRef('');
  const assetLoadVersion = useRef(0);

  // Load authorized customer sites & initial context
  async function loadContext() {
    setLoadingContext(true);
    setError(null);
    try {
      const data = await api.get('/api/customer-support/context/');
      setContextData(data || {});
      const sites = normalizeArray(data, 'sites');
      if (sites.length > 0) {
        const firstSiteId = String(sites[0].id);
        setSelectedSite(firstSiteId);
        await loadAssetsForSite(firstSiteId, data.active_session);
      } else {
        setSelectedSite('');
        setSiteAssets([]);
        setSelectedAsset('');
      }
    } catch (err) {
      setError(err.detail || 'Failed to load customer profile context.');
    } finally {
      setLoadingContext(false);
    }
  }

  // Load customer's conversations with explicit contract normalization
  async function loadConversations(offset = 0) {
    setLoadingConversations(true);
    setConversationsError(null);
    try {
      const res = await api.get(`/api/customer-support/conversations/?limit=10&offset=${offset}`);
      const list = normalizeArray(res, 'conversations');
      setConversations(list);
      setConversationsTotal(typeof res?.count === 'number' ? res.count : list.length);
      setConversationsOffset(offset);
    } catch (err) {
      console.error('Failed to load conversations', err);
      setConversations([]);
      setConversationsError(err.detail || 'Failed to load conversation history.');
    } finally {
      setLoadingConversations(false);
    }
  }

  // Load customer's service tickets with explicit contract normalization
  async function loadTickets(offset = 0) {
    setLoadingTickets(true);
    setTicketsError(null);
    try {
      const res = await api.get(`/api/customer-support/tickets/?limit=10&offset=${offset}`);
      const list = normalizeArray(res, 'tickets');
      setTickets(list);
      setTicketsTotal(typeof res?.count === 'number' ? res.count : list.length);
      setTicketsOffset(offset);
    } catch (err) {
      console.error('Failed to load tickets', err);
      setTickets([]);
      setTicketsError(err.detail || 'Failed to load service tickets.');
    } finally {
      setLoadingTickets(false);
    }
  }

  // Dynamically load assets belonging to the selected site
  async function loadAssetsForSite(siteId, initialActiveSession = null) {
    const requestVersion = ++assetLoadVersion.current;
    if (!siteId) {
      setSiteAssets([]);
      setSelectedAsset('');
      activeAssetRef.current = '';
      return;
    }
    setLoadingAssets(true);
    try {
      const res = await api.get(`/api/sites/${siteId}/assets/`);
      if (requestVersion !== assetLoadVersion.current) return;
      const assets = normalizeArray(res, 'assets');
      setSiteAssets(assets);
      if (assets.length > 0) {
        const firstAssetId = String(assets[0].id);
        setSelectedAsset(firstAssetId);
        activeAssetRef.current = firstAssetId;
        checkActiveSessionForAsset(firstAssetId, initialActiveSession);
      } else {
        setSelectedAsset('');
        activeAssetRef.current = '';
        setActiveSessionNotice(null);
      }
    } catch (err) {
      if (requestVersion !== assetLoadVersion.current) return;
      console.error('Failed to load site assets', err);
      setSiteAssets([]);
      setSelectedAsset('');
      activeAssetRef.current = '';
    } finally {
      if (requestVersion === assetLoadVersion.current) setLoadingAssets(false);
    }
  }

  async function checkActiveSessionForAsset(assetId, preloadedSession = null) {
    if (preloadedSession && String(preloadedSession.asset_id) === String(assetId)) {
      setActiveSessionNotice(preloadedSession);
      return;
    }
    try {
      const res = await api.get(`/api/customer-support/context/?asset_id=${assetId}`);
      if (activeAssetRef.current !== String(assetId)) return;
      if (res.active_conversation) {
        setConversationUuid(res.active_conversation.conversation_uuid);
        if (res.active_conversation.messages && res.active_conversation.messages.length > 0) {
          setMessages(res.active_conversation.messages);
          const lastAsst = [...res.active_conversation.messages].reverse().find(m => m.sender_role === 'assistant');
          if (lastAsst && lastAsst.structured_contract) {
            setResult(lastAsst.structured_contract);
          }
        } else {
          setMessages([]);
          setResult(null);
        }
      } else if (res.active_session) {
        setActiveSessionNotice(res.active_session);
        setConversationUuid(null);
        setMessages([]);
        setResult(null);
      } else {
        setActiveSessionNotice(null);
        setConversationUuid(null);
        setMessages([]);
        setResult(null);
      }
    } catch {
      if (activeAssetRef.current === String(assetId)) setActiveSessionNotice(null);
    }
  }

  async function loadConversationMessages(convUuid) {
    if (!convUuid) return;
    setLoadingMessages(true);
    try {
      const res = await api.get(`/api/customer-support/conversations/${convUuid}/messages/`);
      const msgList = normalizeArray(res, 'messages');
      setMessages(msgList);
      const lastAsst = [...msgList].reverse().find(m => m.sender_role === 'assistant');
      if (lastAsst && lastAsst.structured_contract) {
        setResult(lastAsst.structured_contract);
      }
    } catch (e) {
      console.warn('Could not load messages', e);
    } finally {
      setLoadingMessages(false);
    }
  }

  function handleStartNewConversation() {
    if (messages.length > 0 && !window.confirm('Start a new troubleshooting conversation for this equipment? Your current conversation will remain saved in History.')) {
      return;
    }
    setConversationUuid(null);
    setMessages([]);
    setResult(null);
    setQuery('');
    setResolutionNotice(null);
    setEscalationResult(null);
    setActiveSessionNotice(null);
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    loadContext();
    loadConversations();
    loadTickets();
  }, []);

  function handleSiteChange(e) {
    assetLoadVersion.current += 1;
    setSubmitting(false);
    setResolutionNotice(null);
    setEscalationResult(null);
    setError(null);
    const newSiteId = e.target.value;
    setSelectedSite(newSiteId);
    setSelectedAsset('');
    activeAssetRef.current = '';
    setResult(null);
    setConversationUuid(null);
    setActiveSessionNotice(null);
    if (newSiteId) {
      loadAssetsForSite(newSiteId);
    } else {
      setSiteAssets([]);
    }
  }

  function handleAssetChange(e) {
    setSubmitting(false);
    const newAssetId = e.target.value;
    setSelectedAsset(newAssetId);
    activeAssetRef.current = newAssetId;
    setResult(null);
    setError(null);
    setConversationUuid(null);
    setResolutionNotice(null);
    setEscalationResult(null);
    if (newAssetId) {
      checkActiveSessionForAsset(newAssetId);
    } else {
      setActiveSessionNotice(null);
    }
  }

  function handleResumeSession() {
    if (activeSessionNotice) {
      if (activeSessionNotice.complaint) {
        setQuery(activeSessionNotice.complaint);
      }
      if (activeSessionNotice.contract) {
        setResult(activeSessionNotice.contract);
        if (activeSessionNotice.contract.conversation_uuid) {
          setConversationUuid(activeSessionNotice.contract.conversation_uuid);
        }
        setActiveSessionNotice(null);
        return;
      }
    }
  }

  async function resumeConversationFromHistory(conv) {
    setActiveTab('diagnostics');
    const siteId = String(conv.site_id || (conv.site ? conv.site.id : ''));
    setSelectedSite(siteId);
    if (siteId) {
      await loadAssetsForSite(siteId);
    }
    const assetId = String(conv.asset_id || (conv.asset ? conv.asset.id : ''));
    setSelectedAsset(assetId);
    activeAssetRef.current = assetId;
    const convUuid = String(conv.conversation_uuid || conv.id);
    setConversationUuid(convUuid);
    setQuery('');
    setResult(null);
    setResolutionNotice(null);
    setEscalationResult(null);
    await loadConversationMessages(convUuid);

    // If conversation is already escalated or has a linked service call, load the ticket
    if (conv.linked_service_call_id || conv.linked_service_call) {
      try {
        const ticketRes = await api.get(`/api/customer-support/conversations/${convUuid}/ticket/`);
        setEscalationResult(ticketRes);
      } catch (e) {
        console.warn('Could not load linked ticket', e);
      }
    }
  }

  async function viewConversationTranscript(conv) {
    setActiveTranscript(conv);
    setLoadingTranscript(true);
    setTranscriptError(null);
    try {
      const convUuid = conv.conversation_uuid || conv.id;
      const res = await api.get(`/api/customer-support/conversations/${convUuid}/messages/`);
      setTranscriptMessages(normalizeArray(res, 'messages'));
    } catch (e) {
      console.error('Failed to load transcript messages', e);
      setTranscriptMessages([]);
      setTranscriptError(e.detail || 'Failed to load conversation transcript.');
    } finally {
      setLoadingTranscript(false);
    }
  }

  async function handleSendReply(e) {
    if (e) e.preventDefault();
    if (!replyText.trim() || !activeTranscript) return;
    const convUuid = activeTranscript.conversation_uuid || activeTranscript.id;
    setSendingReply(true);
    try {
      const res = await api.post(`/api/customer-support/conversations/${convUuid}/messages/`, {
        message: replyText.trim(),
      });
      setReplyText('');
      await viewConversationTranscript(activeTranscript);
      loadConversations(conversationsOffset);
    } catch (err) {
      setTranscriptError(err.detail || 'Failed to post message.');
    } finally {
      setSendingReply(false);
    }
  }

  async function submitDiagnosticQuery(queryText, clarificationAnswer = '') {
    const currentAssetId = selectedAsset;
    if (!queryText.trim() || !selectedSite || !currentAssetId) return;

    setSubmitting(true);
    setError(null);
    setResolutionNotice(null);
    setEscalationResult(null);

    const fullTurnInput = clarificationAnswer ? `${queryText} (${clarificationAnswer})` : queryText;
    const tempUserMsg = {
      id: 'temp-' + Date.now(),
      sender_role: 'customer',
      message_text: fullTurnInput,
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, tempUserMsg]);
    setQuery('');
    setCustomClarification('');

    try {
      let turnContract = null;
      if (conversationUuid) {
        const payload = {
          message: queryText.trim(),
        };
        if (clarificationAnswer) {
          payload.clarification_response = clarificationAnswer.trim();
        }
        const res = await api.post(`/api/customer-support/conversations/${conversationUuid}/messages/`, payload);
        if (activeAssetRef.current === currentAssetId) {
          turnContract = res.contract;
          setMessages(prev => [
            ...prev.filter(m => m.id !== tempUserMsg.id),
            res.user_message || tempUserMsg,
            res.assistant_message,
          ]);
        }
      } else {
        const payload = {
          site_id: selectedSite,
          asset_id: currentAssetId,
          question: queryText.trim(),
          customer_id: contextData?.customer?.id,
          session_id: activeSessionNotice?.session_id || null,
        };
        if (clarificationAnswer) {
          payload.clarification_response = clarificationAnswer.trim();
        }
        const res = await api.post('/api/customer-support/query/', payload);
        if (activeAssetRef.current === currentAssetId) {
          turnContract = res;
          if (res.conversation_uuid) {
            setConversationUuid(res.conversation_uuid);
          }
          const asstMsg = {
            id: 'asst-' + Date.now(),
            sender_role: 'assistant',
            message_text: res.answer || res.summary || '',
            structured_contract: res,
            source_refs: res.citations || res.references || [],
            created_at: new Date().toISOString(),
          };
          setMessages(prev => [
            ...prev.filter(m => m.id !== tempUserMsg.id),
            tempUserMsg,
            asstMsg,
          ]);
        }
      }

      if (activeAssetRef.current === currentAssetId && turnContract) {
        setResult(turnContract);
        loadConversations();
      }
    } catch (err) {
      if (activeAssetRef.current === currentAssetId) {
        setError(err.detail || 'Error communicating with AI assistant.');
        setMessages(prev => prev.filter(m => m.id !== tempUserMsg.id));
        setQuery(queryText);
      }
    } finally {
      if (activeAssetRef.current === currentAssetId) {
        setSubmitting(false);
      }
    }
  }

  function handleQuery(e) {
    e.preventDefault();
    submitDiagnosticQuery(query);
  }

  function handleClarificationOption(optionText) {
    submitDiagnosticQuery(query, optionText);
  }

  function handleCustomClarificationSubmit(e) {
    e.preventDefault();
    if (customClarification.trim()) {
      submitDiagnosticQuery(query, customClarification);
    }
  }

  async function handleResolve() {
    if (!result?.interaction_id || !result.can_resolve) return;
    setResolving(true);
    try {
      const res = await api.post(`/api/customer-support/interactions/${result.interaction_id}/resolve/`, { customer_reports_resolved: true });
      setResolutionNotice(res.detail || 'Marked as resolved. Thank you!');
      setActiveSessionNotice(null);
      loadConversations();
    } catch (err) {
      setError(err.detail || 'Could not update interaction.');
    } finally {
      setResolving(false);
    }
  }

  async function handleEscalate(escalationNote = '') {
    if (!result?.interaction_id) return;
    if (!window.confirm('Create a service ticket for this equipment? This saves a service request.')) return;
    setEscalating(true);
    try {
      const activeSessionId = result?.session_id || activeSessionNotice?.session_id || null;
      const payload = {
        note: escalationNote,
        confirm_ticket_creation: true,
        session_id: activeSessionId,
      };
      const res = await api.post(`/api/customer-support/interactions/${result.interaction_id}/escalate/`, payload);
      setEscalationResult(res);
      setActiveSessionNotice(null);
      loadConversations();
      loadTickets();
    } catch (err) {
      setError(err.detail || 'Could not escalate service ticket.');
    } finally {
      setEscalating(false);
    }
  }


  const selectedAssetObj = siteAssets.find(a => String(a.id) === String(selectedAsset));

  return (
    <div className="content-body">
      {/* Page Header */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <span className="badge" style={{ background: '#e0e7ff', color: '#4338ca', fontWeight: 700 }}>
            CUSTOMER AI SUPPORT PLATFORM
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Industrial Equipment Diagnostics & Service Coordination</span>
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Equipment Troubleshooting & Service Portal
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Resolve issues safely using approved manufacturer guidance and Knowledge Base procedures before creating a service dispatch.
        </p>
      </div>

      {/* Primary Navigation Tabs */}
      <div style={{
        display: 'flex',
        gap: '8px',
        borderBottom: '1px solid var(--border)',
        marginBottom: '20px',
        paddingBottom: '2px'
      }}>
        <button
          type="button"
          onClick={() => setActiveTab('diagnostics')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 16px',
            border: 'none',
            background: 'none',
            borderBottom: activeTab === 'diagnostics' ? '3px solid var(--primary)' : '3px solid transparent',
            color: activeTab === 'diagnostics' ? 'var(--primary)' : 'var(--text-muted)',
            fontWeight: activeTab === 'diagnostics' ? 700 : 500,
            cursor: 'pointer',
            fontSize: '0.9rem',
          }}
        >
          <Bot size={18} /> Guided Diagnostics
        </button>

        <button
          type="button"
          onClick={() => { setActiveTab('conversations'); loadConversations(); }}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 16px',
            border: 'none',
            background: 'none',
            borderBottom: activeTab === 'conversations' ? '3px solid var(--primary)' : '3px solid transparent',
            color: activeTab === 'conversations' ? 'var(--primary)' : 'var(--text-muted)',
            fontWeight: activeTab === 'conversations' ? 700 : 500,
            cursor: 'pointer',
            fontSize: '0.9rem',
          }}
        >
          <MessageSquare size={18} /> Support History ({conversations.length})
        </button>

        <button
          type="button"
          onClick={() => { setActiveTab('tickets'); loadTickets(); }}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 16px',
            border: 'none',
            background: 'none',
            borderBottom: activeTab === 'tickets' ? '3px solid var(--primary)' : '3px solid transparent',
            color: activeTab === 'tickets' ? 'var(--primary)' : 'var(--text-muted)',
            fontWeight: activeTab === 'tickets' ? 700 : 500,
            cursor: 'pointer',
            fontSize: '0.9rem',
          }}
        >
          <Ticket size={18} /> Service Tickets ({tickets.length})
        </button>
      </div>

      {error && (
        <div className="alert alert-danger" role="alert">
          <AlertTriangle size={18} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      {resolutionNotice && (
        <div className="alert alert-success" role="status">
          <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
          <div>
            <strong>Issue Reported Resolved</strong>
            <div style={{ fontSize: '0.82rem', marginTop: '2px' }}>{resolutionNotice}</div>
          </div>
        </div>
      )}

      {escalationResult && (
        <div className="alert alert-info" role="status" style={{ display: 'block' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, marginBottom: '6px' }}>
            <Ticket size={18} /> Service Ticket Created: #{escalationResult.servy_id}
          </div>
          <p style={{ fontSize: '0.85rem', marginBottom: '6px' }}>
            {escalationResult.status === 'assigned'
              ? 'A certified service technician has been assigned to your case. Your diagnostic history has been safely handed off.'
              : 'Your service request has been logged and queued for technician dispatch.'}
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Status: <strong>{escalationResult.status?.toUpperCase() || 'OPEN'}</strong> | Priority: <strong>{escalationResult.priority?.toUpperCase() || 'NORMAL'}</strong>
            {escalationResult.technician && escalationResult.technician !== 'Unassigned' && (
              <span> | Technician: <strong>{escalationResult.technician}</strong></span>
            )}
          </div>
        </div>
      )}

      {/* TAB 1: GUIDED DIAGNOSTICS */}
      {activeTab === 'diagnostics' && (
        <div>
          {activeSessionNotice && !result && (
            <div className="alert alert-warning" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <RefreshCw size={16} />
                <span style={{ fontSize: '0.85rem' }}>
                  You have an active troubleshooting session for this equipment (Issue: <em>"{activeSessionNotice.complaint}"</em>).
                </span>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-outline-primary"
                onClick={handleResumeSession}
                disabled={submitting}
              >
                Resume Diagnostic
              </button>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px', alignItems: 'start' }}>

            {/* Equipment Selector Sidebar Card */}
            <div className="card">
              <div className="card-title" style={{ marginBottom: '16px' }}>
                <Building size={18} color="var(--primary)" /> Equipment Selection
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="siteSelect">Facility / Site</label>
                <select
                  id="siteSelect"
                  className="form-control"
                  value={selectedSite}
                  onChange={handleSiteChange}
                  disabled={loadingContext}
                >
                  {!selectedSite && <option value="">Select a Facility</option>}
                  {contextData?.sites?.map(s => (
                    <option key={s.id} value={s.id}>{s.name} ({s.city || 'Facility'})</option>
                  ))}
                </select>
                {contextData?.sites && contextData.sites.length === 0 && (
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    No facilities registered for this customer account.
                  </div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="assetSelect">Equipment / Asset</label>
                <select
                  id="assetSelect"
                  className="form-control"
                  value={selectedAsset}
                  onChange={handleAssetChange}
                  disabled={loadingContext || loadingAssets || siteAssets.length === 0}
                >
                  {!selectedAsset && <option value="">Select Equipment</option>}
                  {siteAssets.map(a => (
                    <option key={a.id} value={a.id}>
                      {a.name} {a.model_number ? `(${a.model_number})` : ''}
                    </option>
                  ))}
                </select>
                {loadingAssets && (
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    Loading equipment for selected facility...
                  </div>
                )}
                {!loadingAssets && selectedSite && siteAssets.length === 0 && (
                  <div style={{ fontSize: '0.75rem', color: '#b91c1c', marginTop: '4px' }}>
                    No active equipment found at this facility.
                  </div>
                )}
              </div>

              {selectedAssetObj && (
                <div style={{
                  background: 'var(--bg-app)',
                  padding: '12px',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border)',
                  marginBottom: '14px',
                  fontSize: '0.8rem'
                }}>
                  <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
                    {selectedAssetObj.name}
                  </div>
                  <div style={{ color: 'var(--text-muted)' }}>
                    Asset Code: <strong>{selectedAssetObj.asset_code || 'N/A'}</strong><br />
                    Model: <strong>{selectedAssetObj.model_number || 'Standard'}</strong><br />
                    Product: {selectedAssetObj.product__name || 'Industrial Equipment'}
                  </div>
                </div>
              )}

              {conversationUuid && (
                <div style={{
                  background: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: 'var(--radius-md)',
                  padding: '10px 12px',
                  marginBottom: '14px',
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                }}>
                  <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '2px' }}>
                    Persistent Incident ID:
                  </div>
                  <code style={{ fontSize: '0.72rem', wordBreak: 'break-all' }}>{conversationUuid}</code>
                </div>
              )}

              <div style={{
                background: 'var(--bg-app)',
                padding: '12px 14px',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border)',
                fontSize: '0.78rem',
                color: 'var(--text-muted)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
                  <ShieldCheck size={14} color="#10b981" /> Tenant Isolation Active
                </div>
                Customer: <strong>{contextData?.customer?.name || user?.username}</strong>. Diagnostic queries are grounded exclusively in authorized documentation for your equipment.
              </div>
            </div>

            {/* Unified Conversation Stream & Guided Diagnostic View */}
            <div className="rag-box" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '620px' }}>
              <div className="rag-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div className="rag-header-left">
                  <div className="rag-icon-badge">
                    <Bot size={20} />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>AI Troubleshooting Assistant</h3>
                    <span style={{ fontSize: '0.75rem', opacity: 0.85 }}>Source-Grounded Pre-Ticket Diagnostics</span>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {conversationUuid && (
                    <span className="badge" style={{ background: '#dcfce7', color: '#166534', fontSize: '0.72rem', fontWeight: 600 }}>
                      Active Conversation
                    </span>
                  )}
                  {conversationUuid && (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={handleStartNewConversation}
                      style={{ fontSize: '0.75rem', padding: '4px 10px', background: '#fff' }}
                      title="Start a fresh conversation for this equipment"
                    >
                      <RefreshCw size={12} style={{ marginRight: '4px' }} /> New Conversation
                    </button>
                  )}
                </div>
              </div>

              {/* Chat Messages Container */}
              <div style={{
                flex: 1,
                overflowY: 'auto',
                maxHeight: '520px',
                padding: '20px 24px',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
                background: '#fafafa',
              }}>
                {loadingMessages && (
                  <div style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
                    <RefreshCw size={20} className="spin" style={{ margin: '0 auto 8px auto', display: 'block' }} />
                    <span style={{ fontSize: '0.85rem' }}>Loading conversation history...</span>
                  </div>
                )}

                {!loadingMessages && messages.length === 0 && (
                  <div style={{
                    padding: '36px 20px',
                    textAlign: 'center',
                    background: '#fff',
                    borderRadius: 'var(--radius-md)',
                    border: '1px dashed var(--border)',
                  }}>
                    <Bot size={36} color="var(--primary)" style={{ margin: '0 auto 12px auto', display: 'block', opacity: 0.8 }} />
                    <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '6px', color: 'var(--text-main)' }}>
                      Start Troubleshooting {selectedAssetObj ? selectedAssetObj.name : 'Equipment'}
                    </h4>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '460px', margin: '0 auto 16px auto' }}>
                      Describe the symptoms or error you are observing. Ask Servy provides source-grounded diagnostic guidance and remembers your reported observations across multiple turns.
                    </p>

                    <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
                      {['Equipment is not cooling properly', 'Weak or restricted airflow', 'Water leaking from the unit', 'Unusual noise during operation'].map((prompt, idx) => (
                        <button
                          key={idx}
                          type="button"
                          className="btn btn-sm btn-outline-secondary"
                          style={{ fontSize: '0.78rem', borderRadius: '16px', background: '#fff' }}
                          onClick={() => { setQuery(prompt); }}
                          disabled={!selectedAsset || submitting}
                        >
                          {prompt}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map((msg, index) => {
                  const isUser = msg.sender_role === 'customer';
                  return (
                    <div
                      key={msg.id || index}
                      style={{
                        display: 'flex',
                        justifyContent: isUser ? 'flex-end' : 'flex-start',
                        gap: '10px',
                        alignItems: 'flex-start',
                      }}
                    >
                      {!isUser && (
                        <div style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '50%',
                          background: '#e0e7ff',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          marginTop: '2px',
                        }}>
                          <Bot size={18} color="#4338ca" />
                        </div>
                      )}

                      <div style={{
                        maxWidth: isUser ? '75%' : '90%',
                        background: isUser ? 'var(--primary)' : '#fff',
                        color: isUser ? '#fff' : 'var(--text-main)',
                        borderRadius: isUser ? '16px 16px 4px 16px' : '4px 16px 16px 16px',
                        padding: isUser ? '12px 16px' : '16px',
                        border: isUser ? 'none' : '1px solid var(--border)',
                        boxShadow: '0 2px 6px rgba(0,0,0,0.04)',
                        fontSize: '0.88rem',
                        lineHeight: 1.45,
                      }}>
                        {isUser ? (
                          <>
                            <div style={{ fontWeight: 600, fontSize: '0.74rem', opacity: 0.85, marginBottom: '4px' }}>
                              You
                            </div>
                            <div style={{ whiteSpace: 'pre-wrap' }}>{msg.message_text}</div>
                          </>
                        ) : (
                          <div>
                            {/* Render Structured Contract Presentation if available */}
                            {msg.structured_contract?.presentation ? (
                              <CustomerAnswer
                                presentation={msg.structured_contract.presentation}
                                assetId={selectedAsset}
                              />
                            ) : msg.structured_contract?.response_type === 'CLARIFICATION' ? (
                              <div style={{
                                background: '#fefce8',
                                border: '1px solid #fde68a',
                                borderRadius: 'var(--radius-md)',
                                padding: '14px',
                              }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#a16207', fontWeight: 700, fontSize: '0.9rem', marginBottom: '6px' }}>
                                  <HelpCircle size={16} /> Clarification Needed
                                </div>
                                <p style={{ fontSize: '0.86rem', color: '#854d0e', marginBottom: '10px' }}>
                                  {msg.structured_contract.clarification_question || msg.structured_contract.summary || msg.message_text}
                                </p>
                                {msg.structured_contract.clarification_options && (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                    {msg.structured_contract.clarification_options.map((opt, oIdx) => (
                                      <button
                                        key={oIdx}
                                        type="button"
                                        className="btn btn-sm btn-outline-primary"
                                        style={{ textAlign: 'left', background: '#fff', fontSize: '0.82rem' }}
                                        onClick={() => handleClarificationOption(opt)}
                                        disabled={submitting}
                                      >
                                        <span>{opt}</span>
                                      </button>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.message_text}</div>
                            )}

                            {/* Actions on Latest Assistant Response */}
                            {index === messages.length - 1 && result && (
                              <div style={{
                                marginTop: '16px',
                                paddingTop: '12px',
                                borderTop: '1px solid var(--border)',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                flexWrap: 'wrap',
                                gap: '8px',
                              }}>
                                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                                  {result.can_resolve ? 'Observed issue resolved?' : 'Need technician dispatch?'}
                                </span>
                                <div style={{ display: 'flex', gap: '8px' }}>
                                  {result.can_resolve && (
                                    <button
                                      type="button"
                                      className="btn btn-sm btn-success"
                                      onClick={handleResolve}
                                      disabled={resolving || resolutionNotice}
                                    >
                                      <CheckCircle2 size={14} /> {resolving ? 'Recording...' : 'Report Resolved'}
                                    </button>
                                  )}
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-primary"
                                    style={{ background: 'linear-gradient(135deg, #e11d48 0%, #be123c 100%)' }}
                                    onClick={() => handleEscalate('Customer requested technician escalation')}
                                    disabled={escalating || escalationResult}
                                  >
                                    <Ticket size={14} /> {escalating ? 'Creating...' : 'Request Service'}
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                        <div style={{ fontSize: '0.68rem', color: isUser ? 'rgba(255,255,255,0.7)' : 'var(--text-muted)', textAlign: isUser ? 'right' : 'left', marginTop: '6px' }}>
                          {msg.created_at ? new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div ref={chatEndRef} />
              </div>

              {/* Message Input Bar */}
              <form onSubmit={handleQuery} style={{
                padding: '16px 20px',
                borderTop: '1px solid var(--border)',
                background: '#fff',
                borderRadius: '0 0 var(--radius-lg) var(--radius-lg)',
                display: 'flex',
                gap: '10px',
                alignItems: 'center',
              }}>
                <input
                  type="text"
                  id="userQuery"
                  className="form-control"
                  placeholder={
                    !selectedSite || !selectedAsset
                      ? "Select facility and equipment on the left to start..."
                      : messages.length === 0
                      ? "Describe the issue (e.g. 'Split AC is not cooling')..."
                      : "Reply or describe next observation (e.g. 'Airflow is weak', 'I checked that already')..."
                  }
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  maxLength={4000}
                  disabled={submitting || !selectedSite || !selectedAsset}
                  style={{ flex: 1 }}
                />
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={submitting || !query.trim() || !selectedSite || !selectedAsset}
                  style={{ minWidth: '100px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
                >
                  {submitting ? (
                    <>
                      <RefreshCw size={14} className="spin" />
                      <span>Thinking...</span>
                    </>
                  ) : (
                    <>
                      <Send size={14} />
                      <span>{messages.length === 0 ? 'Start' : 'Send'}</span>
                    </>
                  )}
                </button>
              </form>
            </div>

          </div>
        </div>
      )}

      {/* TAB 2: SUPPORT CONVERSATION HISTORY */}
      {activeTab === 'conversations' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <History size={18} color="var(--primary)" />
              <span>Diagnostic Conversation History</span>
            </div>
            <button
              type="button"
              className="btn btn-sm btn-outline-primary"
              onClick={loadConversations}
              disabled={loadingConversations}
            >
              <RefreshCw size={14} className={loadingConversations ? 'spin' : ''} /> Refresh
            </button>
          </div>

          {conversationsError && (
            <div className="alert alert-danger" style={{ marginBottom: '16px' }}>
              <AlertTriangle size={16} /> {conversationsError}
            </div>
          )}

          {loadingConversations && (
            <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
              Loading conversation transcripts...
            </div>
          )}

          {!loadingConversations && conversations.length === 0 && (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
              <p>No previous diagnostic conversations found for this account.</p>
            </div>
          )}

          {!loadingConversations && conversations.length > 0 && (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {conversations.map(c => (
                  <div
                    key={c.id || c.conversation_uuid}
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)',
                      padding: '16px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      background: '#fff',
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                        <span className={`badge ${c.status === 'RESOLVED' ? 'badge-success' : c.status === 'ESCALATED' ? 'badge-warning' : 'badge-primary'}`} style={{ fontSize: '0.72rem' }}>
                          {c.status}
                        </span>
                        <strong style={{ fontSize: '0.92rem', color: 'var(--text-main)' }}>
                          {c.title || 'Diagnostic Session'}
                        </strong>
                      </div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        Facility: <strong>{c.site?.name || c.site_name || 'Site'}</strong> | Equipment: <strong>{c.asset?.name || c.asset_name || 'Equipment'}</strong> | Started: {c.created_at ? new Date(c.created_at).toLocaleString() : 'N/A'}
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-primary"
                        onClick={() => viewConversationTranscript(c)}
                      >
                        <FileText size={14} /> View Transcript
                      </button>
                      {c.status === 'ACTIVE' && (
                        <button
                          type="button"
                          className="btn btn-sm btn-primary"
                          onClick={() => resumeConversationFromHistory(c)}
                        >
                          <ArrowRight size={14} /> Resume
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination Controls for Conversations */}
              {conversationsTotal > 10 && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    Showing {conversationsOffset + 1}–{Math.min(conversationsOffset + conversations.length, conversationsTotal)} of {conversationsTotal} conversations
                  </span>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      disabled={conversationsOffset <= 0 || loadingConversations}
                      onClick={() => loadConversations(Math.max(0, conversationsOffset - 10))}
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      disabled={conversationsOffset + 10 >= conversationsTotal || loadingConversations}
                      onClick={() => loadConversations(conversationsOffset + 10)}
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Modal / Panel for Transcript */}
          {activeTranscript && (
            <div style={{
              marginTop: '24px',
              padding: '20px',
              borderTop: '2px solid var(--border)',
              background: 'var(--bg-app)',
              borderRadius: 'var(--radius-md)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>
                  Transcript: {activeTranscript.title}
                </h4>
                <button
                  type="button"
                  className="btn btn-sm btn-link"
                  onClick={() => setActiveTranscript(null)}
                >
                  Close Transcript
                </button>
              </div>

              {transcriptError && (
                <div className="alert alert-danger" style={{ marginBottom: '12px' }}>
                  <AlertTriangle size={14} /> {transcriptError}
                </div>
              )}

              {loadingTranscript && <p style={{ fontSize: '0.85rem' }}>Loading messages...</p>}

              {!loadingTranscript && transcriptMessages.length === 0 && (
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No messages logged for this session.</p>
              )}

              {!loadingTranscript && transcriptMessages.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '16px' }}>
                  {transcriptMessages.map((m, idx) => (
                    <div
                      key={m.id || idx}
                      style={{
                        padding: '12px 14px',
                        borderRadius: 'var(--radius-md)',
                        background: m.sender_role === 'customer' ? '#e0e7ff' : '#ffffff',
                        border: '1px solid var(--border)',
                        maxWidth: '85%',
                        alignSelf: m.sender_role === 'customer' ? 'flex-end' : 'flex-start',
                      }}
                    >
                      <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '4px' }}>
                        {(m.sender_role || 'assistant').toUpperCase()} — {m.created_at ? new Date(m.created_at).toLocaleTimeString() : ''}
                      </div>
                      <div style={{ fontSize: '0.88rem', whiteSpace: 'pre-wrap' }}>
                        {m.sender_role === 'assistant' ? <CustomerAnswer presentation={m.structured_contract?.presentation} assetId={activeTranscript?.asset_id || m.structured_contract?.asset?.id} /> : (m.message_text || m.content || '')}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* In-transcript reply box if conversation is still active */}
              {activeTranscript.status === 'ACTIVE' && (
                <form onSubmit={handleSendReply} style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="Type follow-up observation or response..."
                    value={replyText}
                    onChange={(e) => setReplyText(e.target.value)}
                    disabled={sendingReply}
                    style={{ flex: 1 }}
                  />
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={sendingReply || !replyText.trim()}
                  >
                    <Send size={14} /> {sendingReply ? 'Sending...' : 'Send'}
                  </button>
                </form>
              )}
            </div>
          )}
        </div>
      )}

      {/* TAB 3: CUSTOMER SERVICE TICKETS */}
      {activeTab === 'tickets' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Ticket size={18} color="var(--primary)" />
              <span>Service Calls & Ticket Tracking</span>
            </div>
            <button
              type="button"
              className="btn btn-sm btn-outline-primary"
              onClick={() => loadTickets(ticketsOffset)}
              disabled={loadingTickets}
            >
              <RefreshCw size={14} className={loadingTickets ? 'spin' : ''} /> Refresh Status
            </button>
          </div>

          {ticketsError && (
            <div className="alert alert-danger" style={{ marginBottom: '16px' }}>
              <AlertTriangle size={16} /> {ticketsError}
            </div>
          )}

          {loadingTickets && (
            <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
              Loading service tickets...
            </div>
          )}

          {!loadingTickets && tickets.length === 0 && (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
              <p>No active or historical service tickets found.</p>
            </div>
          )}

          {!loadingTickets && tickets.length > 0 && (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {tickets.map(t => (
                  <div
                    key={t.id}
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)',
                      padding: '16px',
                      background: '#fff',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontWeight: 800, fontSize: '1rem', color: 'var(--primary)' }}>
                          #{t.servy_id}
                        </span>
                        <span className={`badge ${t.status === 'closed' ? 'badge-success' : t.status === 'assigned' ? 'badge-primary' : 'badge-warning'}`}>
                          {t.status?.toUpperCase()}
                        </span>
                        <span className="badge" style={{ background: '#f1f5f9', color: '#475569' }}>
                          Priority: {t.priority?.toUpperCase()}
                        </span>
                      </div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        Created: {t.created_at ? new Date(t.created_at).toLocaleString() : 'N/A'}
                      </span>
                    </div>

                    <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
                      {t.complaint_type || 'Service Request'}
                    </div>

                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                      Facility: <strong>{t.site?.name || t.site_name || 'Facility'}</strong> | Equipment: <strong>{t.asset?.name || t.asset_name || 'Equipment'} ({t.asset?.model_number || t.asset?.asset_code || t.asset_model || 'N/A'})</strong>
                    </div>

                    <div style={{
                      background: 'var(--bg-app)',
                      padding: '8px 12px',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.8rem',
                      color: 'var(--text-main)',
                    }}>
                      Assigned Technician: <strong>{t.technician_name || 'Pending Dispatch'}</strong>
                      {t.resolution_summary && (
                        <div style={{ marginTop: '4px', color: '#166534' }}>
                          Resolution: {t.resolution_summary}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination Controls for Tickets */}
              {ticketsTotal > 10 && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    Showing {ticketsOffset + 1}–{Math.min(ticketsOffset + tickets.length, ticketsTotal)} of {ticketsTotal} tickets
                  </span>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      disabled={ticketsOffset <= 0 || loadingTickets}
                      onClick={() => loadTickets(Math.max(0, ticketsOffset - 10))}
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      disabled={ticketsOffset + 10 >= ticketsTotal || loadingTickets}
                      onClick={() => loadTickets(ticketsOffset + 10)}
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
