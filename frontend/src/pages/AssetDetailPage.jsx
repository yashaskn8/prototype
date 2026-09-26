import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../AuthContext';
import { 
  ArrowLeft, Package, FileText, Bot, History, Ticket, ShieldCheck, 
  Upload, Download, AlertCircle, CheckCircle2, Clock, Calendar, 
  Send, RefreshCw, AlertTriangle, ChevronRight, Check, Activity, Shield, CheckSquare
} from 'lucide-react';

export function AssetDetailPage({ assetId, onBack, onNavigateTab }) {
  const { user } = useAuth();
  const isCustomer = user?.role === 'customer';

  const [asset, setAsset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Tab-specific states
  // Documents
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadTitle, setUploadTitle] = useState('');
  const [uploadDesc, setUploadDesc] = useState('');
  const [uploadDocType, setUploadDocType] = useState('reference');
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadContentText, setUploadContentText] = useState('');
  const [uploading, setUploading] = useState(false);
  const [docNotice, setDocNotice] = useState(null);

  // Ask AI
  const [query, setQuery] = useState('');
  const [submittingQuery, setSubmittingQuery] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [aiError, setAiError] = useState(null);
  const [resolving, setResolving] = useState(false);
  const [escalating, setEscalating] = useState(false);
  const [resolutionNotice, setResolutionNotice] = useState(null);
  const [escalationResult, setEscalationResult] = useState(null);

  // Calls (Service History & Open Tickets)
  const [openCalls, setOpenCalls] = useState([]);
  const [historyCalls, setHistoryCalls] = useState([]);
  const [loadingCalls, setLoadingCalls] = useState(false);

  // Servy Zero-Repeat: Diagnose & Recover State
  const [diagComplaint, setDiagComplaint] = useState('');
  const [diagSession, setDiagSession] = useState(null);
  const [diagSubmitting, setDiagSubmitting] = useState(false);
  const [diagNotice, setDiagNotice] = useState(null);
  const [diagAnswerText, setDiagAnswerText] = useState('');
  const [diagClarifyText, setDiagClarifyText] = useState('');

  // --- Session persistence helpers (survives page refresh) ---
  const SESSION_STORAGE_KEY = `servy-diagnostic-session:${assetId}`;

  function persistSessionId(sessionId) {
    try { sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId); } catch (_) { /* noop */ }
  }
  function clearPersistedSession() {
    try { sessionStorage.removeItem(SESSION_STORAGE_KEY); } catch (_) { /* noop */ }
  }
  function getPersistedSessionId() {
    try { return sessionStorage.getItem(SESSION_STORAGE_KEY) || null; } catch (_) { return null; }
  }

  // --- Status label helper ---
  function getStatusLabel(status) {
    switch (status) {
      case 'RESOLVED': return 'Resolved';
      case 'ESCALATED': return 'Escalated to Service';
      case 'NO_PLAYBOOK_AVAILABLE': return 'Needs Clarification';
      case 'ABANDONED': return 'Session Closed';
      default: return 'Session Active';
    }
  }
  function getStatusColor(status) {
    switch (status) {
      case 'RESOLVED': return { bg: '#dcfce7', color: '#15803d' };
      case 'ESCALATED': return { bg: '#fef3c7', color: '#b45309' };
      case 'NO_PLAYBOOK_AVAILABLE': return { bg: '#fef9c3', color: '#854d0e' };
      case 'ABANDONED': return { bg: '#f1f5f9', color: '#64748b' };
      default: return { bg: '#eff6ff', color: 'var(--primary)' };
    }
  }

  // --- Terminal-state detection for error recovery ---
  function isTerminalError(err) {
    const d = (err?.detail || '').toLowerCase();
    return d.includes('terminal') || d.includes('escalated') || d.includes('resolved')
      || d.includes('session modified') || d.includes('version') || d.includes('conflict')
      || d.includes('immutable') || err?.status === 409;
  }

  async function recoverAuthoritativeSession(sessionId) {
    try {
      const latest = await api.get(`/api/diagnostics/${sessionId}/`);
      setDiagSession(latest);
      persistSessionId(latest.session_id);
      if (latest.status === 'ESCALATED') {
        loadCalls();
      }
      return latest;
    } catch (fetchErr) {
      console.error('Failed to recover authoritative session', fetchErr);
      clearPersistedSession();
      setDiagSession(null);
      return null;
    }
  }

  function makeIdempotencyKey(prefix) {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  async function handleStartDiagnostics(e) {
    if (e) e.preventDefault();
    if (!diagComplaint.trim()) return;
    setDiagSubmitting(true);
    setDiagNotice(null);
    try {
      const res = await api.post(`/api/assets/${assetId}/diagnostics/`, {
        complaint: diagComplaint.trim(),
        idempotency_key: makeIdempotencyKey('diag-start'),
      });
      setDiagSession(res);
      persistSessionId(res.session_id);
      setDiagNotice({ type: 'info', text: 'Diagnostic session initiated with verified asset locking.' });
    } catch (err) {
      setDiagNotice({ type: 'error', text: err.detail || 'Failed to start diagnostic recovery session.' });
    } finally {
      setDiagSubmitting(false);
    }
  }

  async function handleSubmitAnswer(nodeId, val) {
    if (!diagSession) return;
    setDiagSubmitting(true);
    setDiagNotice(null);
    try {
      const res = await api.post(`/api/diagnostics/${diagSession.session_id}/answers/`, {
        node_id: nodeId,
        value: val,
        expected_version: diagSession.version,
        idempotency_key: makeIdempotencyKey('diag-ans'),
      });
      setDiagSession(res);
      persistSessionId(res.session_id);
      setDiagAnswerText('');
      // If auto-escalation occurred, refresh tickets
      if (res.status === 'ESCALATED') {
        setDiagNotice({ type: 'info', text: 'Service Call created with deterministic Recovery Passport.' });
        loadCalls();
      }
    } catch (err) {
      if (isTerminalError(err)) {
        setDiagNotice({ type: 'info', text: 'Your diagnostic session has changed. Refreshing latest state...' });
        await recoverAuthoritativeSession(diagSession.session_id);
      } else {
        setDiagNotice({ type: 'error', text: err.detail || 'Failed to submit observation.' });
      }
    } finally {
      setDiagSubmitting(false);
    }
  }

  async function handleCompleteAction(nodeId) {
    if (!diagSession) return;
    setDiagSubmitting(true);
    setDiagNotice(null);
    try {
      const res = await api.post(`/api/diagnostics/${diagSession.session_id}/actions/${nodeId}/complete/`, {
        expected_version: diagSession.version,
        idempotency_key: makeIdempotencyKey('diag-act'),
      });
      setDiagSession(res);
      persistSessionId(res.session_id);
      if (res.status === 'ESCALATED') {
        setDiagNotice({ type: 'info', text: 'Service Call created with deterministic Recovery Passport.' });
        loadCalls();
      }
    } catch (err) {
      if (isTerminalError(err)) {
        setDiagNotice({ type: 'info', text: 'Your diagnostic session has changed. Refreshing latest state...' });
        await recoverAuthoritativeSession(diagSession.session_id);
      } else {
        setDiagNotice({ type: 'error', text: err.detail || 'Failed to confirm action.' });
      }
    } finally {
      setDiagSubmitting(false);
    }
  }

  async function handleResolveSession() {
    if (!diagSession) return;
    setDiagSubmitting(true);
    setDiagNotice(null);
    try {
      const res = await api.post(`/api/diagnostics/${diagSession.session_id}/resolve/`, {
        expected_version: diagSession.version,
        summary: 'Customer verified complete resolution after guided procedure.',
      });
      setDiagSession(res);
      persistSessionId(res.session_id);
      setDiagNotice({ type: 'success', text: 'Diagnostic session marked resolved after customer verification.' });
    } catch (err) {
      if (isTerminalError(err)) {
        setDiagNotice({ type: 'info', text: 'Session already finalized. Refreshing...' });
        await recoverAuthoritativeSession(diagSession.session_id);
      } else {
        setDiagNotice({ type: 'error', text: err.detail || 'Failed to resolve session.' });
      }
    } finally {
      setDiagSubmitting(false);
    }
  }

  async function handleEscalateSession(customReason) {
    if (!diagSession) return;
    setDiagSubmitting(true);
    setDiagNotice(null);
    try {
      const res = await api.post(`/api/diagnostics/${diagSession.session_id}/escalate/`, {
        reason: customReason || 'Escalated by customer during diagnostic procedure.',
        expected_version: diagSession.version,
        idempotency_key: makeIdempotencyKey('diag-esc'),
      });
      setDiagSession(res);
      persistSessionId(res.session_id);
      setDiagNotice({ type: 'info', text: 'Service Call created with deterministic Recovery Passport.' });
      loadCalls();
    } catch (err) {
      if (isTerminalError(err)) {
        setDiagNotice({ type: 'info', text: 'Session already escalated. Refreshing...' });
        await recoverAuthoritativeSession(diagSession.session_id);
      } else {
        setDiagNotice({ type: 'error', text: err.detail || 'Failed to escalate session.' });
      }
    } finally {
      setDiagSubmitting(false);
    }
  }

  async function refreshSession(sessionId) {
    return recoverAuthoritativeSession(sessionId);
  }

  async function loadAssetDetail() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(`/api/assets/${assetId}/`);
      setAsset(res.asset);
    } catch (err) {
      setError(err.detail || 'Failed to load asset details.');
    } finally {
      setLoading(false);
    }
  }

  async function loadDocuments() {
    setLoadingDocs(true);
    try {
      const res = await api.get(`/api/assets/${assetId}/documents/`);
      setDocuments(res.documents || []);
    } catch (err) {
      console.error('Failed to load asset documents', err);
    } finally {
      setLoadingDocs(false);
    }
  }

  async function loadCalls() {
    setLoadingCalls(true);
    try {
      const [resOpen, resHistory] = await Promise.all([
        api.get(`/api/assets/${assetId}/calls/?scope=open`),
        api.get(`/api/assets/${assetId}/calls/?scope=history`)
      ]);
      setOpenCalls(resOpen.calls || []);
      setHistoryCalls(resHistory.calls || []);
    } catch (err) {
      console.error('Failed to load asset calls', err);
    } finally {
      setLoadingCalls(false);
    }
  }

  useEffect(() => {
    loadAssetDetail();
  }, [assetId]);

  // Restore persisted diagnostic session on mount/assetId change
  useEffect(() => {
    const savedId = getPersistedSessionId();
    if (savedId && !diagSession) {
      recoverAuthoritativeSession(savedId);
    }
  }, [assetId]);

  useEffect(() => {
    if (activeTab === 'documents') {
      loadDocuments();
    } else if (activeTab === 'history' || activeTab === 'tickets') {
      loadCalls();
    }
  }, [activeTab]);

  async function handleUploadDoc(e) {
    e.preventDefault();
    if (!uploadTitle.trim()) return;
    setUploading(true);
    setDocNotice(null);

    const formData = new FormData();
    formData.append('title', uploadTitle.trim());
    formData.append('description', uploadDesc.trim());
    formData.append('doc_type', uploadDocType);
    if (uploadFile) {
      formData.append('file', uploadFile);
    }
    if (uploadContentText.trim()) {
      formData.append('content_text', uploadContentText.trim());
    }

    try {
      const res = await api.upload(`/api/assets/${assetId}/documents/upload/`, formData);
      setDocNotice(res.message || 'Document uploaded successfully.');
      setUploadModalOpen(false);
      setUploadTitle('');
      setUploadDesc('');
      setUploadFile(null);
      setUploadContentText('');
      loadDocuments();
    } catch (err) {
      setDocNotice(err.detail || 'Upload failed. Please check file format and size.');
    } finally {
      setUploading(false);
    }
  }

  async function handleApproveForRag(docId) {
    try {
      const res = await api.post(`/api/knowledge/${docId}/approve-for-rag/`);
      setDocNotice(res.message || 'Approved for AI RAG.');
      loadDocuments();
    } catch (err) {
      setDocNotice(err.detail || 'Approval failed.');
    }
  }

  async function handleRemoveFromRag(docId) {
    try {
      const res = await api.post(`/api/knowledge/${docId}/remove-from-rag/`);
      setDocNotice(res.message || 'Removed from AI RAG.');
      loadDocuments();
    } catch (err) {
      setDocNotice(err.detail || 'Removal failed.');
    }
  }

  async function handleAiQuery(e) {
    e.preventDefault();
    if (!query.trim() || !asset) return;

    setSubmittingQuery(true);
    setAiError(null);
    setResolutionNotice(null);
    setEscalationResult(null);

    try {
      const res = await api.post('/api/customer-support/query/', {
        site_id: asset.site?.id,
        asset_id: asset.id,
        question: query.trim(),
        customer_id: asset.customer?.id,
      });
      setAiResult(res);
    } catch (err) {
      setAiError(err.detail || 'Error communicating with AI assistant.');
    } finally {
      setSubmittingQuery(false);
    }
  }

  async function handleResolveInteraction() {
    if (!aiResult?.interaction_id) return;
    setResolving(true);
    try {
      const res = await api.post(`/api/customer-support/interactions/${aiResult.interaction_id}/resolve/`);
      setResolutionNotice(res.detail || 'Diagnostic marked as resolved. Thank you!');
    } catch (err) {
      setAiError(err.detail || 'Could not update interaction.');
    } finally {
      setResolving(false);
    }
  }

  async function handleEscalateInteraction() {
    if (!aiResult?.interaction_id) return;
    setEscalating(true);
    try {
      const res = await api.post(`/api/customer-support/interactions/${aiResult.interaction_id}/escalate/`);
      setEscalationResult(res);
      loadCalls();
    } catch (err) {
      setAiError(err.detail || 'Could not escalate service ticket.');
    } finally {
      setEscalating(false);
    }
  }

  if (loading) {
    return (
      <div className="content-body" style={{ textAlign: 'center', padding: '60px 0' }}>
        <RefreshCw size={24} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '12px' }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading asset intelligence hub...</p>
      </div>
    );
  }

  if (error || !asset) {
    return (
      <div className="content-body">
        <button className="btn btn-secondary" onClick={onBack} style={{ marginBottom: '16px' }}>
          <ArrowLeft size={16} /> Back to Assets
        </button>
        <div className="alert alert-danger">
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <div>{error || 'Asset not found.'}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="content-body">
      {/* Top Header / Breadcrumb */}
      <div style={{ marginBottom: '20px' }}>
        <button 
          className="btn btn-secondary" 
          onClick={onBack} 
          style={{ marginBottom: '14px', display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem' }}
        >
          <ArrowLeft size={14} /> Back to Equipment Registry
        </button>

        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em', margin: 0 }}>
                {asset.name}
              </h2>
              <span className="badge" style={{ background: '#f1f5f9', color: '#1e293b', fontFamily: 'var(--font-mono)' }}>
                {asset.asset_code}
              </span>
              <span className={`badge ${asset.status === 'active' ? 'badge-resolved' : 'badge-open'}`}>
                {asset.status.toUpperCase()}
              </span>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginTop: '6px' }}>
              {asset.product?.name || 'Not specified'} 
              {asset.model_number && ` • Model ${asset.model_number}`}
              {asset.serial_number && ` • S/N: ${asset.serial_number}`}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            {asset.warranty_until && (
              <span style={{ 
                display: 'inline-flex', 
                alignItems: 'center', 
                gap: '6px', 
                fontSize: '0.82rem', 
                padding: '6px 14px', 
                borderRadius: '999px',
                fontWeight: 600,
                background: asset.is_warranty_expired ? '#fef2f2' : '#ecfdf5',
                color: asset.is_warranty_expired ? '#b91c1c' : '#059669',
                border: `1px solid ${asset.is_warranty_expired ? '#fecaca' : '#a7f3d0'}`
              }}>
                <ShieldCheck size={14} />
                {asset.is_warranty_expired 
                  ? 'Warranty Expired' 
                  : `Warranty Active (${asset.warranty_days_remaining} days left)`}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Workspace Tabs Navigation */}
      <div style={{ borderBottom: '1px solid var(--border)', marginBottom: '24px', display: 'flex', gap: '8px', overflowX: 'auto' }}>
        {[
          { id: 'overview', label: 'Overview', icon: Package },
          { id: 'diagnostics', label: 'Diagnose & Recover', icon: Activity },
          { id: 'documents', label: `Documents (${asset.documents_count})`, icon: FileText },
          { id: 'ask-ai', label: 'Quick AI Q&A', icon: Bot },
          { id: 'tickets', label: `Open Tickets (${asset.open_calls_count})`, icon: Ticket },
          { id: 'history', label: 'Service History', icon: History },
          { id: 'warranty', label: 'Warranty & Asset Details', icon: ShieldCheck },
        ].map(t => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 16px',
                border: 'none',
                background: 'transparent',
                fontWeight: isActive ? 700 : 500,
                color: isActive ? 'var(--primary)' : 'var(--text-muted)',
                borderBottom: isActive ? '2px solid var(--primary)' : '2px solid transparent',
                cursor: 'pointer',
                fontSize: '0.88rem',
                whiteSpace: 'nowrap'
              }}
            >
              <Icon size={16} />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: OVERVIEW */}
      {activeTab === 'overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginBottom: '24px' }}>
            <div className="card">
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700, marginBottom: '6px' }}>
                Operational Location
              </div>
              <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-main)', marginBottom: '4px' }}>
                {asset.site?.name || 'Central Facility'}
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                {asset.site?.address || 'Site Address Unspecified'}
                {asset.site?.city && `, ${asset.site.city}`}
              </div>
              <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid var(--border)', fontSize: '0.82rem' }}>
                Customer: <strong>{asset.customer?.name}</strong>
              </div>
            </div>

            <div className="card">
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700, marginBottom: '6px' }}>
                Model & Product Hierarchy
              </div>
              <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-main)', marginBottom: '4px' }}>
                {asset.product?.name || 'Generic Asset'}
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Brand: {asset.product?.brand_name || 'Standard'} • Category: {asset.product?.category_name || 'General'}
              </div>
              {asset.product?.domain_name && (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  Domain: {asset.product.domain_name}
                </div>
              )}
            </div>

            <div className="card">
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700, marginBottom: '6px' }}>
                Active Service Health
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '4px' }}>
                <span style={{ fontSize: '1.6rem', fontWeight: 800, color: asset.open_calls_count > 0 ? '#d97706' : '#059669' }}>
                  {asset.open_calls_count}
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Active Tickets</span>
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                {asset.open_calls_count === 0 
                  ? 'All operational systems healthy with zero pending dispatches.' 
                  : 'Requires engineer attention or customer diagnostic follow-up.'}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title" style={{ marginBottom: '12px' }}>Quick Actions</div>
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
              <button className="btn btn-primary" onClick={() => setActiveTab('diagnostics')}>
                <Activity size={16} /> Diagnose & Recover (Zero-Repeat)
              </button>
              <button className="btn btn-secondary" onClick={() => setActiveTab('documents')}>
                <FileText size={16} /> View Technical Documents ({asset.documents_count})
              </button>
              <button className="btn btn-secondary" onClick={() => setActiveTab('ask-ai')}>
                <Bot size={16} /> Quick AI Q&A
              </button>
              <button className="btn btn-secondary" onClick={() => setActiveTab('tickets')}>
                <Ticket size={16} /> View Open Tickets ({asset.open_calls_count})
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TAB: DIAGNOSE & RECOVER (SERVY ZERO-REPEAT) */}
      {activeTab === 'diagnostics' && (
        <div>
          {diagNotice && (
            <div className={`alert alert-${diagNotice.type === 'error' ? 'danger' : diagNotice.type === 'success' ? 'success' : 'info'}`} style={{ marginBottom: '16px' }}>
              {diagNotice.type === 'error' ? <AlertCircle size={18} style={{ flexShrink: 0 }} /> : <CheckCircle2 size={18} style={{ flexShrink: 0 }} />}
              <div>{diagNotice.text}</div>
            </div>
          )}

          {!diagSession ? (
            <div className="card" style={{ maxWidth: '800px', margin: '0 auto', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <div style={{ padding: '12px', borderRadius: '12px', background: '#eff6ff', color: 'var(--primary)' }}>
                  <Activity size={28} />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-main)' }}>
                    Diagnose & Recover (Zero-Repeat)
                  </h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    Tell Servy once. Either Servy safely helps recover your machine, or the engineer receives everything already checked so you never restart from zero.
                  </p>
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px', border: '1px solid var(--border)', marginBottom: '20px' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>
                  Target Machine Locked
                </div>
                <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-main)' }}>
                  {asset.name} • {asset.model_number || asset.asset_code}
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Product: {asset.product?.name || 'General Equipment'}
                </div>
              </div>

              <form onSubmit={handleStartDiagnostics}>
                <div className="form-group" style={{ marginBottom: '20px' }}>
                  <label className="form-label" style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                    Describe what is malfunctioning or abnormal:
                  </label>
                  <textarea
                    className="form-control"
                    rows={4}
                    placeholder="e.g. My readings are unstable after cleaning cycle. Thermostat temperature fluctuating."
                    value={diagComplaint}
                    onChange={e => setDiagComplaint(e.target.value)}
                    required
                    style={{ fontSize: '0.9rem', lineHeight: '1.5' }}
                  />
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                    Servy will bind this diagnostic incident to {asset.name} and guide you step-by-step using approved manufacturer procedures.
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={diagSubmitting || !diagComplaint.trim()}
                    style={{ padding: '10px 24px', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: '8px' }}
                  >
                    {diagSubmitting ? <RefreshCw size={16} className="spin" /> : <Activity size={16} />}
                    {diagSubmitting ? 'Initializing Session...' : 'Start Guided Recovery'}
                  </button>
                </div>
              </form>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '20px' }}>
              {/* Header Status Bar */}
              <div className="card" style={{ padding: '18px 24px', borderLeft: '4px solid var(--primary)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '14px' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <span className="badge" style={{ background: getStatusColor(diagSession.status).bg, color: getStatusColor(diagSession.status).color, fontWeight: 700 }}>
                        {getStatusLabel(diagSession.status)}
                      </span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        Target: <strong>{asset.name}</strong> ({asset.asset_code})
                      </span>
                    </div>
                    <div style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-main)' }}>
                      "{diagSession.complaint}"
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                      Evidence Completeness
                    </div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--primary)' }}>
                      {Math.round((diagSession.evidence_completeness || 0) * 100)}%
                    </div>
                  </div>
                </div>

                {/* Progress bar */}
                <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                  <div 
                    style={{ 
                      width: `${Math.round((diagSession.evidence_completeness || 0) * 100)}%`, 
                      height: '100%', 
                      background: 'var(--primary)', 
                      transition: 'width 0.4s ease' 
                    }} 
                  />
                </div>
              </div>

              {/* Main Interactive Stage */}
              {diagSession.status === 'RESOLVED' ? (
                <div className="card" style={{ textAlign: 'center', padding: '40px 24px', background: '#f0fdf4', border: '1px solid #bbf7d0' }}>
                  <div style={{ display: 'inline-flex', padding: '16px', borderRadius: '50%', background: '#dcfce7', color: '#16a34a', marginBottom: '16px' }}>
                    <CheckCircle2 size={40} />
                  </div>
                  <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: '#15803d', marginBottom: '8px' }}>
                    Diagnostic Session Resolved
                  </h3>
                  <p style={{ maxWidth: '520px', margin: '0 auto 24px auto', color: '#166534', fontSize: '0.9rem' }}>
                    Diagnostic session marked resolved after customer verification. All diagnostic observations and confirmations have been recorded.
                  </p>
                  <button 
                    className="btn btn-secondary"
                    onClick={() => { clearPersistedSession(); setDiagSession(null); setDiagComplaint(''); }}
                  >
                    Start New Recovery Session
                  </button>
                </div>
              ) : diagSession.status === 'ESCALATED' ? (
                <div className="card" style={{ border: '1px solid #cbd5e1' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '16px', background: '#f8fafc', borderRadius: '8px', borderBottom: '1px solid var(--border)', marginBottom: '20px' }}>
                    <Ticket size={24} color="#d97706" />
                    <div>
                      <h4 style={{ fontWeight: 800, fontSize: '1.05rem', color: 'var(--text-main)' }}>
                        Service Call Dispatched: #{diagSession.escalated_call?.servy_id || 'SER-ESCALATED'}
                      </h4>
                      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                        Field engineering has been notified. The deterministic Recovery Passport has been attached so the technician will NOT repeat what you already verified.
                      </p>
                    </div>
                  </div>

                  {diagSession.recovery_passport && (
                    <div style={{ padding: '0 8px' }}>
                      <h5 style={{ fontWeight: 700, fontSize: '0.95rem', color: '#b45309', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <ShieldCheck size={16} /> DO NOT REPEAT WITH CUSTOMER (Verified Checks)
                      </h5>
                      <div style={{ background: '#fffbeb', border: '1px solid #fef3c7', borderRadius: '8px', padding: '14px', marginBottom: '18px' }}>
                        {diagSession.recovery_passport.do_not_repeat_items?.length > 0 ? (
                          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.85rem', color: '#92400e' }}>
                            {diagSession.recovery_passport.do_not_repeat_items.map((item, idx) => (
                              <li key={idx} style={{ marginBottom: '4px' }}>{item}</li>
                            ))}
                          </ul>
                        ) : (
                          <div style={{ fontSize: '0.85rem', color: '#92400e' }}>No repetitive checks to exclude.</div>
                        )}
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <button className="btn btn-secondary" onClick={() => setActiveTab('tickets')}>
                          <Ticket size={14} /> View in Open Tickets
                        </button>
                        <button className="btn btn-secondary" onClick={() => { clearPersistedSession(); setDiagSession(null); setDiagComplaint(''); }}>
                          Close Diagnostic View
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : diagSession.current_node ? (
                <div className="card" style={{ padding: '24px', border: '2px solid var(--border)' }}>
                  {/* Node Type Indicator */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                    <span className="badge" style={{ 
                      background: diagSession.current_node.node_type === 'SAFE_ACTION' ? '#ecfdf5' : '#f1f5f9',
                      color: diagSession.current_node.node_type === 'SAFE_ACTION' ? '#059669' : '#1e293b',
                      fontWeight: 700,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      {diagSession.current_node.node_type === 'SAFE_ACTION' && <Shield size={12} />}
                      {diagSession.current_node.node_type === 'OBSERVE' && 'Step 1: Observation'}
                      {diagSession.current_node.node_type === 'SAFE_ACTION' && 'Step 2: Authorized Operator Procedure'}
                      {diagSession.current_node.node_type === 'VERIFY' && 'Step 3: Outcome Verification'}
                      {diagSession.current_node.node_type === 'ESCALATE' && 'Escalation Notice'}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Version: {diagSession.version}
                    </span>
                  </div>

                  {/* OBSERVE Node */}
                  {diagSession.current_node.node_type === 'OBSERVE' && (
                    <div>
                      <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '16px', color: 'var(--text-main)' }}>
                        {diagSession.current_node.question}
                      </h3>

                      {diagSession.current_node.response_schema === 'BOOLEAN' ? (
                        <div style={{ display: 'flex', gap: '16px', marginTop: '20px' }}>
                          <button
                            className="btn btn-primary"
                            style={{ flex: 1, padding: '14px', fontSize: '1rem', fontWeight: 700 }}
                            disabled={diagSubmitting}
                            onClick={() => handleSubmitAnswer(diagSession.current_node.node_id, true)}
                          >
                            Yes
                          </button>
                          <button
                            className="btn btn-secondary"
                            style={{ flex: 1, padding: '14px', fontSize: '1rem', fontWeight: 700 }}
                            disabled={diagSubmitting}
                            onClick={() => handleSubmitAnswer(diagSession.current_node.node_id, false)}
                          >
                            No
                          </button>
                        </div>
                      ) : (
                        <div>
                          <input
                            type="text"
                            className="form-control"
                            placeholder="Enter observed reading or code..."
                            value={diagAnswerText}
                            onChange={e => setDiagAnswerText(e.target.value)}
                            style={{ marginBottom: '14px' }}
                          />
                          <button
                            className="btn btn-primary"
                            disabled={diagSubmitting || !diagAnswerText.trim()}
                            onClick={() => handleSubmitAnswer(diagSession.current_node.node_id, diagAnswerText.trim())}
                          >
                            Submit Observation
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* SAFE_ACTION Node */}
                  {diagSession.current_node.node_type === 'SAFE_ACTION' && (
                    <div>
                      <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px', padding: '18px', marginBottom: '20px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#15803d', fontWeight: 700, fontSize: '0.85rem', marginBottom: '8px' }}>
                          <ShieldCheck size={18} /> Canonical Approved Procedure (Green Safety Class)
                        </div>
                        <div style={{ fontSize: '1.02rem', fontWeight: 600, color: '#166534', lineHeight: 1.6, marginBottom: '14px' }}>
                          {diagSession.current_node.instruction}
                        </div>

                        {diagSession.current_node.source_citation && (
                          <div style={{ fontSize: '0.78rem', color: '#15803d', borderTop: '1px solid #bbf7d0', paddingTop: '8px' }}>
                            Source: <strong>{diagSession.current_node.source_citation.document_title}</strong> (Section: {diagSession.current_node.source_citation.heading}, v{diagSession.current_node.source_citation.version})
                          </div>
                        )}
                      </div>

                      <button
                        className="btn btn-primary"
                        style={{ width: '100%', padding: '14px', fontSize: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                        disabled={diagSubmitting}
                        onClick={() => handleCompleteAction(diagSession.current_node.node_id)}
                      >
                        <CheckSquare size={18} />
                        I Have Completed This Action
                      </button>
                    </div>
                  )}

                  {/* VERIFY Node */}
                  {diagSession.current_node.node_type === 'VERIFY' && (
                    <div>
                      <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '16px', color: 'var(--text-main)' }}>
                        {diagSession.current_node.question}
                      </h3>
                      <div style={{ display: 'flex', gap: '16px', marginTop: '20px' }}>
                        <button
                          className="btn btn-primary"
                          style={{ flex: 1, padding: '14px', fontSize: '1rem', fontWeight: 700 }}
                          disabled={diagSubmitting}
                          onClick={() => handleSubmitAnswer(diagSession.current_node.node_id, true)}
                        >
                          Yes, Issue Resolved
                        </button>
                        <button
                          className="btn btn-secondary"
                          style={{ flex: 1, padding: '14px', fontSize: '1rem', fontWeight: 700 }}
                          disabled={diagSubmitting}
                          onClick={() => handleSubmitAnswer(diagSession.current_node.node_id, false)}
                        >
                          No, Issue Persists
                        </button>
                      </div>
                    </div>
                  )}

                  {/* ESCALATE Node */}
                  {diagSession.current_node.node_type === 'ESCALATE' && (
                    <div style={{ textAlign: 'center', padding: '20px 0' }}>
                      <AlertTriangle size={36} color="#d97706" style={{ marginBottom: '12px' }} />
                      <h3 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '8px' }}>
                        Specialized Engineering Service Required
                      </h3>
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '20px' }}>
                        {diagSession.current_node.reason}
                      </p>
                      <button
                        className="btn btn-primary"
                        style={{ padding: '12px 28px', fontWeight: 700 }}
                        disabled={diagSubmitting}
                        onClick={() => handleEscalateSession(diagSession.current_node.reason)}
                      >
                        Dispatch Engineering Call
                      </button>
                    </div>
                  )}

                  {/* Bottom manual escalation link */}
                  {diagSession.current_node.node_type !== 'ESCALATE' && (
                    <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        Need immediate technician dispatch?
                      </span>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '6px 12px', fontSize: '0.78rem' }}
                        disabled={diagSubmitting}
                        onClick={() => handleEscalateSession('Customer requested early technician escalation.')}
                      >
                        Escalate to Service
                      </button>
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: DOCUMENTS */}
      {activeTab === 'documents' && (
        <div>
          {docNotice && (
            <div className="alert alert-info" style={{ marginBottom: '16px' }}>
              <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
              <div>{docNotice}</div>
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Asset Knowledge & Technical Documentation</h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Hierarchy-aware repository including machine-specific reports and inherited product manuals.
              </p>
            </div>
            <button className="btn btn-primary" onClick={() => setUploadModalOpen(true)}>
              <Upload size={16} /> Upload Document
            </button>
          </div>

          {/* Upload Modal */}
          {uploadModalOpen && (
            <div className="card" style={{ marginBottom: '24px', border: '2px solid var(--primary)' }}>
              <div className="card-title" style={{ marginBottom: '14px' }}>
                <Upload size={18} color="var(--primary)" /> Upload Asset Document
              </div>
              <form onSubmit={handleUploadDoc}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '14px' }}>
                  <div className="form-group">
                    <label className="form-label">Document Title *</label>
                    <input 
                      type="text" 
                      className="form-control" 
                      placeholder="e.g. Site Calibration Certificate 2026" 
                      value={uploadTitle}
                      onChange={e => setUploadTitle(e.target.value)}
                      required 
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Document Classification</label>
                    <select 
                      className="form-control"
                      value={uploadDocType}
                      onChange={e => setUploadDocType(e.target.value)}
                    >
                      <option value="reference">Reference / Report</option>
                      <option value="maintenance">Maintenance Guide</option>
                      <option value="troubleshooting">Troubleshooting</option>
                      <option value="cleaning">Cleaning Procedure</option>
                      <option value="user_guide">User Guide</option>
                      <option value="installation">Installation Guide</option>
                    </select>
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Description / Summary</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    placeholder="Brief description of the document" 
                    value={uploadDesc}
                    onChange={e => setUploadDesc(e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Attach File (PDF, DOCX, TXT, MD, CSV - Max 10MB)</label>
                  <input 
                    type="file" 
                    className="form-control"
                    onChange={e => setUploadFile(e.target.files[0] || null)}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Or Direct Text Content</label>
                  <textarea 
                    className="form-control" 
                    rows={3} 
                    placeholder="Paste technical procedures or diagnostic notes..."
                    value={uploadContentText}
                    onChange={e => setUploadContentText(e.target.value)}
                  />
                </div>

                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
                  {isCustomer ? (
                    <span><strong>Trust Boundary Notice:</strong> Customer uploads are saved as records of truth and require staff approval before being ingested into AI RAG.</span>
                  ) : (
                    <span><strong>Staff Notice:</strong> Document will be indexed immediately and synced to ChromaDB.</span>
                  )}
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                  <button type="button" className="btn btn-secondary" onClick={() => setUploadModalOpen(false)}>
                    Cancel
                  </button>
                  <button type="submit" className="btn btn-primary" disabled={uploading}>
                    {uploading ? 'Processing...' : 'Upload & Save'}
                  </button>
                </div>
              </form>
            </div>
          )}

          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div className="table-responsive">
              <table className="table">
                <thead>
                  <tr>
                    <th>Document Title</th>
                    <th>Scope Level</th>
                    <th>Classification</th>
                    <th>AI RAG Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {loadingDocs ? (
                    <tr>
                      <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                        <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                        <p>Loading technical documentation...</p>
                      </td>
                    </tr>
                  ) : documents.length === 0 ? (
                    <tr>
                      <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                        No documentation found for this asset.
                      </td>
                    </tr>
                  ) : (
                    documents.map(d => (
                      <tr key={d.id}>
                        <td>
                          <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <FileText size={16} color="var(--primary)" />
                            {d.title}
                          </div>
                          {d.description && (
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                              {d.description}
                            </div>
                          )}
                        </td>
                        <td>
                          <span className="badge" style={{ 
                            background: d.scope_level === 'asset' ? '#e0e7ff' : '#f1f5f9',
                            color: d.scope_level === 'asset' ? '#4338ca' : '#1e293b',
                            textTransform: 'capitalize'
                          }}>
                            {d.scope_level === 'asset' ? 'Machine Specific' : `${d.scope_level} Manual`}
                          </span>
                        </td>
                        <td>
                          <span className="badge" style={{ background: '#f8fafc', border: '1px solid var(--border)' }}>
                            {d.doc_type_display || d.doc_type}
                          </span>
                        </td>
                        <td>
                          {d.is_rag_enabled ? (
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', color: '#059669', fontWeight: 600 }}>
                              <Check size={14} /> Active in RAG ({d.index_status})
                            </span>
                          ) : (
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                              Pending Approval
                            </span>
                          )}
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                            {d.has_file && (
                              <a 
                                href={`/api/knowledge/${d.id}/download/`} 
                                className="btn btn-secondary" 
                                style={{ padding: '4px 8px', fontSize: '0.75rem' }} 
                                download
                              >
                                <Download size={14} /> Download
                              </a>
                            )}
                            {!isCustomer && (
                              d.is_rag_enabled ? (
                                <button 
                                  className="btn btn-secondary"
                                  style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                                  onClick={() => handleRemoveFromRag(d.id)}
                                >
                                  Disable RAG
                                </button>
                              ) : (
                                <button 
                                  className="btn btn-primary"
                                  style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                                  onClick={() => handleApproveForRag(d.id)}
                                >
                                  Approve RAG
                                </button>
                              )
                            )}
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: ASK AI (AUTOMATICALLY SCOPED) */}
      {activeTab === 'ask-ai' && (
        <div>
          {aiError && (
            <div className="alert alert-danger" style={{ marginBottom: '16px' }}>
              <AlertTriangle size={18} style={{ flexShrink: 0 }} />
              <div>{aiError}</div>
            </div>
          )}

          {resolutionNotice && (
            <div className="alert alert-success" style={{ marginBottom: '16px' }}>
              <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
              <div>{resolutionNotice}</div>
            </div>
          )}

          {escalationResult && (
            <div className="alert alert-info" style={{ marginBottom: '16px', display: 'block' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, marginBottom: '4px' }}>
                <Ticket size={18} /> Ticket Dispatched: #{escalationResult.servy_id}
              </div>
              <p style={{ fontSize: '0.85rem' }}>
                Service call dispatched to engineering team with query context carry-forward. Track progress in the Open Tickets tab.
              </p>
            </div>
          )}

          <div className="rag-box">
            <div className="rag-header">
              <div className="rag-header-left">
                <div className="rag-icon-badge">
                  <Bot size={20} />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>AI Troubleshooting: {asset.name}</h3>
                  <span style={{ fontSize: '0.75rem', opacity: 0.85 }}>
                    Pre-grounded on {asset.product?.name || 'Asset Manuals'}
                  </span>
                </div>
              </div>
              <span className="badge" style={{ background: '#ecfdf5', color: '#059669', fontWeight: 700 }}>
                Target Machine Locked
              </span>
            </div>

            <form onSubmit={handleAiQuery} style={{ padding: '24px' }}>
              <div className="form-group">
                <label className="form-label" htmlFor="assetAiQuery">
                  Describe what is malfunctioning or abnormal:
                </label>
                <textarea
                  id="assetAiQuery"
                  className="form-control"
                  rows={3}
                  placeholder={`Ask a question about ${asset.name} (e.g. error code 04, sample pump failure, cleaning checklist)...`}
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  required
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Grounded on manufacturer service manuals and verified customer procedures.
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={submittingQuery || !query.trim()}
                >
                  {submittingQuery ? (
                    <>
                      <RefreshCw size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
                      Analyzing Manuals...
                    </>
                  ) : (
                    <>
                      <Send size={16} /> Run Diagnostics
                    </>
                  )}
                </button>
              </div>
            </form>

            {aiResult && (
              <div className="rag-answer-panel">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle2 size={16} color="var(--primary)" /> Recommended Resolution
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Backend: {aiResult.retrieval_backend || 'RAG'}
                  </span>
                </div>

                <div className="rag-answer-body" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                  {aiResult.answer}
                </div>

                {aiResult.references && aiResult.references.length > 0 && (
                  <div className="rag-citation-box">
                    <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '8px' }}>
                      TECHNICAL CITATIONS ({aiResult.references.length})
                    </div>
                    <div>
                      {aiResult.references.map((ref, idx) => (
                        <span key={idx} className="citation-chip">
                          <FileText size={14} color="var(--primary)" />
                          <strong>{ref.title || ref.reference}</strong>
                          {ref.heading && <span style={{ opacity: 0.85 }}>({ref.heading})</span>}
                          {ref.doc_type && <span className="badge" style={{ fontSize: '0.65rem' }}>{ref.doc_type}</span>}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Resolve vs Escalate Buttons */}
                <div style={{ 
                  marginTop: '24px', 
                  paddingTop: '20px', 
                  borderTop: '1px solid var(--border)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '12px'
                }}>
                  <div>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                      Did this guidance solve the issue?
                    </span>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      If not resolved, escalate directly into a dispatched service ticket for this machine.
                    </p>
                  </div>

                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                      type="button"
                      className="btn btn-success"
                      onClick={handleResolveInteraction}
                      disabled={resolving || resolutionNotice}
                    >
                      <CheckCircle2 size={16} /> Mark Solved
                    </button>
                    <button
                      type="button"
                      className="btn btn-warning"
                      onClick={handleEscalateInteraction}
                      disabled={escalating || escalationResult}
                    >
                      <Ticket size={16} /> Escalate to Service Ticket
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 4: OPEN TICKETS */}
      {activeTab === 'tickets' && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>Active Service Calls</h3>
            <button className="btn btn-secondary" onClick={loadCalls} title="Refresh calls">
              <RefreshCw size={14} />
            </button>
          </div>
          <div className="table-responsive">
            <table className="table">
              <thead>
                <tr>
                  <th>Ticket ID</th>
                  <th>Complaint</th>
                  <th>Priority</th>
                  <th>Status</th>
                  <th>Created At</th>
                </tr>
              </thead>
              <tbody>
                {loadingCalls ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                      <p>Loading open tickets...</p>
                    </td>
                  </tr>
                ) : openCalls.length === 0 ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      No open service calls for this machine. All systems clear.
                    </td>
                  </tr>
                ) : (
                  openCalls.map(c => (
                    <tr key={c.id}>
                      <td>
                        <span className="badge" style={{ background: '#f1f5f9', color: '#1e293b', fontFamily: 'var(--font-mono)' }}>
                          #{c.servy_id}
                        </span>
                      </td>
                      <td>
                        <div style={{ fontWeight: 600 }}>{c.complaint_type}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{c.complaint_text}</div>
                      </td>
                      <td>
                        <span className={`badge ${c.priority === 'critical' ? 'badge-high' : 'badge-normal'}`}>
                          {c.priority.toUpperCase()}
                        </span>
                      </td>
                      <td>
                        <span className="badge badge-open">{c.status.toUpperCase()}</span>
                      </td>
                      <td style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                        {new Date(c.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 5: SERVICE HISTORY */}
      {activeTab === 'history' && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>Resolved Service Interventions</h3>
            <button className="btn btn-secondary" onClick={loadCalls} title="Refresh history">
              <RefreshCw size={14} />
            </button>
          </div>
          <div className="table-responsive">
            <table className="table">
              <thead>
                <tr>
                  <th>Ticket ID</th>
                  <th>Complaint</th>
                  <th>Resolution Summary</th>
                  <th>Closed Date</th>
                </tr>
              </thead>
              <tbody>
                {loadingCalls ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                      <p>Loading historical calls...</p>
                    </td>
                  </tr>
                ) : historyCalls.length === 0 ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      No previous service interventions recorded for this asset.
                    </td>
                  </tr>
                ) : (
                  historyCalls.map(c => (
                    <tr key={c.id}>
                      <td>
                        <span className="badge" style={{ background: '#f1f5f9', color: '#1e293b', fontFamily: 'var(--font-mono)' }}>
                          #{c.servy_id}
                        </span>
                      </td>
                      <td>
                        <div style={{ fontWeight: 600 }}>{c.complaint_type}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{c.complaint_text}</div>
                      </td>
                      <td>
                        <div style={{ fontSize: '0.82rem', color: '#059669', fontWeight: 500 }}>
                          {c.resolution_text || 'No resolution summary recorded.'}
                        </div>
                      </td>
                      <td style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                        {c.closed_at ? new Date(c.closed_at).toLocaleDateString() : 'Closed'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 6: WARRANTY & ASSET DETAILS */}
      {activeTab === 'warranty' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
          <div className="card">
            <div className="card-title" style={{ marginBottom: '16px' }}>
              <ShieldCheck size={18} color="var(--primary)" /> Coverage Status
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Warranty Expiration Date
                </span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)', marginTop: '2px' }}>
                  {asset.warranty_until ? new Date(asset.warranty_until).toLocaleDateString() : 'No Warranty Registered'}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Remaining Days
                </span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: asset.is_warranty_expired ? '#b91c1c' : '#059669', marginTop: '2px' }}>
                  {asset.warranty_until 
                    ? (asset.is_warranty_expired ? 'Expired' : `${asset.warranty_days_remaining} Days Remaining`) 
                    : 'N/A'}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Operational Status
                </span>
                <div style={{ marginTop: '4px' }}>
                  <span className={`badge ${asset.status === 'active' ? 'badge-resolved' : 'badge-open'}`}>
                    {asset.status.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title" style={{ marginBottom: '16px' }}>
              <Package size={18} color="var(--primary)" /> Machine Identification
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Asset Tag / Code
                </span>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, marginTop: '2px' }}>
                  {asset.asset_code}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Model Number
                </span>
                <div style={{ fontWeight: 600, marginTop: '2px' }}>
                  {asset.model_number || 'Standard Model'}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Serial Number
                </span>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, marginTop: '2px' }}>
                  {asset.serial_number || 'Not Stamped'}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Product Model
                </span>
                <div style={{ fontWeight: 600, marginTop: '2px' }}>
                  {asset.product?.name || 'Generic Equipment'}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
