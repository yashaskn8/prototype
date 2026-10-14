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
  Wrench,
  AlertOctagon,
  ChevronRight,
  UserCheck
} from 'lucide-react';

export function CustomerSupportPage() {
  const { user } = useAuth();
  const [contextData, setContextData] = useState(null);
  const [loadingContext, setLoadingContext] = useState(true);

  const [selectedSite, setSelectedSite] = useState('');
  const [siteAssets, setSiteAssets] = useState([]);
  const [loadingAssets, setLoadingAssets] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [query, setQuery] = useState('');
  const [customClarification, setCustomClarification] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [activeSessionNotice, setActiveSessionNotice] = useState(null);
  const [resolving, setResolving] = useState(false);
  const [escalating, setEscalating] = useState(false);
  const [resolutionNotice, setResolutionNotice] = useState(null);
  const [escalationResult, setEscalationResult] = useState(null);

  // Request race condition tracking: ensure late responses for previous assets are discarded
  const activeAssetRef = useRef('');

  // Load authorized customer sites from API
  async function loadContext() {
    setLoadingContext(true);
    setError(null);
    try {
      const data = await api.get('/api/customer-support/context/');
      setContextData(data);
      if (data.sites && data.sites.length > 0) {
        const firstSiteId = String(data.sites[0].id);
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

  // Dynamically load assets belonging to the selected site
  async function loadAssetsForSite(siteId, initialActiveSession = null) {
    if (!siteId) {
      setSiteAssets([]);
      setSelectedAsset('');
      activeAssetRef.current = '';
      return;
    }
    setLoadingAssets(true);
    try {
      const res = await api.get(`/api/sites/${siteId}/assets/`);
      const assets = res.assets || [];
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
      console.error('Failed to load site assets', err);
      setSiteAssets([]);
      setSelectedAsset('');
      activeAssetRef.current = '';
    } finally {
      setLoadingAssets(false);
    }
  }

  async function checkActiveSessionForAsset(assetId, preloadedSession = null) {
    if (preloadedSession && String(preloadedSession.asset_id) === String(assetId)) {
      setActiveSessionNotice(preloadedSession);
      return;
    }
    try {
      const res = await api.get(`/api/customer-support/context/?asset_id=${assetId}`);
      if (res.active_session) {
        setActiveSessionNotice(res.active_session);
      } else {
        setActiveSessionNotice(null);
      }
    } catch {
      setActiveSessionNotice(null);
    }
  }

  useEffect(() => {
    loadContext();
  }, []);

  function handleSiteChange(e) {
    const newSiteId = e.target.value;
    setSelectedSite(newSiteId);
    setSelectedAsset('');
    activeAssetRef.current = '';
    setResult(null);
    setActiveSessionNotice(null);
    if (newSiteId) {
      loadAssetsForSite(newSiteId);
    } else {
      setSiteAssets([]);
    }
  }

  function handleAssetChange(e) {
    const newAssetId = e.target.value;
    setSelectedAsset(newAssetId);
    activeAssetRef.current = newAssetId;
    setResult(null);
    setError(null);
    setResolutionNotice(null);
    setEscalationResult(null);
    if (newAssetId) {
      checkActiveSessionForAsset(newAssetId);
    } else {
      setActiveSessionNotice(null);
    }
  }

  function handleResumeSession() {
    if (activeSessionNotice?.complaint) {
      setQuery(activeSessionNotice.complaint);
      submitDiagnosticQuery(activeSessionNotice.complaint);
    }
  }

  async function submitDiagnosticQuery(queryText, clarificationAnswer = '') {
    const currentAssetId = selectedAsset;
    if (!queryText.trim() || !selectedSite || !currentAssetId) return;

    setSubmitting(true);
    setError(null);
    setResolutionNotice(null);
    setEscalationResult(null);

    try {
      const payload = {
        site_id: selectedSite,
        asset_id: currentAssetId,
        question: queryText.trim(),
        customer_id: contextData?.customer?.id,
      };
      if (clarificationAnswer) {
        payload.clarification_response = clarificationAnswer.trim();
      }

      const res = await api.post('/api/customer-support/query/', payload);

      // Verify that the user did not switch equipment while the request was in flight
      if (activeAssetRef.current === currentAssetId) {
        setResult(res);
        setCustomClarification('');
      }
    } catch (err) {
      if (activeAssetRef.current === currentAssetId) {
        setError(err.detail || 'Error communicating with AI assistant.');
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
    if (!result?.interaction_id) return;
    setResolving(true);
    try {
      const res = await api.post(`/api/customer-support/interactions/${result.interaction_id}/resolve/`);
      setResolutionNotice(res.detail || 'Marked as resolved. Thank you!');
      setActiveSessionNotice(null);
    } catch (err) {
      setError(err.detail || 'Could not update interaction.');
    } finally {
      setResolving(false);
    }
  }

  async function handleEscalate(escalationNote = '') {
    if (!result?.interaction_id) return;
    setEscalating(true);
    try {
      const payload = escalationNote ? { note: escalationNote } : {};
      const res = await api.post(`/api/customer-support/interactions/${result.interaction_id}/escalate/`, payload);
      setEscalationResult(res);
      setActiveSessionNotice(null);
    } catch (err) {
      setError(err.detail || 'Could not escalate service ticket.');
    } finally {
      setEscalating(false);
    }
  }

  function handleCannotPerformCheck(stepAction) {
    handleEscalate(`Customer reported unable to perform diagnostic step: "${stepAction}"`);
  }

  const selectedAssetObj = siteAssets.find(a => String(a.id) === String(selectedAsset));

  return (
    <div className="content-body">
      {/* Page Header */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <span className="badge" style={{ background: '#e0e7ff', color: '#4338ca', fontWeight: 700 }}>
            CUSTOMER AI SUPPORT
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Pre-Ticket Diagnostic Self-Service</span>
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Equipment Troubleshooting Assistant
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Resolve issues safely using approved manufacturer guidance and Knowledge Base procedures before creating a service dispatch.
        </p>
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
            A certified service technician has been assigned to your case. Your original issue and diagnostic observations have been safely handed off.
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Status: <strong>{escalationResult.status?.toUpperCase() || 'ASSIGNED'}</strong> | Priority: <strong>{escalationResult.priority?.toUpperCase() || 'NORMAL'}</strong>
          </div>
        </div>
      )}

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
                Asset Code: {selectedAssetObj.asset_code || 'N/A'}<br />
                Model: {selectedAssetObj.model_number || 'Standard'}<br />
                Product: {selectedAssetObj.product__name || 'Industrial Equipment'}
              </div>
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
            Customer account: <strong>{contextData?.customer?.name || user?.username}</strong>. Diagnostic queries are grounded exclusively in authorized documentation for your equipment.
          </div>
        </div>

        {/* Query Input & Guided Diagnostic View */}
        <div className="rag-box">
          <div className="rag-header">
            <div className="rag-header-left">
              <div className="rag-icon-badge">
                <Bot size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>AI Troubleshooting Assistant</h3>
                <span style={{ fontSize: '0.75rem', opacity: 0.85 }}>Source-Grounded Pre-Ticket Diagnostics</span>
              </div>
            </div>
          </div>

          {/* Issue Intake Form */}
          <form onSubmit={handleQuery} style={{ padding: '24px' }}>
            <div className="form-group">
              <label className="form-label" htmlFor="userQuery">
                Describe the problem or symptom you are observing:
              </label>
              <textarea
                id="userQuery"
                className="form-control"
                rows={3}
                placeholder="e.g. The analyzer switches on, but fails to draw a sample. What should I check?"
                value={query}
                onChange={e => setQuery(e.target.value)}
                required
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Enter your issue in plain language. If the symptom is ambiguous, Servy will ask for clarification.
              </div>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={submitting || !query.trim() || !selectedSite || !selectedAsset}
              >
                {submitting ? (
                  <>
                    <RefreshCw size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
                    Analyzing Guidance...
                  </>
                ) : (
                  <>
                    <Send size={16} /> Run Diagnostics
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Results Display */}
          {result && (
            <div className="rag-answer-panel" style={{ padding: '0 24px 24px 24px' }}>

              {/* Case 1: Clarification Needed */}
              {result.response_type === 'CLARIFICATION' && (
                <div style={{
                  background: '#fefce8',
                  border: '1px solid #fde68a',
                  borderRadius: 'var(--radius-md)',
                  padding: '20px',
                  marginBottom: '20px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#a16207', fontWeight: 700, fontSize: '0.95rem', marginBottom: '8px' }}>
                    <HelpCircle size={18} /> Clarification Needed
                  </div>
                  <p style={{ fontSize: '0.9rem', color: '#854d0e', marginBottom: '14px' }}>
                    {result.clarification_question || result.summary}
                  </p>

                  {result.clarification_options && result.clarification_options.length > 0 && (
                    <div style={{ marginBottom: '14px' }}>
                      <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#a16207', marginBottom: '8px' }}>
                        SELECT AN OBSERVATION:
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {result.clarification_options.map((opt, idx) => (
                          <button
                            key={idx}
                            type="button"
                            className="btn btn-sm btn-outline-primary"
                            style={{ textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#fff' }}
                            onClick={() => handleClarificationOption(opt)}
                            disabled={submitting}
                          >
                            <span>{opt}</span>
                            <ChevronRight size={14} />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <form onSubmit={handleCustomClarificationSubmit} style={{ marginTop: '12px' }}>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <input
                        type="text"
                        className="form-control"
                        placeholder="Or type specific detail (e.g. Error 04 displayed)"
                        value={customClarification}
                        onChange={e => setCustomClarification(e.target.value)}
                        disabled={submitting}
                      />
                      <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={submitting || !customClarification.trim()}
                      >
                        Submit
                      </button>
                    </div>
                  </form>
                </div>
              )}

              {/* Case 2: Severe Safety Hazard Alert */}
              {result.response_type === 'ESCALATION_REQUIRED' && result.stop_conditions && result.stop_conditions.length > 0 && (
                <div style={{
                  background: '#fef2f2',
                  border: '1px solid #fecaca',
                  borderRadius: 'var(--radius-md)',
                  padding: '20px',
                  marginBottom: '20px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#b91c1c', fontWeight: 800, fontSize: '1rem', marginBottom: '8px' }}>
                    <AlertOctagon size={20} /> Severe Safety Warning — Immediate Action Required
                  </div>
                  <p style={{ fontSize: '0.9rem', color: '#991b1b', lineHeight: 1.6, marginBottom: '14px' }}>
                    {result.stop_conditions[0]}
                  </p>
                  <button
                    type="button"
                    className="btn btn-danger"
                    style={{ background: '#dc2626', color: '#fff' }}
                    onClick={() => handleEscalate('Safety hazard escalation')}
                    disabled={escalating || escalationResult}
                  >
                    <Ticket size={16} /> {escalating ? 'Dispatching...' : 'Request Immediate Service Technician'}
                  </button>
                </div>
              )}

              {/* Diagnostic Summary */}
              {result.summary && result.response_type !== 'CLARIFICATION' && (
                <div style={{
                  background: '#eff6ff',
                  border: '1px solid #bfdbfe',
                  borderRadius: 'var(--radius-md)',
                  padding: '16px',
                  marginBottom: '16px',
                }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#1e40af', letterSpacing: '0.03em', marginBottom: '6px' }}>
                    🔍 DIAGNOSTIC SUMMARY
                  </div>
                  <div style={{ fontSize: '0.9rem', color: '#1e3a8a', lineHeight: 1.6 }}>
                    {result.summary}
                  </div>
                </div>
              )}

              {/* Initial Safe Checks */}
              {result.check_first && result.check_first.length > 0 && (
                <div style={{
                  background: '#fefce8',
                  border: '1px solid #fde68a',
                  borderRadius: 'var(--radius-md)',
                  padding: '16px',
                  marginBottom: '16px',
                }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#a16207', letterSpacing: '0.03em', marginBottom: '8px' }}>
                    ⚡ CHECK FIRST (INITIAL SAFE OBSERVATIONS)
                  </div>
                  <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.88rem', color: '#713f12' }}>
                    {result.check_first.map((chk, idx) => (
                      <li key={idx} style={{ marginBottom: '6px' }}>{chk}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Customer Safe Troubleshooting Steps */}
              {result.customer_safe_actions && result.customer_safe_actions.length > 0 && (
                <div style={{
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  borderRadius: 'var(--radius-md)',
                  padding: '16px',
                  marginBottom: '16px',
                }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#166534', letterSpacing: '0.03em', marginBottom: '12px' }}>
                    🔧 APPROVED CUSTOMER TROUBLESHOOTING STEPS
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {result.customer_safe_actions.map((act, idx) => (
                      <div key={idx} style={{
                        background: '#ffffff',
                        border: '1px solid #dcfce7',
                        borderRadius: 'var(--radius-sm)',
                        padding: '12px 14px',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px' }}>
                          <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                            <span style={{
                              background: '#16a34a',
                              color: '#fff',
                              borderRadius: '50%',
                              width: '24px',
                              height: '24px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: '0.75rem',
                              fontWeight: 700,
                              flexShrink: 0,
                            }}>{act.step_number || idx + 1}</span>
                            <div>
                              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#14532d' }}>
                                {act.action}
                              </div>
                              {act.why && (
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '4px' }}>
                                  Reason: {act.why}
                                </div>
                              )}
                            </div>
                          </div>
                          <span className="badge" style={{ background: '#dcfce7', color: '#166534', fontSize: '0.68rem', flexShrink: 0 }}>
                            Customer Safe
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '8px' }}>
                          <button
                            type="button"
                            className="btn btn-sm btn-link"
                            style={{ fontSize: '0.75rem', color: '#b91c1c', padding: 0 }}
                            onClick={() => handleCannotPerformCheck(act.action)}
                            disabled={escalating}
                          >
                            I cannot perform this check →
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Technician-Only Warnings Banner */}
              {result.technician_only_actions && result.technician_only_actions.length > 0 && (
                <div style={{
                  background: '#fef2f2',
                  border: '1px solid #fecaca',
                  borderRadius: 'var(--radius-md)',
                  padding: '16px',
                  marginBottom: '16px',
                }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#b91c1c', letterSpacing: '0.03em', marginBottom: '8px' }}>
                    🚨 TECHNICIAN-ONLY WORK REQUIRED (DO NOT ATTEMPT)
                  </div>
                  <p style={{ fontSize: '0.82rem', color: '#7f1d1d', marginBottom: '10px' }}>
                    The following procedures require internal cabinet access, mains voltage handling, or certified calibration tools:
                  </p>
                  <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.85rem', color: '#991b1b' }}>
                    {result.technician_only_actions.map((tech, idx) => (
                      <li key={idx} style={{ marginBottom: '6px' }}>
                        <strong>{tech.action}</strong>
                        {tech.warning && <span style={{ fontSize: '0.78rem', opacity: 0.9 }}> ({tech.warning})</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Expected Result */}
              {result.expected_result && (
                <div style={{
                  background: '#f0f9ff',
                  border: '1px solid #bae6fd',
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 16px',
                  marginBottom: '16px',
                }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0369a1', letterSpacing: '0.03em', marginBottom: '4px' }}>
                    ✅ EXPECTED RESULT
                  </div>
                  <div style={{ fontSize: '0.88rem', color: '#0c4a6e' }}>
                    {result.expected_result}
                  </div>
                </div>
              )}

              {/* Stop & Escalate Conditions */}
              {result.stop_conditions && result.stop_conditions.length > 0 && (
                <div style={{
                  background: '#faf5ff',
                  border: '1px solid #e9d5ff',
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 16px',
                  marginBottom: '16px',
                }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#7e22ce', letterSpacing: '0.03em', marginBottom: '6px' }}>
                    🛑 STOP AND ESCALATE IF
                  </div>
                  <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.85rem', color: '#581c87' }}>
                    {result.stop_conditions.map((stop, idx) => (
                      <li key={idx} style={{ marginBottom: '4px' }}>{stop}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Authorized Citations */}
              {result.citations && result.citations.length > 0 && (
                <div className="rag-citation-box" style={{ marginTop: '16px' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '8px' }}>
                    AUTHORIZED TECHNICAL REFERENCES ({result.citations.length})
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {result.citations.map((ref, idx) => (
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

              {/* Action Buttons: Resolve vs Escalate */}
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
                    Did this guidance resolve your issue?
                  </span>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Confirming resolution safely closes this diagnostic inquiry.
                  </p>
                </div>

                <div style={{ display: 'flex', gap: '10px' }}>
                  <button
                    type="button"
                    className="btn btn-success"
                    onClick={handleResolve}
                    disabled={resolving || resolutionNotice}
                  >
                    <CheckCircle2 size={16} /> {resolving ? 'Recording...' : 'Mark Issue Resolved'}
                  </button>

                  <button
                    type="button"
                    className="btn btn-primary"
                    style={{ background: 'linear-gradient(135deg, #e11d48 0%, #be123c 100%)' }}
                    onClick={() => handleEscalate('Customer requested technician escalation')}
                    disabled={escalating || escalationResult}
                  >
                    <Ticket size={16} /> {escalating ? 'Creating Ticket...' : 'Escalate to Service Ticket'}
                  </button>
                </div>
              </div>

            </div>
          )}
        </div>

      </div>
    </div>
  );
}
