import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../AuthContext';
import { 
  Bot, 
  Send, 
  CheckCircle2, 
  AlertTriangle, 
  FileText, 
  History, 
  ShieldCheck, 
  RefreshCw, 
  Building, 
  Package, 
  ArrowRight,
  Ticket
} from 'lucide-react';

export function CustomerSupportPage({ onNavigate }) {
  const { user } = useAuth();
  const [contextData, setContextData] = useState(null);
  const [loadingContext, setLoadingContext] = useState(true);

  const [selectedSite, setSelectedSite] = useState('');
  const [selectedAsset, setSelectedAsset] = useState('');
  const [query, setQuery] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [resolving, setResolving] = useState(false);
  const [escalating, setEscalating] = useState(false);
  const [resolutionNotice, setResolutionNotice] = useState(null);
  const [escalationResult, setEscalationResult] = useState(null);

  // Load customer sites and assets
  async function loadContext() {
    setLoadingContext(true);
    setError(null);
    try {
      const data = await api.get('/api/customer-support/context/');
      setContextData(data);
      if (data.sites && data.sites.length > 0) {
        setSelectedSite(data.sites[0].id);
      }
    } catch (err) {
      setError(err.detail || 'Failed to load customer profile context.');
    } finally {
      setLoadingContext(false);
    }
  }

  useEffect(() => {
    loadContext();
  }, []);

  // Filter assets by selected site
  const siteAssets = contextData?.assets?.filter(
    a => !selectedSite || String(a.site_id) === String(selectedSite)
  ) || [];

  async function handleQuery(e) {
    e.preventDefault();
    if (!query.trim()) return;

    setSubmitting(true);
    setError(null);
    setResolutionNotice(null);
    setEscalationResult(null);

    try {
      const res = await api.post('/api/customer-support/query/', {
        site_id: selectedSite || null,
        asset_id: selectedAsset || null,
        query: query.trim(),
      });
      setResult(res);
    } catch (err) {
      setError(err.detail || 'Error communicating with AI assistant.');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleResolve() {
    if (!result?.interaction_id) return;
    setResolving(true);
    try {
      const res = await api.post(`/api/customer-support/${result.interaction_id}/resolve/`);
      setResolutionNotice(res.detail || 'Marked as resolved. Thank you!');
    } catch (err) {
      setError(err.detail || 'Could not update interaction.');
    } finally {
      setResolving(false);
    }
  }

  async function handleEscalate() {
    if (!result?.interaction_id) return;
    setEscalating(true);
    try {
      const res = await api.post(`/api/customer-support/${result.interaction_id}/escalate/`);
      setEscalationResult(res);
    } catch (err) {
      setError(err.detail || 'Could not escalate service ticket.');
    } finally {
      setEscalating(false);
    }
  }

  return (
    <div className="content-body">
      {/* Page Header */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <span className="badge" style={{ background: '#e0e7ff', color: '#4338ca', fontWeight: 700 }}>
            WORKFLOW 1
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Pre-Ticket Self-Service</span>
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Customer AI Self-Service Diagnostic
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Troubleshoot issues against verified technical manuals and historical cases before creating a service dispatch.
        </p>
      </div>

      {error && (
        <div className="alert alert-danger">
          <AlertTriangle size={18} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      {resolutionNotice && (
        <div className="alert alert-success">
          <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
          <div>{resolutionNotice}</div>
        </div>
      )}

      {escalationResult && (
        <div className="alert alert-info" style={{ display: 'block' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, marginBottom: '4px' }}>
            <Ticket size={18} /> Ticket Successfully Created: {escalationResult.servy_id}
          </div>
          <p style={{ fontSize: '0.85rem' }}>
            A field service ticket has been dispatched to our engineering team. You can track this in your Service Calls register.
          </p>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px', alignItems: 'start' }}>
        
        {/* Equipment Selector Sidebar Card */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: '16px' }}>
            <Building size={18} color="var(--primary)" /> Equipment Selection
          </div>

          <div className="form-group">
            <label className="form-label">Facility / Site</label>
            <select
              className="form-control"
              value={selectedSite}
              onChange={e => {
                setSelectedSite(e.target.value);
                setSelectedAsset('');
              }}
              disabled={loadingContext}
            >
              <option value="">All Customer Sites</option>
              {contextData?.sites?.map(s => (
                <option key={s.id} value={s.id}>{s.name} ({s.city || 'Site'})</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Equipment / Asset (Optional)</label>
            <select
              className="form-control"
              value={selectedAsset}
              onChange={e => setSelectedAsset(e.target.value)}
              disabled={loadingContext}
            >
              <option value="">General Inquiry / Any Asset</option>
              {siteAssets.map(a => (
                <option key={a.id} value={a.id}>
                  {a.name} {a.model_number ? `(${a.model_number})` : ''}
                </option>
              ))}
            </select>
          </div>

          <div style={{ 
            background: 'var(--bg-app)', 
            padding: '14px', 
            borderRadius: 'var(--radius-md)', 
            border: '1px solid var(--border)',
            fontSize: '0.78rem',
            color: 'var(--text-muted)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
              <ShieldCheck size={14} color="#10b981" /> Tenant Security Active
            </div>
            Customer account: <strong>{contextData?.customer?.name || user?.username}</strong>. Queries are strictly restricted to your company's equipment manuals and historical cases.
          </div>
        </div>

        {/* Query Input & RAG Output Box */}
        <div className="rag-box">
          <div className="rag-header">
            <div className="rag-header-left">
              <div className="rag-icon-badge">
                <Bot size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>AI Troubleshooting Assistant</h3>
                <span style={{ fontSize: '0.75rem', opacity: 0.85 }}>Hardened RAG with Grounded Citations</span>
              </div>
            </div>

            <div style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }}></span>
              Ready
            </div>
          </div>

          {/* Prompt / Question Form */}
          <form onSubmit={handleQuery} style={{ padding: '24px' }}>
            <div className="form-group">
              <label className="form-label" htmlFor="userQuery">
                Describe the problem or symptom you are observing:
              </label>
              <textarea
                id="userQuery"
                className="form-control"
                rows={3}
                placeholder="e.g. The analyzer is not drawing a sample. What should I check?"
                value={query}
                onChange={e => setQuery(e.target.value)}
                required
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Protected against prompt injection and query control exploits.
              </div>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={submitting || !query.trim()}
              >
                {submitting ? (
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

          {/* Results Display */}
          {result && (
            <div className="rag-answer-panel">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle2 size={16} color="var(--primary)" /> Diagnostic Guidance
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Backend: {result.retrieval_backend || 'Dense Semantic (ChromaDB)'}
                </span>
              </div>

              <div className="rag-answer-body">
                {result.answer}
              </div>

              {/* Verified Citations */}
              {result.references && result.references.length > 0 && (
                <div className="rag-citation-box">
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '8px' }}>
                    VERIFIED TECHNICAL REFERENCES ({result.references.length})
                  </div>
                  <div>
                    {result.references.map((ref, idx) => (
                      <span key={idx} className="citation-chip">
                        <FileText size={14} color="var(--primary)" />
                        <strong>{ref.title || ref.source}</strong>
                        {ref.chunk_index !== undefined && <span style={{ opacity: 0.7 }}>(Sec. #{ref.chunk_index})</span>}
                        {ref.doc_type && <span className="badge" style={{ fontSize: '0.65rem' }}>{ref.doc_type}</span>}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Past Resolutions (Same Customer) */}
              {result.past_resolutions && result.past_resolutions.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#15803d', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <History size={14} /> PREVIOUS SIMILAR RESOLUTIONS ON YOUR SITES
                  </div>
                  {result.past_resolutions.map((pr, idx) => (
                    <div key={idx} className="resolution-card">
                      <h4>Case {pr.servy_id || `#${idx + 1}`}: {pr.complaint_type || 'Service Issue'}</h4>
                      <p><strong>Fix Applied:</strong> {pr.resolution_text || 'Standard calibration performed.'}</p>
                    </div>
                  ))}
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
                    Did this answer solve your issue?
                  </span>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Closing without dispatch saves downtime and maintenance costs.
                  </p>
                </div>

                <div style={{ display: 'flex', gap: '10px' }}>
                  <button
                    type="button"
                    className="btn btn-success"
                    onClick={handleResolve}
                    disabled={resolving || resolutionNotice}
                  >
                    <CheckCircle2 size={16} /> {resolving ? 'Saving...' : 'Issue Resolved'}
                  </button>

                  <button
                    type="button"
                    className="btn btn-primary"
                    style={{ background: 'linear-gradient(135deg, #e11d48 0%, #be123c 100%)' }}
                    onClick={handleEscalate}
                    disabled={escalating || escalationResult}
                  >
                    <Ticket size={16} /> {escalating ? 'Escalating...' : 'Escalate to Service Ticket'}
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
