import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { 
  Wrench, 
  Send, 
  FileText, 
  Lock, 
  RefreshCw, 
  AlertCircle, 
  History, 
  PhoneCall, 
  User, 
  MapPin, 
  Cpu
} from 'lucide-react';

export function EngineerCopilotPage({ initialCallId }) {
  const [calls, setCalls] = useState([]);
  const [loadingCalls, setLoadingCalls] = useState(true);
  const [selectedCallId, setSelectedCallId] = useState(initialCallId || '');
  const [callDetail, setCallDetail] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const [query, setQuery] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Load calls list for selection
  async function loadCalls() {
    setLoadingCalls(true);
    try {
      const res = await api.get('/api/calls/');
      const list = res.calls || (Array.isArray(res) ? res : []);
      setCalls(list);
      if (!selectedCallId && list.length > 0) {
        setSelectedCallId(list[0].id);
      }
    } catch (err) {
      console.error('Error fetching calls', err);
    } finally {
      setLoadingCalls(false);
    }
  }

  // Load selected call details
  async function loadDetail(callId) {
    if (!callId) return;
    setLoadingDetail(true);
    try {
      const res = await api.get(`/api/calls/${callId}/`);
      setCallDetail(res.call);
    } catch (err) {
      console.error('Error fetching call detail', err);
    } finally {
      setLoadingDetail(false);
    }
  }

  useEffect(() => {
    loadCalls();
  }, []);

  useEffect(() => {
    if (selectedCallId) {
      loadDetail(selectedCallId);
      setResult(null);
    }
  }, [selectedCallId]);

  async function handleQuery(e) {
    e.preventDefault();
    if (!query.trim() || !selectedCallId) return;

    setSubmitting(true);
    setError(null);

    try {
      const res = await api.post('/api/engineer-copilot/query/', {
        call_id: selectedCallId,
        question: query.trim(),
      });
      setResult(res);
    } catch (err) {
      setError(err.detail || 'Failed to query engineer copilot.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="content-body">
      {/* Page Header */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <span className="badge" style={{ background: '#fdf4ff', color: '#7c3aed', fontWeight: 700 }}>
            WORKFLOW 2
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Post-Ticket Field Engineering Copilot</span>
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Technician & Dispatch AI Copilot
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Query technical specs, wiring schematics, and cross-customer fleet resolution memory for active field tickets.
        </p>
      </div>

      {error && (
        <div className="alert alert-danger">
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '24px', alignItems: 'start' }}>
        
        {/* Call Selection & Ticket Context Card */}
        <div>
          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-title" style={{ marginBottom: '16px' }}>
              <PhoneCall size={18} color="var(--primary)" /> Select Active Ticket
            </div>

            <div className="form-group">
              <label className="form-label">Service Call</label>
              <select
                className="form-control"
                value={selectedCallId}
                onChange={e => setSelectedCallId(e.target.value)}
                disabled={loadingCalls}
              >
                {calls.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.servy_id} - {c.customer__name} ({c.complaint_type || 'Issue'})
                  </option>
                ))}
              </select>
            </div>

            {loadingDetail ? (
              <div style={{ textAlign: 'center', padding: '16px', color: 'var(--text-muted)' }}>
                Loading ticket details...
              </div>
            ) : callDetail ? (
              <div style={{ 
                background: '#f8fafc', 
                borderRadius: 'var(--radius-md)', 
                padding: '16px', 
                border: '1px solid var(--border)',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 800, color: 'var(--primary)', fontSize: '0.95rem' }}>
                    {callDetail.servy_id}
                  </span>
                  <span className={`badge badge-${callDetail.status}`}>
                    {callDetail.status}
                  </span>
                </div>

                <div style={{ fontSize: '0.82rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <User size={14} color="#64748b" /> {callDetail.customer?.name}
                </div>

                <div style={{ fontSize: '0.82rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <MapPin size={14} color="#64748b" /> {callDetail.site?.name || 'Main Site'}
                </div>

                <div style={{ fontSize: '0.82rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Cpu size={14} color="#64748b" /> {callDetail.asset?.name} {callDetail.asset?.model_number ? `[${callDetail.asset?.model_number}]` : ''}
                </div>

                <div style={{ 
                  marginTop: '6px', 
                  paddingTop: '8px', 
                  borderTop: '1px solid #e2e8f0',
                  fontSize: '0.8rem',
                  color: 'var(--text-muted)'
                }}>
                  <strong style={{ color: 'var(--text-main)' }}>Complaint:</strong> {callDetail.complaint_text || callDetail.complaint_type}
                </div>
              </div>
            ) : null}
          </div>

          <div style={{ 
            background: '#ffffff', 
            borderRadius: 'var(--radius-md)', 
            padding: '16px', 
            border: '1px solid var(--border)',
            fontSize: '0.78rem',
            color: 'var(--text-muted)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, color: '#7c3aed', marginBottom: '6px' }}>
              <Lock size={14} /> Confidential Access Permitted
            </div>
            As authorized technical staff, search includes proprietary internal blueprints and sanitized case history across your tenant.
          </div>
        </div>

        {/* Query Input & Copilot Output */}
        <div className="rag-box">
          <div className="rag-header" style={{ background: 'linear-gradient(135deg, #1e1b4b 0%, #4c1d95 100%)' }}>
            <div className="rag-header-left">
              <div className="rag-icon-badge">
                <Wrench size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Field Engineering AI Copilot</h3>
                <span style={{ fontSize: '0.75rem', opacity: 0.85 }}>Technical manuals & fleet repair memory</span>
              </div>
            </div>

            <div style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#a855f7' }}></span>
              Online
            </div>
          </div>

          <form onSubmit={handleQuery} style={{ padding: '24px' }}>
            <div className="form-group">
              <label className="form-label" htmlFor="engQuery">
                Ask a technical diagnostic or procedure question for ticket {callDetail?.servy_id || ''}:
              </label>
              <textarea
                id="engQuery"
                className="form-control"
                rows={3}
                placeholder="e.g. What is the calibration procedure for the flow sensor? Check clearance tolerances."
                value={query}
                onChange={e => setQuery(e.target.value)}
                required
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Includes confidential manuals and sanitized resolution memory.
              </div>
              <button
                type="submit"
                className="btn btn-primary"
                style={{ background: 'linear-gradient(135deg, #7c3aed 0%, #6366f1 100%)' }}
                disabled={submitting || !query.trim() || !selectedCallId}
              >
                {submitting ? (
                  <>
                    <RefreshCw size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
                    Retrieving Schematics...
                  </>
                ) : (
                  <>
                    <Send size={16} /> Consult Copilot
                  </>
                )}
              </button>
            </div>
          </form>

          {result && (
            <div className="rag-answer-panel">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Wrench size={16} color="var(--accent)" /> Technical Recommendation
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
                    RETRIEVED MANUALS & BLUEPRINTS ({result.references.length})
                  </div>
                  <div>
                    {result.references.map((ref, idx) => (
                      <span key={idx} className="citation-chip">
                        <FileText size={14} color="#7c3aed" />
                        <strong>{ref.title || ref.source}</strong>
                        {ref.chunk_index !== undefined && <span style={{ opacity: 0.7 }}>(Sec. #{ref.chunk_index})</span>}
                        {ref.is_confidential && (
                          <span className="badge badge-confidential" style={{ fontSize: '0.65rem' }}>
                            <Lock size={10} /> Confidential
                          </span>
                        )}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Cross-Customer Sanitized Past Cases */}
              {result.past_resolutions && result.past_resolutions.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#1e40af', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <History size={14} /> FLEET-WIDE SANITIZED CASE MEMORY ({result.past_resolutions.length})
                  </div>
                  {result.past_resolutions.map((pr, idx) => (
                    <div key={idx} className="resolution-card" style={{ background: '#eff6ff', borderColor: '#bfdbfe', padding: '10px 14px', borderRadius: '6px', marginBottom: '8px' }}>
                      <h4 style={{ color: '#1d4ed8', margin: '0 0 4px 0', fontSize: '0.9rem', fontWeight: 600 }}>
                        Resolved Case #{pr.servy_id || pr.service_call_id}: {pr.complaint_type || pr.title || 'Fleet Resolution'}
                      </h4>
                      <p style={{ color: '#1e40af', margin: 0, fontSize: '0.85rem' }}>
                        <strong>Action Taken:</strong> {pr.resolution_text || 'Resolved according to standard technical procedures.'}
                      </p>
                    </div>
                  ))}
                </div>
              )}

            </div>
          )}
        </div>

      </div>
    </div>
  );
}
