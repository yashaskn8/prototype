import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../AuthContext';
import { 
  PhoneCall, 
  Search, 
  Filter, 
  Download, 
  Upload, 
  Eye, 
  X, 
  AlertCircle, 
  CheckCircle2, 
  RefreshCw, 
  FileSpreadsheet, 
  Wrench,
  Clock,
  User,
  MapPin,
  Cpu
} from 'lucide-react';

export function CallRegisterPage({ onOpenCopilot }) {
  const { user } = useAuth();
  const [calls, setCalls] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [priorityFilter, setPriorityFilter] = useState('');

  // Call Detail Modal State
  const [selectedCall, setSelectedCall] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailData, setDetailData] = useState(null);

  // Excel Import Modal State
  const [showImportModal, setShowImportModal] = useState(false);
  const [importFile, setImportFile] = useState(null);
  const [importing, setImporting] = useState(false);
  const [importNotice, setImportNotice] = useState(null);

  async function loadCalls() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/calls/');
      setCalls(res.calls || []);
    } catch (err) {
      setError(err.detail || 'Failed to load call register.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadCalls();
  }, []);

  async function openCallDetail(callId) {
    setSelectedCall(callId);
    setDetailLoading(true);
    setDetailData(null);
    try {
      const res = await api.get(`/api/calls/${callId}/`);
      setDetailData(res);
    } catch (err) {
      console.error('Failed to load call detail', err);
    } finally {
      setDetailLoading(false);
    }
  }

  function handleExport() {
    window.location.href = '/api/calls/export/';
  }

  async function handleImportSubmit(e) {
    e.preventDefault();
    if (!importFile) return;

    setImporting(true);
    setImportNotice(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', importFile);

    try {
      const res = await api.upload('/api/calls/import/', formData);
      setImportNotice({ type: 'success', msg: res.detail });
      setImportFile(null);
      loadCalls();
      setTimeout(() => setShowImportModal(false), 2000);
    } catch (err) {
      setImportNotice({ type: 'danger', msg: err.detail || 'Failed to import Excel file.' });
    } finally {
      setImporting(false);
    }
  }

  // Filter calls
  const filteredCalls = calls.filter(c => {
    const matchSearch = !search || 
      c.servy_id.toLowerCase().includes(search.toLowerCase()) ||
      (c.customer__name && c.customer__name.toLowerCase().includes(search.toLowerCase())) ||
      (c.complaint_type && c.complaint_type.toLowerCase().includes(search.toLowerCase()));
    
    const matchStatus = !statusFilter || c.status === statusFilter;
    const matchPriority = !priorityFilter || c.priority === priorityFilter;
    return matchSearch && matchStatus && matchPriority;
  });

  const isStaff = user?.role !== 'customer';

  return (
    <div className="content-body">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
            Service Call Register
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            Comprehensive ticket management, SLA monitoring, and resolution history.
          </p>
        </div>

        {isStaff && (
          <div style={{ display: 'flex', gap: '10px' }}>
            <button className="btn btn-secondary" onClick={handleExport}>
              <Download size={16} /> Export Excel
            </button>
            <button className="btn btn-primary" onClick={() => setShowImportModal(true)}>
              <Upload size={16} /> Import Excel
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="alert alert-danger">
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      {/* Filter Bar */}
      <div className="card" style={{ marginBottom: '20px', padding: '16px 20px' }}>
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: 1, minWidth: '240px' }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '12px' }} />
            <input
              type="text"
              className="form-control"
              style={{ paddingLeft: '38px', paddingRight: '12px' }}
              placeholder="Search by Servy ID, Customer, or Issue..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <div style={{ width: '160px' }}>
            <select
              className="form-control"
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
            >
              <option value="">All Statuses</option>
              <option value="open">Open</option>
              <option value="assigned">Assigned</option>
              <option value="in_progress">In Progress</option>
              <option value="resolved">Resolved</option>
              <option value="closed">Closed</option>
            </select>
          </div>

          <div style={{ width: '160px' }}>
            <select
              className="form-control"
              value={priorityFilter}
              onChange={e => setPriorityFilter(e.target.value)}
            >
              <option value="">All Priorities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>

          <button className="btn btn-secondary" onClick={loadCalls} title="Refresh register">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-responsive">
          <table className="table">
            <thead>
              <tr>
                <th>Servy ID</th>
                <th>Customer & Site</th>
                <th>Issue / Complaint</th>
                <th>Priority</th>
                <th>Status</th>
                <th>Created</th>
                <th style={{ textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                    <p>Loading service call records...</p>
                  </td>
                </tr>
              ) : filteredCalls.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    No service calls match the specified filter.
                  </td>
                </tr>
              ) : (
                filteredCalls.map(c => (
                  <tr key={c.id}>
                    <td style={{ fontWeight: 700, color: 'var(--primary)' }}>
                      {c.servy_id}
                    </td>
                    <td>
                      <div style={{ fontWeight: 600 }}>{c.customer__name}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{c.site__name || 'Main Site'}</div>
                    </td>
                    <td>
                      <div style={{ fontWeight: 500 }}>{c.complaint_type || 'General Service'}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {c.complaint_text}
                      </div>
                    </td>
                    <td>
                      <span className={`badge badge-${c.priority}`}>
                        {c.priority}
                      </span>
                    </td>
                    <td>
                      <span className={`badge badge-${c.status}`}>
                        {c.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      {new Date(c.created_at).toLocaleDateString()}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => openCallDetail(c.id)}
                      >
                        <Eye size={14} /> View
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Call Detail Modal */}
      {selectedCall && (
        <div className="modal-backdrop">
          <div className="modal-content" style={{ maxWidth: '720px' }}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <PhoneCall size={20} color="var(--primary)" />
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800 }}>
                  Ticket Detail: {detailData?.call?.servy_id || 'Loading...'}
                </h3>
              </div>
              <button 
                onClick={() => setSelectedCall(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8' }}
              >
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              {detailLoading || !detailData ? (
                <div style={{ textAlign: 'center', padding: '36px' }}>Loading ticket details...</div>
              ) : (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                    <div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Customer & Site</span>
                      <div style={{ fontWeight: 600 }}>{detailData.call.customer?.name}</div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{detailData.call.site?.name}</div>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Asset / Equipment</span>
                      <div style={{ fontWeight: 600 }}>{detailData.call.asset?.name || 'Unassigned'}</div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Model: {detailData.call.asset?.model_number || 'N/A'}</div>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Assigned Technician</span>
                      <div style={{ fontWeight: 600 }}>{detailData.call.technician?.name || 'Unassigned Dispatch'}</div>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Status & Priority</span>
                      <div>
                        <span className={`badge badge-${detailData.call.status}`} style={{ marginRight: '6px' }}>
                          {detailData.call.status}
                        </span>
                        <span className={`badge badge-${detailData.call.priority}`}>
                          {detailData.call.priority}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div style={{ background: '#f8fafc', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', marginBottom: '20px' }}>
                    <h4 style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '4px' }}>
                      COMPLAINT & SYMPTOM
                    </h4>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>
                      {detailData.call.complaint_text || detailData.call.complaint_type}
                    </p>
                  </div>

                  {detailData.call.resolution_text && (
                    <div style={{ background: '#f0fdf4', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid #bbf7d0', marginBottom: '20px' }}>
                      <h4 style={{ fontSize: '0.82rem', fontWeight: 700, color: '#15803d', marginBottom: '4px' }}>
                        FINAL RESOLUTION RECORD
                      </h4>
                      <p style={{ fontSize: '0.88rem', color: '#166534' }}>
                        {detailData.call.resolution_text}
                      </p>
                    </div>
                  )}

                  {detailData.updates && detailData.updates.length > 0 && (
                    <div style={{ marginBottom: '20px' }}>
                      <h4 style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '8px' }}>
                        SERVICE TIMELINE UPDATES ({detailData.updates.length})
                      </h4>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {detailData.updates.map(up => (
                          <div key={up.id} style={{ padding: '10px 14px', background: '#ffffff', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: '0.82rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                              <strong>{up.author__full_name || 'System'}</strong>
                              <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                                {new Date(up.created_at).toLocaleString()}
                              </span>
                            </div>
                            <div>{up.note}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="modal-footer">
              {isStaff && (
                <button 
                  className="btn btn-primary"
                  onClick={() => {
                    const id = selectedCall;
                    setSelectedCall(null);
                    onOpenCopilot(id);
                  }}
                >
                  <Wrench size={16} /> Open in Engineer Copilot
                </button>
              )}
              <button className="btn btn-secondary" onClick={() => setSelectedCall(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Excel Import Modal */}
      {showImportModal && (
        <div className="modal-backdrop">
          <div className="modal-content" style={{ maxWidth: '520px' }}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileSpreadsheet size={20} color="var(--primary)" />
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800 }}>Import Service Calls (Excel)</h3>
              </div>
              <button 
                onClick={() => setShowImportModal(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8' }}
              >
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleImportSubmit}>
              <div className="modal-body">
                {importNotice && (
                  <div className={`alert alert-${importNotice.type}`}>
                    {importNotice.type === 'success' ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
                    <div>{importNotice.msg}</div>
                  </div>
                )}

                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
                  Upload a standardized <code>.xlsx</code> workbook with columns: Customer, Site, Asset, Complaint, and Resolution. Formula prefixes are neutralized for security.
                </p>

                <div className="form-group">
                  <label className="form-label">Select .xlsx File</label>
                  <input
                    type="file"
                    className="form-control"
                    accept=".xlsx"
                    onChange={e => setImportFile(e.target.files[0])}
                    required
                  />
                </div>
              </div>

              <div className="modal-footer">
                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  onClick={() => setShowImportModal(false)}
                  disabled={importing}
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={importing || !importFile}
                >
                  {importing ? 'Processing Workbook...' : 'Upload & Import'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}
