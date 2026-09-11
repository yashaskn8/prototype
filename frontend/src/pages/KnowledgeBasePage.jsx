import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../AuthContext';
import { 
  BookOpen, 
  Search, 
  Upload, 
  Download, 
  Trash2, 
  Lock, 
  CheckCircle, 
  AlertTriangle, 
  ShieldAlert, 
  RefreshCw, 
  FileText, 
  X,
  Plus
} from 'lucide-react';

export function KnowledgeBasePage() {
  const { user } = useAuth();
  const [docs, setDocs] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [docTypeFilter, setDocTypeFilter] = useState('');
  const [indexStatusFilter, setIndexStatusFilter] = useState('');
  const [ragFilter, setRagFilter] = useState('');
  const [confidentialFilter, setConfidentialFilter] = useState('');

  // Upload modal state
  const [showUpload, setShowUpload] = useState(false);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [isConfidential, setIsConfidential] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadNotice, setUploadNotice] = useState(null);

  const isStaff = ['admin', 'manager', 'technician', 'superuser'].includes(user?.role);

  async function loadDocs() {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (search.trim()) params.append('q', search.trim());
      if (docTypeFilter) params.append('doc_type', docTypeFilter);
      if (indexStatusFilter) params.append('index_status', indexStatusFilter);
      if (ragFilter) params.append('is_rag_enabled', ragFilter);
      if (confidentialFilter) params.append('is_confidential', confidentialFilter);

      const qs = params.toString() ? `?${params.toString()}` : '';
      const res = await api.get(`/api/knowledge/${qs}`);
      setDocs(res.documents || []);
      setTotalCount(res.count !== undefined ? res.count : (res.documents || []).length);
    } catch (err) {
      setError(err.detail || 'Failed to load knowledge base.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDocs();
  }, [docTypeFilter, indexStatusFilter, ragFilter, confidentialFilter]);

  async function handleUpload(e) {
    e.preventDefault();
    if (!title || !file) return;

    setUploading(true);
    setUploadNotice(null);

    const formData = new FormData();
    formData.append('title', title);
    formData.append('description', description);
    formData.append('tags', tags);
    formData.append('is_confidential', isConfidential ? 'true' : 'false');
    formData.append('file', file);

    try {
      const res = await api.upload('/api/knowledge/upload/', formData);
      setUploadNotice({ type: 'success', msg: `Document uploaded. Status: ${res.index_status || 'INDEXED'}` });
      setTitle('');
      setDescription('');
      setTags('');
      setIsConfidential(false);
      setFile(null);
      loadDocs();
      setTimeout(() => setShowUpload(false), 2000);
    } catch (err) {
      setUploadNotice({ type: 'danger', msg: err.detail || 'Upload rejected by security validation.' });
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(docId, docTitle) {
    if (!window.confirm(`Are you sure you want to delete "${docTitle}" and purge its vector index?`)) return;

    try {
      await api.delete(`/api/knowledge/${docId}/delete/`);
      loadDocs();
    } catch (err) {
      alert(err.detail || 'Failed to delete document.');
    }
  }

  const canUpload = ['admin', 'manager', 'technician', 'superuser'].includes(user?.role);
  const canDelete = ['admin', 'manager', 'superuser'].includes(user?.role);

  return (
    <div className="content-body">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
              Technical Knowledge Base & Manuals
            </h2>
            <span className="badge" style={{ background: '#e0e7ff', color: '#4338ca', fontWeight: 700, fontSize: '0.82rem', padding: '4px 10px' }}>
              {totalCount} {totalCount === 1 ? 'document' : 'documents'}
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginTop: '4px' }}>
            Multi-tier vector-indexed documents, technical manuals, and equipment operating guides.
          </p>
        </div>

        {canUpload && (
          <button className="btn btn-primary" onClick={() => setShowUpload(true)}>
            <Plus size={16} /> Upload New Document
          </button>
        )}
      </div>

      {error && (
        <div className="alert alert-danger">
          <AlertTriangle size={18} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      {/* Filter Bar */}
      <div className="card" style={{ marginBottom: '20px', padding: '16px 20px' }}>
        <form onSubmit={e => { e.preventDefault(); loadDocs(); }} style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ position: 'relative', flex: 1, minWidth: '220px' }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '12px' }} />
            <input
              type="text"
              className="form-control"
              style={{ paddingLeft: '38px' }}
              placeholder="Search manuals by title, description, or tag..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '130px' }}
            value={docTypeFilter}
            onChange={e => setDocTypeFilter(e.target.value)}
          >
            <option value="">All Doc Types</option>
            <option value="manual">Manuals</option>
            <option value="sop">SOPs</option>
            <option value="schematic">Schematics</option>
            <option value="bulletin">Bulletins</option>
            <option value="guide">User Guides</option>
          </select>

          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '130px' }}
            value={indexStatusFilter}
            onChange={e => setIndexStatusFilter(e.target.value)}
          >
            <option value="">All Index Statuses</option>
            <option value="INDEXED">Indexed</option>
            <option value="NOT_INDEXED">Not Indexed</option>
            <option value="INDEXING">Indexing</option>
            <option value="FAILED">Failed</option>
          </select>

          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '120px' }}
            value={ragFilter}
            onChange={e => setRagFilter(e.target.value)}
          >
            <option value="">RAG Status</option>
            <option value="true">RAG Enabled</option>
            <option value="false">RAG Disabled</option>
          </select>

          {isStaff && (
            <select
              className="form-control"
              style={{ width: 'auto', minWidth: '130px' }}
              value={confidentialFilter}
              onChange={e => setConfidentialFilter(e.target.value)}
            >
              <option value="">All Classifications</option>
              <option value="false">Public</option>
              <option value="true">Confidential Only</option>
            </select>
          )}

          <button type="submit" className="btn btn-secondary" title="Search / Refresh">
            <RefreshCw size={14} /> Search
          </button>
        </form>
      </div>

      {/* Documents Grid / Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-responsive">
          <table className="table">
            <thead>
              <tr>
                <th>Document Title</th>
                <th>Classification</th>
                <th>Product / Asset</th>
                <th>Vector Status</th>
                <th>Chunks</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                    <p>Loading technical library...</p>
                  </td>
                </tr>
              ) : docs.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    No knowledge documents found matching the criteria.
                  </td>
                </tr>
              ) : (
                docs.map(doc => (
                  <tr key={doc.id}>
                    <td>
                      <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FileText size={16} color="var(--primary)" />
                        {doc.title}
                      </div>
                      {doc.description && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px', maxWidth: '340px' }}>
                          {doc.description}
                        </div>
                      )}
                    </td>
                    <td>
                      {doc.is_confidential ? (
                        <span className="badge badge-confidential">
                          <Lock size={11} /> Confidential
                        </span>
                      ) : (
                        <span className="badge" style={{ background: '#f1f5f9', color: '#475569' }}>
                          Public Manual
                        </span>
                      )}
                    </td>
                    <td>
                      <div style={{ fontWeight: 500 }}>{doc.product__name || doc.asset__name || 'General Fleet'}</div>
                    </td>
                    <td>
                      <span className={`badge badge-${(doc.index_status || 'indexed').toLowerCase()}`}>
                        {doc.index_status || 'INDEXED'}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                        {doc.chunks_count || 0} chunks
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end' }}>
                        {doc.has_file && (
                          <a
                            href={`/api/knowledge/${doc.id}/download/`}
                            className="btn btn-secondary btn-sm"
                            title="Download authenticated attachment"
                          >
                            <Download size={13} /> Download
                          </a>
                        )}
                        {canDelete && (
                          <button
                            className="btn btn-danger btn-sm"
                            onClick={() => handleDelete(doc.id, doc.title)}
                            title="Purge document and vectors"
                          >
                            <Trash2 size={13} />
                          </button>
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

      {/* Upload Modal */}
      {showUpload && (
        <div className="modal-backdrop">
          <div className="modal-content">
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Upload size={20} color="var(--primary)" />
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800 }}>Upload Knowledge Manual</h3>
              </div>
              <button 
                onClick={() => setShowUpload(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8' }}
              >
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleUpload}>
              <div className="modal-body">
                {uploadNotice && (
                  <div className={`alert alert-${uploadNotice.type}`}>
                    {uploadNotice.type === 'success' ? <CheckCircle size={18} /> : <AlertTriangle size={18} />}
                    <div>{uploadNotice.msg}</div>
                  </div>
                )}

                <div className="form-group">
                  <label className="form-label">Document Title</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="e.g. BioAnalyzer 4000 Service Manual"
                    value={title}
                    onChange={e => setTitle(e.target.value)}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Summary / Description</label>
                  <textarea
                    className="form-control"
                    rows={2}
                    placeholder="Describe maintenance procedures covered in this manual..."
                    value={description}
                    onChange={e => setDescription(e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Tags (Comma-separated)</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="analyzer, calibration, optics"
                    value={tags}
                    onChange={e => setTags(e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={isConfidential}
                      onChange={e => setIsConfidential(e.target.checked)}
                    />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Mark as Confidential (Staff Only)</span>
                  </label>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginTop: '2px' }}>
                    Confidential documents are excluded from Customer self-service portals.
                  </span>
                </div>

                <div className="form-group">
                  <label className="form-label">Upload Document File (.pdf, .docx, .txt, .md)</label>
                  <input
                    type="file"
                    className="form-control"
                    accept=".pdf,.docx,.txt,.md"
                    onChange={e => setFile(e.target.files[0])}
                    required
                  />
                </div>
              </div>

              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowUpload(false)}
                  disabled={uploading}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={uploading || !title || !file}
                >
                  {uploading ? 'Validating & Indexing...' : 'Upload & Sync Vectors'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}
