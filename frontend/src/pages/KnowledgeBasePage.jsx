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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');

  // Upload modal state
  const [showUpload, setShowUpload] = useState(false);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [isConfidential, setIsConfidential] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadNotice, setUploadNotice] = useState(null);

  async function loadDocs() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/knowledge/');
      setDocs(res.documents || []);
    } catch (err) {
      setError(err.detail || 'Failed to load knowledge base.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDocs();
  }, []);

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

  const filteredDocs = docs.filter(d => 
    !search || 
    d.title.toLowerCase().includes(search.toLowerCase()) ||
    (d.description && d.description.toLowerCase().includes(search.toLowerCase())) ||
    (d.tags && d.tags.toLowerCase().includes(search.toLowerCase()))
  );

  const canUpload = ['admin', 'manager', 'technician', 'superuser'].includes(user?.role);
  const canDelete = ['admin', 'manager', 'superuser'].includes(user?.role);

  return (
    <div className="content-body">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
            Technical Knowledge Base & Manuals
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            Multi-tier vector-indexed documents, wiring schematics, and confidential service bulletins.
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
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: 1 }}>
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
          <button className="btn btn-secondary" onClick={loadDocs} title="Refresh documents">
            <RefreshCw size={14} />
          </button>
        </div>
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
              ) : filteredDocs.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    No knowledge documents found matching the criteria.
                  </td>
                </tr>
              ) : (
                filteredDocs.map(doc => (
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
