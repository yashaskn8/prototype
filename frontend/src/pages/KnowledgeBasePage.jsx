import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../AuthContext';
import { 
  Search, 
  Upload, 
  Download, 
  Trash2, 
  Lock, 
  CheckCircle, 
  AlertTriangle, 
  RefreshCw, 
  FileText, 
  Link as LinkIcon,
  Video,
  FileCode,
  X,
  Plus,
  Bot,
  ExternalLink
} from 'lucide-react';

export function KnowledgeBasePage() {
  const { user } = useAuth();
  const [docs, setDocs] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [docTypeFilter, setDocTypeFilter] = useState('');
  const [sourceTypeFilter, setSourceTypeFilter] = useState('');
  const [indexStatusFilter, setIndexStatusFilter] = useState('');
  const [ragFilter, setRagFilter] = useState('');
  const [confidentialFilter, setConfidentialFilter] = useState('');

  // Upload modal state
  const [showUpload, setShowUpload] = useState(false);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [sourceType, setSourceType] = useState('file');
  const [sourceUrl, setSourceUrl] = useState('');
  const [contentText, setContentText] = useState('');
  const [tags, setTags] = useState('');
  const [isConfidential, setIsConfidential] = useState(false);
  const [disableSharing, setDisableSharing] = useState(false);
  const [isRagEnabled, setIsRagEnabled] = useState(true);
  const [docType, setDocType] = useState('reference');
  
  // Dynamic taxonomy selections
  const [selectedCustomer, setSelectedCustomer] = useState('');
  const [selectedDomain, setSelectedDomain] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedProduct, setSelectedProduct] = useState('');
  const [selectedAsset, setSelectedAsset] = useState('');
  const [metadataOptions, setMetadataOptions] = useState(null);

  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadNotice, setUploadNotice] = useState(null);

  const isStaff = ['admin', 'manager', 'technician', 'superuser'].includes(user?.role);
  const canUpload = ['admin', 'manager', 'technician', 'superuser'].includes(user?.role);
  const canDelete = ['admin', 'manager', 'superuser'].includes(user?.role);

  async function loadDocs() {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (search.trim()) params.append('q', search.trim());
      if (docTypeFilter) params.append('doc_type', docTypeFilter);
      if (sourceTypeFilter) params.append('source_type', sourceTypeFilter);
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

  async function loadMetadataOptions() {
    try {
      const res = await api.get('/api/knowledge/metadata-options/');
      setMetadataOptions(res);
    } catch (err) {
      console.warn('Metadata options not loaded:', err);
    }
  }

  useEffect(() => {
    loadDocs();
  }, [docTypeFilter, sourceTypeFilter, indexStatusFilter, ragFilter, confidentialFilter]);

  useEffect(() => {
    if (canUpload) {
      loadMetadataOptions();
    }
  }, [canUpload]);

  async function handleUpload(e) {
    e.preventDefault();
    if (!title) return;
    if (sourceType === 'file' && !file) {
      alert('Please select a file to upload.');
      return;
    }
    if ((sourceType === 'link' || sourceType === 'video') && !sourceUrl) {
      alert('Please provide the URL/link.');
      return;
    }
    if (sourceType === 'text' && !contentText) {
      alert('Please enter text content.');
      return;
    }

    setUploading(true);
    setUploadNotice(null);

    const formData = new FormData();
    formData.append('title', title.trim());
    formData.append('description', description.trim());
    formData.append('source_type', sourceType);
    formData.append('source_url', sourceUrl.trim());
    formData.append('content_text', contentText.trim());
    formData.append('tags', tags.trim());
    formData.append('is_confidential', isConfidential ? 'true' : 'false');
    formData.append('disable_sharing', disableSharing ? 'true' : 'false');
    formData.append('is_rag_enabled', isRagEnabled ? 'true' : 'false');
    formData.append('doc_type', docType);

    if (selectedCustomer) formData.append('customer_id', selectedCustomer);
    if (selectedDomain) formData.append('domain_id', selectedDomain);
    if (selectedCategory) formData.append('category_id', selectedCategory);
    if (selectedProduct) formData.append('product_id', selectedProduct);
    if (selectedAsset) formData.append('asset_id', selectedAsset);

    if (file) {
      formData.append('file', file);
    }

    try {
      const res = await api.upload('/api/knowledge/upload/', formData);
      setUploadNotice({ 
        type: 'success', 
        msg: res.message || `Document saved. Index status: ${res.index_status || 'NOT_INDEXED'}` 
      });
      setTitle('');
      setDescription('');
      setSourceUrl('');
      setContentText('');
      setTags('');
      setIsConfidential(false);
      setDisableSharing(false);
      setFile(null);
      setSelectedCustomer('');
      setSelectedDomain('');
      setSelectedCategory('');
      setSelectedProduct('');
      setSelectedAsset('');
      loadDocs();
      setTimeout(() => setShowUpload(false), 2000);
    } catch (err) {
      setUploadNotice({ type: 'danger', msg: err.detail || 'Upload rejected by validation.' });
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

  function getSourceIcon(type) {
    switch (type) {
      case 'video': return <Video size={15} color="#e11d48" />;
      case 'link': return <LinkIcon size={15} color="#0284c7" />;
      case 'text': return <FileCode size={15} color="#16a34a" />;
      default: return <FileText size={15} color="var(--primary)" />;
    }
  }

  return (
    <div className="content-body">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
              Organisation Knowledge Base & Content
            </h2>
            <span className="badge" style={{ background: '#e0e7ff', color: '#4338ca', fontWeight: 700, fontSize: '0.82rem', padding: '4px 10px' }}>
              {totalCount} {totalCount === 1 ? 'item' : 'items'}
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginTop: '4px' }}>
            Organisation-wide content repository: technical manuals, SOPs, documents, media, and hyperlinks.
          </p>
        </div>

        {canUpload && (
          <button className="btn btn-primary" onClick={() => setShowUpload(true)}>
            <Plus size={16} /> Add Content / Manual
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
          <div style={{ position: 'relative', flex: 1, minWidth: '200px' }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '12px' }} />
            <input
              type="text"
              className="form-control"
              style={{ paddingLeft: '36px' }}
              placeholder="Search by title, tags, or description..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '130px' }}
            value={sourceTypeFilter}
            onChange={e => setSourceTypeFilter(e.target.value)}
          >
            <option value="">All Content Types</option>
            <option value="file">Files / Documents</option>
            <option value="text">Text Articles</option>
            <option value="link">Hyperlinks</option>
            <option value="video">Videos</option>
          </select>

          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '130px' }}
            value={docTypeFilter}
            onChange={e => setDocTypeFilter(e.target.value)}
          >
            <option value="">All Doc Types</option>
            <option value="installation">Installation Guide</option>
            <option value="troubleshooting">Troubleshooting</option>
            <option value="maintenance">Maintenance</option>
            <option value="cleaning">Cleaning</option>
            <option value="user_guide">User Guide</option>
            <option value="service_manual">Service Manual</option>
            <option value="faq">FAQ</option>
            <option value="reference">Reference</option>
          </select>

          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '120px' }}
            value={ragFilter}
            onChange={e => setRagFilter(e.target.value)}
          >
            <option value="">AI / RAG</option>
            <option value="true">RAG Enabled</option>
            <option value="false">Browsing Only</option>
          </select>

          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '130px' }}
            value={indexStatusFilter}
            onChange={e => setIndexStatusFilter(e.target.value)}
          >
            <option value="">All Statuses</option>
            <option value="INDEXED">Indexed</option>
            <option value="NOT_INDEXED">Not Indexed</option>
            <option value="INDEXING">Indexing</option>
            <option value="FAILED">Failed</option>
          </select>

          {isStaff && (
            <select
              className="form-control"
              style={{ width: 'auto', minWidth: '130px' }}
              value={confidentialFilter}
              onChange={e => setConfidentialFilter(e.target.value)}
            >
              <option value="">All Access</option>
              <option value="false">Public</option>
              <option value="true">Confidential</option>
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
                <th>Title / Description</th>
                <th>Content Type</th>
                <th>Classification</th>
                <th>Scope / Equipment</th>
                <th>AI / RAG Eligibility</th>
                <th>Vector Status</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                    <p>Loading knowledge library...</p>
                  </td>
                </tr>
              ) : docs.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    No knowledge items found matching the criteria.
                  </td>
                </tr>
              ) : (
                docs.map(doc => (
                  <tr key={doc.id}>
                    <td>
                      <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {getSourceIcon(doc.source_type)}
                        {doc.title}
                      </div>
                      {doc.description && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px', maxWidth: '320px' }}>
                          {doc.description}
                        </div>
                      )}
                      {doc.tags && (
                        <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
                          Tags: {doc.tags}
                        </div>
                      )}
                    </td>
                    <td>
                      <span className="badge" style={{ textTransform: 'capitalize', background: '#f8fafc', color: '#334155' }}>
                        {doc.source_type || 'file'}
                      </span>
                    </td>
                    <td>
                      {doc.is_confidential ? (
                        <span className="badge badge-confidential">
                          <Lock size={11} /> Confidential
                        </span>
                      ) : (
                        <span className="badge" style={{ background: '#f1f5f9', color: '#475569' }}>
                          Public
                        </span>
                      )}
                      {doc.disable_sharing && (
                        <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>No Sharing</div>
                      )}
                    </td>
                    <td>
                      <div style={{ fontWeight: 500, fontSize: '0.85rem' }}>
                        {doc.product__name || doc.asset__name || 'General Fleet'}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                        {doc.doc_type ? doc.doc_type.replace('_', ' ') : 'Reference'}
                      </div>
                    </td>
                    <td>
                      {doc.is_rag_enabled ? (
                        <span className="badge" style={{ background: '#ecfdf5', color: '#047857', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                          <Bot size={12} /> AI Active
                        </span>
                      ) : (
                        <span className="badge" style={{ background: '#f1f5f9', color: '#64748b' }}>
                          Browsing Only
                        </span>
                      )}
                    </td>
                    <td>
                      <span className={`badge badge-${(doc.index_status || 'indexed').toLowerCase()}`}>
                        {doc.index_status || 'NOT_INDEXED'}
                      </span>
                      {doc.chunks_count > 0 && (
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                          {doc.chunks_count} chunks
                        </div>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end' }}>
                        {doc.has_file && (
                          <a
                            href={`/api/knowledge/${doc.id}/download/`}
                            className="btn btn-secondary btn-sm"
                            title="Download authenticated attachment"
                          >
                            <Download size={13} />
                          </a>
                        )}
                        {doc.source_url && (
                          <a
                            href={doc.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn btn-secondary btn-sm"
                            title="Open external link"
                          >
                            <ExternalLink size={13} />
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

      {/* Upload & Add Modal */}
      {showUpload && (
        <div className="modal-backdrop">
          <div className="modal-content" style={{ maxWidth: '640px', maxHeight: '90vh', overflowY: 'auto' }}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Upload size={20} color="var(--primary)" />
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800 }}>Add Knowledge Base Content</h3>
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
                  <label className="form-label">Name / Title *</label>
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
                  <label className="form-label">Description / Summary</label>
                  <textarea
                    className="form-control"
                    rows={2}
                    placeholder="Brief description of this content..."
                    value={description}
                    onChange={e => setDescription(e.target.value)}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div className="form-group">
                    <label className="form-label">Content Type</label>
                    <select
                      className="form-control"
                      value={sourceType}
                      onChange={e => setSourceType(e.target.value)}
                    >
                      <option value="file">File Upload (.pdf, .docx, .txt, .md, .csv)</option>
                      <option value="link">Hyperlink / Web Reference</option>
                      <option value="video">Video URL</option>
                      <option value="text">Raw Text Article</option>
                    </select>
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
                </div>

                {sourceType === 'file' && (
                  <div className="form-group">
                    <label className="form-label">Upload File (.pdf, .docx, .txt, .md, .csv) *</label>
                    <input
                      type="file"
                      className="form-control"
                      accept=".pdf,.docx,.txt,.md,.csv"
                      onChange={e => setFile(e.target.files[0])}
                      required
                    />
                  </div>
                )}

                {(sourceType === 'link' || sourceType === 'video') && (
                  <div className="form-group">
                    <label className="form-label">Hyperlink / Media URL *</label>
                    <input
                      type="url"
                      className="form-control"
                      placeholder="https://..."
                      value={sourceUrl}
                      onChange={e => setSourceUrl(e.target.value)}
                      required
                    />
                  </div>
                )}

                {sourceType === 'text' && (
                  <div className="form-group">
                    <label className="form-label">Article Text Content *</label>
                    <textarea
                      className="form-control"
                      rows={4}
                      placeholder="Enter the full procedure or guide text..."
                      value={contentText}
                      onChange={e => setContentText(e.target.value)}
                      required
                    />
                  </div>
                )}

                <div style={{ display: 'flex', gap: '20px', marginTop: '10px', marginBottom: '16px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={isConfidential}
                      onChange={e => setIsConfidential(e.target.checked)}
                    />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Confidential (Staff Only)</span>
                  </label>

                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={disableSharing}
                      onChange={e => setDisableSharing(e.target.checked)}
                    />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Disable Sharing</span>
                  </label>
                </div>

                {/* Technical RAG Metadata (Admin Control Layer) */}
                <div style={{ 
                  background: 'var(--bg-subtle, #f8fafc)', 
                  border: '1px solid var(--border-color, #e2e8f0)', 
                  borderRadius: '8px', 
                  padding: '14px',
                  marginTop: '12px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                    <div style={{ fontSize: '0.88rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Bot size={16} color="var(--primary)" /> Technical AI & RAG Classification
                    </div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={isRagEnabled}
                        onChange={e => setIsRagEnabled(e.target.checked)}
                      />
                      <span style={{ fontSize: '0.82rem', fontWeight: 700, color: isRagEnabled ? '#059669' : '#64748b' }}>
                        Use for AI / RAG
                      </span>
                    </label>
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
                    Enable indexing for AI self-service diagnostics and engineer copilot. Scope to specific equipment or leave blank for fleet-wide manuals.
                  </p>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.78rem' }}>Technical Doc Type</label>
                      <select
                        className="form-control"
                        value={docType}
                        onChange={e => setDocType(e.target.value)}
                      >
                        <option value="reference">Reference</option>
                        <option value="installation">Installation Guide</option>
                        <option value="troubleshooting">Troubleshooting Manual</option>
                        <option value="maintenance">Maintenance Procedure</option>
                        <option value="cleaning">Cleaning Procedure</option>
                        <option value="user_guide">User Guide</option>
                        <option value="service_manual">Service Manual</option>
                        <option value="faq">FAQ</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.78rem' }}>Customer Scope (Optional)</label>
                      <select
                        className="form-control"
                        value={selectedCustomer}
                        onChange={e => setSelectedCustomer(e.target.value)}
                      >
                        <option value="">All Customers (Fleet-wide)</option>
                        {metadataOptions?.customers?.map(c => (
                          <option key={c.id} value={c.id}>{c.name}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.78rem' }}>Product Domain (Optional)</label>
                      <select
                        className="form-control"
                        value={selectedDomain}
                        onChange={e => setSelectedDomain(e.target.value)}
                      >
                        <option value="">All Domains</option>
                        {metadataOptions?.domains?.map(d => (
                          <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.78rem' }}>Product Category (Optional)</label>
                      <select
                        className="form-control"
                        value={selectedCategory}
                        onChange={e => setSelectedCategory(e.target.value)}
                      >
                        <option value="">All Categories</option>
                        {metadataOptions?.categories?.map(c => (
                          <option key={c.id} value={c.id}>{c.name}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.78rem' }}>Product (Optional)</label>
                      <select
                        className="form-control"
                        value={selectedProduct}
                        onChange={e => setSelectedProduct(e.target.value)}
                      >
                        <option value="">All Products</option>
                        {metadataOptions?.products?.map(p => (
                          <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.78rem' }}>Specific Asset (Optional)</label>
                      <select
                        className="form-control"
                        value={selectedAsset}
                        onChange={e => setSelectedAsset(e.target.value)}
                      >
                        <option value="">General (No Specific Physical Asset)</option>
                        {metadataOptions?.assets?.map(a => (
                          <option key={a.id} value={a.id}>{a.name} ({a.asset_code})</option>
                        ))}
                      </select>
                    </div>
                  </div>
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
                  disabled={uploading || !title}
                >
                  {uploading ? 'Processing & Indexing...' : (isRagEnabled ? 'Save & Sync to RAG' : 'Save Content')}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}
