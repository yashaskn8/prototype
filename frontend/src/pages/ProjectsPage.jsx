import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Briefcase, Search, RefreshCw, AlertCircle } from 'lucide-react';

export function ProjectsPage() {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');

  async function loadProjects() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/projects/');
      setProjects(res.projects || []);
    } catch (err) {
      setError(err.detail || 'Failed to load projects.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadProjects();
  }, []);

  const filtered = projects.filter(p =>
    !search ||
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    (p.code && p.code.toLowerCase().includes(search.toLowerCase())) ||
    (p.customer__name && p.customer__name.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="content-body">
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Service Projects & Contracts
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Maintenance service agreements, multi-site deployments, and warranty scopes.
        </p>
      </div>

      {error && (
        <div className="alert alert-danger">
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      <div className="card" style={{ marginBottom: '20px', padding: '16px 20px' }}>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '12px' }} />
            <input
              type="text"
              className="form-control"
              style={{ paddingLeft: '38px' }}
              placeholder="Search by project name, code, or client..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <button className="btn btn-secondary" onClick={loadProjects} title="Refresh">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-responsive">
          <table className="table">
            <thead>
              <tr>
                <th>Project Code</th>
                <th>Project Name</th>
                <th>Client / Customer</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                    <p>Loading projects...</p>
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    No projects found matching the criteria.
                  </td>
                </tr>
              ) : (
                filtered.map(p => (
                  <tr key={p.id}>
                    <td>
                      <span className="badge" style={{ background: '#e0e7ff', color: '#4338ca', fontFamily: 'var(--font-mono)' }}>
                        {p.code}
                      </span>
                    </td>
                    <td>
                      <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Briefcase size={16} color="var(--primary)" />
                        {p.name}
                      </div>
                    </td>
                    <td>
                      <div style={{ fontWeight: 500 }}>{p.customer__name}</div>
                    </td>
                    <td>
                      <span className={`badge badge-${(p.status || 'in_progress').toLowerCase()}`}>
                        {p.status || 'Active'}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
