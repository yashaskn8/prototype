import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Users, Search, RefreshCw, AlertCircle, Shield, Phone, Mail, MapPin } from 'lucide-react';

export function StaffPage() {
  const [staff, setStaff] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');

  async function loadStaff() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/users/');
      setStaff(res || []);
    } catch (err) {
      setError(err.detail || 'Failed to load staff directory.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadStaff();
  }, []);

  const filtered = staff.filter(s =>
    !search ||
    s.full_name.toLowerCase().includes(search.toLowerCase()) ||
    (s.role && s.role.toLowerCase().includes(search.toLowerCase())) ||
    (s.branch__name && s.branch__name.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="content-body">
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Field Service Team Directory
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Authorized engineers, dispatch coordinators, branch managers, and warehouse supervisors.
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
              placeholder="Search staff by name, role, or branch..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <button className="btn btn-secondary" onClick={loadStaff} title="Refresh">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-responsive">
          <table className="table">
            <thead>
              <tr>
                <th>Member Name</th>
                <th>Role</th>
                <th>Contact Details</th>
                <th>Branch & Operational Zone</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                    <p>Loading staff directory...</p>
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    No staff members found matching the search.
                  </td>
                </tr>
              ) : (
                filtered.map(s => (
                  <tr key={s.id}>
                    <td>
                      <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div className="avatar" style={{ width: '30px', height: '30px', fontSize: '0.75rem' }}>
                          {s.full_name.substring(0, 2).toUpperCase()}
                        </div>
                        {s.full_name}
                      </div>
                    </td>
                    <td>
                      <span className="badge" style={{ background: '#e0e7ff', color: '#4338ca', textTransform: 'capitalize' }}>
                        {s.role.replace('_', ' ')}
                      </span>
                    </td>
                    <td>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Mail size={12} color="#64748b" /> {s.email || 'internal@servy.local'}
                      </div>
                      {s.phone && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Phone size={12} color="#64748b" /> {s.phone}
                        </div>
                      )}
                    </td>
                    <td>
                      <div style={{ fontWeight: 500 }}>{s.branch__name || 'Headquarters'}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{s.zone__name || 'Standard Zone'}</div>
                    </td>
                    <td>
                      <span className={`badge ${s.is_active ? 'badge-resolved' : 'badge-closed'}`}>
                        {s.is_active ? 'Active' : 'Inactive'}
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
