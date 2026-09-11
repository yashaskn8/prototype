import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Package, Search, RefreshCw, AlertCircle, Building, Tag, QrCode } from 'lucide-react';

export function AssetsPage() {
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');

  async function loadAssets() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/assets/');
      setAssets(res.assets || []);
    } catch (err) {
      setError(err.detail || 'Failed to load assets.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAssets();
  }, []);

  const filteredAssets = assets.filter(a => 
    !search || 
    a.name.toLowerCase().includes(search.toLowerCase()) ||
    (a.asset_code && a.asset_code.toLowerCase().includes(search.toLowerCase())) ||
    (a.model_number && a.model_number.toLowerCase().includes(search.toLowerCase())) ||
    (a.customer__name && a.customer__name.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="content-body">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
            Equipment & Asset Registry
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            Track deployed customer machinery, serial tags, and operational health.
          </p>
        </div>
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
              placeholder="Search assets by name, code, model, or customer..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <button className="btn btn-secondary" onClick={loadAssets} title="Refresh assets">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-responsive">
          <table className="table">
            <thead>
              <tr>
                <th>Asset Name</th>
                <th>Asset Code</th>
                <th>Model / Product</th>
                <th>Customer & Site</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                    <p>Loading asset database...</p>
                  </td>
                </tr>
              ) : filteredAssets.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                    No assets found matching the search criteria.
                  </td>
                </tr>
              ) : (
                filteredAssets.map(a => (
                  <tr key={a.id}>
                    <td>
                      <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Package size={16} color="var(--primary)" />
                        {a.name}
                      </div>
                      {a.serial_number && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          S/N: {a.serial_number}
                        </div>
                      )}
                    </td>
                    <td>
                      <span className="badge" style={{ background: '#f1f5f9', color: '#1e293b', fontFamily: 'var(--font-mono)' }}>
                        {a.asset_code || 'N/A'}
                      </span>
                    </td>
                    <td>
                      <div style={{ fontWeight: 500 }}>{a.model_number || 'Standard'}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{a.product__name || 'General Product'}</div>
                    </td>
                    <td>
                      <div style={{ fontWeight: 600 }}>{a.customer__name}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{a.site__name || 'Main Location'}</div>
                    </td>
                    <td>
                      <span className="badge badge-resolved">
                        {a.status || 'Active'}
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
