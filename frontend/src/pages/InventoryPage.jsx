import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Boxes, PackageCheck, AlertCircle, RefreshCw, Search } from 'lucide-react';

export function InventoryPage() {
  const [activeTab, setActiveTab] = useState('inventory');
  const [inventory, setInventory] = useState([]);
  const [partRequests, setPartRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      const [invRes, reqRes] = await Promise.all([
        api.get('/api/inventory/'),
        api.get('/api/part-requests/'),
      ]);
      setInventory(invRes.items || (Array.isArray(invRes) ? invRes : []));
      setPartRequests(reqRes.part_requests || (Array.isArray(reqRes) ? reqRes : []));
    } catch (err) {
      setError(err.detail || 'Failed to load inventory data.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const filteredInv = inventory.filter(i => {
    const name = i.name || i.spare_name || '';
    const sku = i.sku || i.ipn || '';
    return !search ||
      name.toLowerCase().includes(search.toLowerCase()) ||
      sku.toLowerCase().includes(search.toLowerCase());
  });

  const filteredReq = partRequests.filter(r =>
    !search ||
    (r.spare_description && r.spare_description.toLowerCase().includes(search.toLowerCase())) ||
    (r.indent_id && r.indent_id.toLowerCase().includes(search.toLowerCase())) ||
    (r.requester__full_name && r.requester__full_name.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="content-body">
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Spares Inventory & Part Indents
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Real-time stock monitoring, technician part indent requests, and store dispatch workflows.
        </p>
      </div>

      {error && (
        <div className="alert alert-danger">
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button
          className={`btn ${activeTab === 'inventory' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('inventory')}
        >
          <Boxes size={16} /> Warehouse Inventory ({inventory.length})
        </button>
        <button
          className={`btn ${activeTab === 'requests' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('requests')}
        >
          <PackageCheck size={16} /> Technician Part Requests ({partRequests.length})
        </button>
      </div>

      {/* Search & Refresh */}
      <div className="card" style={{ marginBottom: '20px', padding: '16px 20px' }}>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '12px' }} />
            <input
              type="text"
              className="form-control"
              style={{ paddingLeft: '38px' }}
              placeholder={activeTab === 'inventory' ? "Search parts by name or SKU..." : "Search indents by technician, part, or ID..."}
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <button className="btn btn-secondary" onClick={loadData} title="Refresh">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Content Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-responsive">
          {activeTab === 'inventory' ? (
            <table className="table">
              <thead>
                <tr>
                  <th>Part / Item Name</th>
                  <th>SKU Code</th>
                  <th>Branch / Store</th>
                  <th>Quantity in Stock</th>
                  <th>Stock Status</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      <RefreshCw size={20} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
                      <p>Loading inventory...</p>
                    </td>
                  </tr>
                ) : filteredInv.length === 0 ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      No inventory items found.
                    </td>
                  </tr>
                ) : (
                  filteredInv.map(item => {
                    const isLow = item.quantity <= (item.reorder_level || 5);
                    return (
                      <tr key={item.id}>
                        <td style={{ fontWeight: 600 }}>{item.name || item.spare_name}</td>
                        <td>
                          <span className="badge" style={{ background: '#f1f5f9', color: '#1e293b', fontFamily: 'var(--font-mono)' }}>
                            {item.sku || item.ipn}
                          </span>
                        </td>
                        <td>{item.branch__name || 'Central Warehouse'}</td>
                        <td style={{ fontWeight: 700 }}>{item.quantity}</td>
                        <td>
                          <span className={`badge ${isLow ? 'badge-high' : 'badge-resolved'}`}>
                            {isLow ? 'Low Stock' : 'In Stock'}
                          </span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Indent ID</th>
                  <th>Requested Spare</th>
                  <th>Technician</th>
                  <th>Linked Service Call</th>
                  <th>Manager Approval</th>
                  <th>Store Status</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      Loading part requests...
                    </td>
                  </tr>
                ) : filteredReq.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
                      No part requests recorded.
                    </td>
                  </tr>
                ) : (
                  filteredReq.map(req => (
                    <tr key={req.id}>
                      <td style={{ fontWeight: 700, color: 'var(--primary)' }}>{req.indent_id}</td>
                      <td style={{ fontWeight: 500 }}>{req.spare_description}</td>
                      <td>{req.requester__full_name || 'Field Technician'}</td>
                      <td>{req.service_call__servy_id || 'Direct Request'}</td>
                      <td>
                        <span className={`badge badge-${(req.manager_status || 'approved').toLowerCase()}`}>
                          {req.manager_status || 'Approved'}
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${(req.store_status || 'issued').toLowerCase()}`}>
                          {req.store_status || 'Issued'}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
