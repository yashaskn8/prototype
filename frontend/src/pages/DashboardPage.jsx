import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../AuthContext';
import { 
  Package, 
  PhoneCall, 
  AlertCircle, 
  BookOpen, 
  Bot, 
  ArrowUpRight, 
  Clock, 
  CheckCircle2, 
  Building,
  TrendingUp,
  RefreshCw
} from 'lucide-react';

export function DashboardPage({ onNavigate }) {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function loadDashboard() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/dashboard/');
      setData(res);
    } catch (err) {
      setError(err.detail || 'Failed to load dashboard data.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
  }, [user]);

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <RefreshCw size={24} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
        <p>Loading operations metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-danger" style={{ margin: '20px' }}>
        <AlertCircle size={20} />
        <div>{error}</div>
      </div>
    );
  }

  const metrics = data?.metrics || {};
  const isCustomer = user?.role === 'customer';

  return (
    <div className="content-body">
      {/* Welcome Banner */}
      <div style={{ 
        background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)', 
        borderRadius: 'var(--radius-lg)', 
        padding: '28px 32px', 
        color: '#ffffff',
        marginBottom: '28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        boxShadow: 'var(--shadow-md)'
      }}>
        <div>
          <span style={{ 
            display: 'inline-flex', 
            alignItems: 'center', 
            gap: '6px', 
            background: 'rgba(255, 255, 255, 0.15)', 
            padding: '3px 10px', 
            borderRadius: '999px',
            fontSize: '0.75rem',
            fontWeight: 600,
            marginBottom: '10px'
          }}>
            <Building size={12} /> {data?.tenant?.name || 'Enterprise'}
          </span>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '4px' }}>
            Welcome back, {user?.username}
          </h2>
          <p style={{ color: '#c7d2fe', fontSize: '0.9rem' }}>
            {isCustomer 
              ? 'Self-service diagnostic portal with verified knowledge resolution.' 
              : 'Field service dispatch, technician copilot, and knowledge repository.'}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button 
            className="btn btn-primary"
            style={{ background: '#ffffff', color: '#312e81', fontWeight: 700 }}
            onClick={() => onNavigate(isCustomer ? 'customer-support' : 'engineer-copilot')}
          >
            <Bot size={16} /> {isCustomer ? 'Diagnose Issue' : 'AI Copilot'}
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-icon purple">
            <Package size={22} />
          </div>
          <div>
            <div className="kpi-value">{metrics.assets || 0}</div>
            <div className="kpi-label">Active Assets</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon blue">
            <PhoneCall size={22} />
          </div>
          <div>
            <div className="kpi-value">{metrics.service_calls || 0}</div>
            <div className="kpi-label">Service Calls</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon amber">
            <Clock size={22} />
          </div>
          <div>
            <div className="kpi-value">{metrics.open_calls || 0}</div>
            <div className="kpi-label">Open Tickets</div>
          </div>
        </div>

        {!isCustomer && (
          <div className="kpi-card">
            <div className="kpi-icon green">
              <BookOpen size={22} />
            </div>
            <div>
              <div className="kpi-value">{metrics.knowledge_documents || 0}</div>
              <div className="kpi-label">KB Documents</div>
            </div>
          </div>
        )}

        <div className="kpi-card">
          <div className="kpi-icon purple">
            <Bot size={22} />
          </div>
          <div>
            <div className="kpi-value">{metrics.interactions_today || 0}</div>
            <div className="kpi-label">AI Queries Today</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon red">
            <TrendingUp size={22} />
          </div>
          <div>
            <div className="kpi-value">{metrics.escalations_today || 0}</div>
            <div className="kpi-label">Escalated Tickets</div>
          </div>
        </div>
      </div>

      {/* Two-Column Grid: Recent Calls & Recent AI Interactions */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '24px' }}>
        
        {/* Recent Service Calls */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <PhoneCall size={18} color="var(--primary)" />
              Recent Service Calls
            </div>
            <button className="btn btn-secondary btn-sm" onClick={() => onNavigate('calls')}>
              View All <ArrowUpRight size={14} />
            </button>
          </div>

          <div className="table-responsive">
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Customer / Site</th>
                  <th>Status</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                {(!data?.recent_calls || data.recent_calls.length === 0) ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px' }}>
                      No service calls found.
                    </td>
                  </tr>
                ) : (
                  data.recent_calls.map(call => (
                    <tr key={call.id}>
                      <td style={{ fontWeight: 600 }}>{call.servy_id}</td>
                      <td>
                        <div style={{ fontWeight: 500 }}>{call.customer__name}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{call.site__name || 'Main Site'}</div>
                      </td>
                      <td>
                        <span className={`badge badge-${call.status}`}>
                          {call.status.replace('_', ' ')}
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${call.priority}`}>
                          {call.priority}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Recent AI Self-Service Sessions */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Bot size={18} color="var(--accent)" />
              Recent AI Self-Service Queries
            </div>
            <button className="btn btn-secondary btn-sm" onClick={() => onNavigate('customer-support')}>
              Portal <ArrowUpRight size={14} />
            </button>
          </div>

          <div className="table-responsive">
            <table className="table">
              <thead>
                <tr>
                  <th>Customer</th>
                  <th>Asset</th>
                  <th>Outcome</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {(!data?.recent_interactions || data.recent_interactions.length === 0) ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px' }}>
                      No AI interactions recorded yet.
                    </td>
                  </tr>
                ) : (
                  data.recent_interactions.map(item => (
                    <tr key={item.id}>
                      <td style={{ fontWeight: 500 }}>{item.customer__name}</td>
                      <td>{item.asset__name || 'General Equipment'}</td>
                      <td>
                        {item.escalated_call ? (
                          <span className="badge badge-high">Escalated</span>
                        ) : item.resolved_without_call ? (
                          <span className="badge badge-resolved">Resolved</span>
                        ) : (
                          <span className="badge badge-open">Consulted</span>
                        )}
                      </td>
                      <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        {new Date(item.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
