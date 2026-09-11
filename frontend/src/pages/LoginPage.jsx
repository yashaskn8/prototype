import React, { useState } from 'react';
import { useAuth } from '../AuthContext';
import { Shield, KeyRound, User, Lock, AlertCircle, ArrowRight } from 'lucide-react';

export function LoginPage() {
  const { login, error: authError } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [localError, setLocalError] = useState(null);

  const demoAccounts = [
    { label: 'Admin', role: 'Full System Superuser', u: 'admin', p: 'adminpass123' },
    { label: 'Manager', role: 'Service Operations', u: 'manager', p: 'managerpass123' },
    { label: 'Technician', role: 'Apex Services Tech', u: 'tech1', p: 'techpass123' },
    { label: 'Customer', role: 'Acme Diagnostics', u: 'acme_user', p: 'acmepass123' },
  ];

  async function handleSubmit(e) {
    if (e) e.preventDefault();
    if (!username || !password) {
      setLocalError('Please enter both username and password.');
      return;
    }
    setSubmitting(true);
    setLocalError(null);
    const res = await login(username, password);
    setSubmitting(false);
    if (!res.success) {
      setLocalError(res.error);
    }
  }

  function fillDemo(u, p) {
    setUsername(u);
    setPassword(p);
    setLocalError(null);
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-brand">
          <div className="brand-icon" style={{ width: '44px', height: '44px', fontSize: '1.4rem' }}>
            S
          </div>
          <div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
              Servy RAG
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Intelligent Enterprise Field Service
            </p>
          </div>
        </div>

        {(localError || authError) && (
          <div className="alert alert-danger">
            <AlertCircle size={18} style={{ flexShrink: 0 }} />
            <div>{localError || authError}</div>
          </div>
        )}

        <div style={{ marginBottom: '20px' }}>
          <label style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block', marginBottom: '8px' }}>
            Quick Demo Personas
          </label>
          <div className="persona-buttons">
            {demoAccounts.map(acc => (
              <button
                key={acc.u}
                type="button"
                className="persona-btn"
                onClick={() => fillDemo(acc.u, acc.p)}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <strong>{acc.label}</strong>
                  <ArrowRight size={12} color="#6366f1" />
                </div>
                <span>{acc.role}</span>
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="username">Username</label>
            <div style={{ position: 'relative' }}>
              <User size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '13px' }} />
              <input
                id="username"
                type="text"
                className="form-control"
                style={{ paddingLeft: '38px' }}
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="Enter username"
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password">Password</label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '13px' }} />
              <input
                id="password"
                type="password"
                className="form-control"
                style={{ paddingLeft: '38px' }}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Enter password"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', padding: '12px', marginTop: '10px' }}
            disabled={submitting}
          >
            {submitting ? 'Authenticating...' : 'Sign In to Workspace'}
          </button>
        </form>

        <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--border)', textAlign: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <Shield size={12} color="#10b981" /> Hardened RAG Security Boundary Enforced
          </span>
        </div>
      </div>
    </div>
  );
}
