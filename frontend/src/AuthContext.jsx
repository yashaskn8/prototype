import React, { createContext, useContext, useState, useEffect } from 'react';
import { api, bootstrapCsrf } from './api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [availableTenants, setAvailableTenants] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function checkAuth() {
    try {
      setLoading(true);
      await bootstrapCsrf();
      const data = await api.get('/api/auth/me/');
      setUser(data.user);
      setAvailableTenants(data.available_tenants || []);
      setError(null);
    } catch (err) {
      setUser(null);
      setAvailableTenants([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    checkAuth();
  }, []);

  async function login(username, password) {
    setError(null);
    try {
      const data = await api.post('/api/auth/login/', { username, password });
      setUser(data.user);
      // Refresh context
      await checkAuth();
      return { success: true };
    } catch (err) {
      const msg = err.detail || 'Invalid username or password';
      setError(msg);
      return { success: false, error: msg };
    }
  }

  async function logout() {
    try {
      await api.post('/api/auth/logout/', {});
    } catch (err) {
      console.error('Logout error', err);
    } finally {
      setUser(null);
      setAvailableTenants([]);
    }
  }

  async function switchTenant(tenantId) {
    try {
      await api.post('/api/auth/switch-tenant/', { tenant_id: tenantId });
      await checkAuth();
      return { success: true };
    } catch (err) {
      return { success: false, error: err.detail || 'Could not switch tenant' };
    }
  }

  return (
    <AuthContext.Provider value={{ user, availableTenants, loading, error, login, logout, switchTenant, refreshUser: checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
