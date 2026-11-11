import React, { useEffect, useState } from 'react';
import { AuthProvider, useAuth } from './AuthContext';
import { Navigation } from './components/Navigation';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { CustomerSupportPage } from './pages/CustomerSupportPage';
import { EngineerCopilotPage } from './pages/EngineerCopilotPage';
import { CallRegisterPage } from './pages/CallRegisterPage';
import { KnowledgeBasePage } from './pages/KnowledgeBasePage';
import { AssetsPage } from './pages/AssetsPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { InventoryPage } from './pages/InventoryPage';
import { StaffPage } from './pages/StaffPage';
import { ShieldCheck, RefreshCw } from 'lucide-react';
import { RealDemoPage } from './pages/RealDemoPage';
import { TechnicianInboxPage } from './pages/TechnicianInboxPage';

function isTechnicianInboxLocation() {
  return window.location.pathname === '/technician/inbox' || window.location.hash === '#/technician/inbox';
}

function MainLayout() {
  const { user, loading } = useAuth();
  const [activeTab, setActiveTab] = useState(() => isTechnicianInboxLocation() ? 'technician-inbox' : 'dashboard');
  const [copilotCallId, setCopilotCallId] = useState(null);

  const inboxPath = import.meta.env.DEV ? '/technician/inbox' : '/app/#/technician/inbox';
  const appRootPath = import.meta.env.DEV ? '/' : '/app/';

  useEffect(() => {
    function handlePopState() {
      setActiveTab(isTechnicianInboxLocation() ? 'technician-inbox' : 'dashboard');
    }
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  useEffect(() => {
    if (user?.role === 'customer' && activeTab === 'technician-inbox') {
      window.history.replaceState({}, '', appRootPath);
    }
  }, [user?.role, activeTab, appRootPath]);

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-app)' }}>
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          <div className="brand-icon" style={{ margin: '0 auto 16px', width: '48px', height: '48px' }}>S</div>
          <RefreshCw size={24} className="spin" style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
          <p style={{ fontWeight: 600 }}>Initializing Servy RAG Security Boundary...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <LoginPage />;
  }

  function navigateToTab(tab) {
    setActiveTab(tab);
    const targetPath = tab === 'technician-inbox' ? inboxPath : appRootPath;
    if (window.location.pathname !== targetPath || window.location.hash) window.history.pushState({ servyTab: tab }, '', targetPath);
  }

  function handleOpenCopilotForCall(callId) {
    setCopilotCallId(callId);
    navigateToTab('engineer-copilot');
  }

  const currentTab = user.role === 'customer' && activeTab === 'technician-inbox'
    ? 'customer-support'
    : activeTab;

  const titles = {
    'dashboard': 'Operational Dashboard',
    'customer-support': 'Workflow 1: Customer AI Self-Service',
    'engineer-copilot': 'Workflow 2: Engineer Field Copilot',
    'calls': 'Service Call Register',
    'technician-inbox': 'Technician Incoming Tickets',
    'knowledge': 'Knowledge Base & Schematics',
    'assets': 'Equipment & Assets',
    'projects': 'Service Contracts & Projects',
    'inventory': 'Inventory & Part Requests',
    'staff': 'Team & Dispatch Directory',
  };

  return (
    <div className="app-container">
      <Navigation activeTab={currentTab} setActiveTab={navigateToTab} />
      
      <main className="main-area">
        <header className="top-bar">
          <div className="top-bar-title">
            <h2>{titles[currentTab] || 'Servy RAG'}</h2>
            <p>
              Tenant: <strong>{user.tenant_name || 'Standard'}</strong> | Role: <strong style={{ textTransform: 'capitalize' }}>{user.role}</strong>
            </p>
          </div>

          <div className="top-bar-actions">
            <span style={{ 
              display: 'inline-flex', 
              alignItems: 'center', 
              gap: '6px', 
              fontSize: '0.78rem', 
              color: '#059669', 
              background: '#ecfdf5', 
              padding: '6px 12px', 
              borderRadius: '999px',
              border: '1px solid #a7f3d0',
              fontWeight: 600
            }}>
              <ShieldCheck size={14} /> Zero-Cost Local RAG Active
            </span>
          </div>
        </header>

        <div>
          {currentTab === 'dashboard' && <DashboardPage onNavigate={navigateToTab} />}
          {currentTab === 'customer-support' && <CustomerSupportPage onNavigate={navigateToTab} />}
          {currentTab === 'technician-inbox' && <TechnicianInboxPage />}
          {currentTab === 'engineer-copilot' && <EngineerCopilotPage initialCallId={copilotCallId} />}
          {currentTab === 'calls' && <CallRegisterPage onOpenCopilot={handleOpenCopilotForCall} />}
          {currentTab === 'knowledge' && <KnowledgeBasePage />}
          {currentTab === 'assets' && <AssetsPage />}
          {currentTab === 'projects' && <ProjectsPage />}
          {currentTab === 'inventory' && <InventoryPage />}
          {currentTab === 'staff' && <StaffPage />}
        </div>
      </main>
    </div>
  );
}

export default function App() {
  if (import.meta.env.VITE_SERVY_REAL_DEMO === '1') return <RealDemoPage />;
  return (
    <AuthProvider>
      <MainLayout />
    </AuthProvider>
  );
}
