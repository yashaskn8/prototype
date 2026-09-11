import React, { useState } from 'react';
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

function MainLayout() {
  const { user, loading } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [copilotCallId, setCopilotCallId] = useState(null);

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

  function handleOpenCopilotForCall(callId) {
    setCopilotCallId(callId);
    setActiveTab('engineer-copilot');
  }

  const titles = {
    'dashboard': 'Operational Dashboard',
    'customer-support': 'Workflow 1: Customer AI Self-Service',
    'engineer-copilot': 'Workflow 2: Engineer Field Copilot',
    'calls': 'Service Call Register',
    'knowledge': 'Knowledge Base & Schematics',
    'assets': 'Equipment & Assets',
    'projects': 'Service Contracts & Projects',
    'inventory': 'Inventory & Part Requests',
    'staff': 'Team & Dispatch Directory',
  };

  return (
    <div className="app-container">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="main-area">
        <header className="top-bar">
          <div className="top-bar-title">
            <h2>{titles[activeTab] || 'Servy RAG'}</h2>
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
          {activeTab === 'dashboard' && <DashboardPage onNavigate={setActiveTab} />}
          {activeTab === 'customer-support' && <CustomerSupportPage onNavigate={setActiveTab} />}
          {activeTab === 'engineer-copilot' && <EngineerCopilotPage initialCallId={copilotCallId} />}
          {activeTab === 'calls' && <CallRegisterPage onOpenCopilot={handleOpenCopilotForCall} />}
          {activeTab === 'knowledge' && <KnowledgeBasePage />}
          {activeTab === 'assets' && <AssetsPage />}
          {activeTab === 'projects' && <ProjectsPage />}
          {activeTab === 'inventory' && <InventoryPage />}
          {activeTab === 'staff' && <StaffPage />}
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainLayout />
    </AuthProvider>
  );
}
