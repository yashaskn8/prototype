import React from 'react';
import { useAuth } from '../AuthContext';
import { 
  LayoutDashboard, 
  Bot, 
  Wrench, 
  PhoneCall, 
  BookOpen, 
  Package, 
  Briefcase, 
  Boxes, 
  Users, 
  LogOut, 
  ShieldCheck,
  Building2
} from 'lucide-react';

export function Navigation({ activeTab, setActiveTab }) {
  const { user, availableTenants, logout, switchTenant } = useAuth();

  if (!user) return null;

  const role = user.role;
  const isCustomer = role === 'customer';
  const isTechnician = ['technician', 'admin', 'manager', 'superuser'].includes(role);
  const isStaff = !isCustomer;
  const isSuperuser = role === 'superuser';

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, show: true },
    { id: 'customer-support', label: 'AI Self-Service', icon: Bot, show: true, badge: 'Pre-Ticket' },
    { id: 'engineer-copilot', label: 'Engineer Copilot', icon: Wrench, show: isTechnician, badge: 'Staff' },
    { id: 'calls', label: 'Service Calls', icon: PhoneCall, show: true },
    { id: 'knowledge', label: 'Knowledge Base', icon: BookOpen, show: isTechnician },
    { id: 'assets', label: 'Assets', icon: Package, show: true },
    { id: 'projects', label: 'Projects', icon: Briefcase, show: true },
    { id: 'inventory', label: 'Inventory & Spares', icon: Boxes, show: isStaff },
    { id: 'staff', label: 'Team Directory', icon: Users, show: isStaff },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">S</div>
        <div className="brand-text">
          <h1>Servy RAG</h1>
          <span>Field Service AI</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">Core Operations</div>
        {navItems.filter(item => item.show).map(item => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon className="icon" />
              <span style={{ flex: 1 }}>{item.label}</span>
              {item.badge && (
                <span style={{ 
                  fontSize: '0.65rem', 
                  padding: '2px 6px', 
                  borderRadius: '4px',
                  background: item.badge === 'Pre-Ticket' ? '#4f46e5' : '#334155',
                  color: '#fff',
                  fontWeight: 600
                }}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        {isSuperuser && availableTenants.length > 0 && (
          <div>
            <label style={{ fontSize: '0.7rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
              <Building2 size={12} /> Active Tenant (Admin)
            </label>
            <select
              className="tenant-selector"
              value={user.tenant_id || ''}
              onChange={(e) => switchTenant(e.target.value)}
            >
              {availableTenants.map(t => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
        )}

        <div className="user-profile-badge">
          <div className="avatar">
            {user.username.substring(0, 2).toUpperCase()}
          </div>
          <div className="user-meta">
            <div className="user-name" title={user.username}>{user.username}</div>
            <div className="user-role">
              <ShieldCheck size={12} color="#818cf8" />
              <span style={{ textTransform: 'capitalize' }}>{user.role}</span>
            </div>
          </div>
        </div>

        <button 
          onClick={logout}
          className="btn btn-secondary btn-sm"
          style={{ width: '100%', borderColor: 'rgba(255,255,255,0.15)', background: 'transparent', color: '#cbd5e1' }}
        >
          <LogOut size={14} /> Sign Out
        </button>
      </div>
    </aside>
  );
}
