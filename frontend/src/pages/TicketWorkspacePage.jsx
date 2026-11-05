import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle, Check, ChevronDown, ClipboardList, Download, FilePlus2,
  ImagePlus, Inbox, LoaderCircle, Paperclip, RefreshCw, Send, Ticket, X,
} from 'lucide-react';
import { useAuth } from '../AuthContext';
import { api } from '../api';
import {
  createLocalTicket, getLocalTicket, getLocalTicketAttachment, listLocalTickets,
  subscribeTicketWorkspace, updateLocalTicket,
} from './localTicketStore';
import './TicketWorkspacePage.css';

const TICKET_TYPES = ['Breakdown Service', 'Installation / Demo', 'Service', 'Preventive Maintenance (PM)', 'AMC'];
const SEVERITIES = ['Low', 'Medium', 'High', 'Critical'];
const PRIORITIES = [
  { value: 'low', label: 'Low' }, { value: 'normal', label: 'Normal' },
  { value: 'high', label: 'High' }, { value: 'critical', label: 'Critical' },
];
const MACHINE_STATUSES = ['Running', 'Intermittent', 'Stopped', 'Unknown'];
const COMPLAINT_TYPES = [
  'Cooling / performance', 'Electrical', 'Noise / vibration', 'Leakage',
  'Installation / commissioning', 'Preventive maintenance', 'Other',
];
const TICKET_STATUSES = [
  ['open', 'Open'], ['assigned', 'Assigned'], ['in_progress', 'In progress'],
  ['marked_for_closure', 'Marked for closure'], ['closed', 'Closed'],
];
const MAX_FILE_SIZE = 25 * 1024 * 1024;

function formatDate(value) {
  return new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}

function SelectField({ label, value, onChange, options, required = false, placeholder }) {
  return (
    <label className="ltw-field">
      <span>{label}{required && <b aria-hidden="true"> *</b>}</span>
      <span className="ltw-select-wrap">
        <select value={value} onChange={onChange} required={required}>
          {placeholder && <option value="">{placeholder}</option>}
          {options.map((option) => {
            const item = typeof option === 'string' ? { value: option, label: option } : option;
            return <option key={item.value} value={item.value}>{item.label}</option>;
          })}
        </select>
        <ChevronDown size={16} aria-hidden="true" />
      </span>
    </label>
  );
}

export function TicketWorkspacePage({ prefill, onClearPrefill }) {
  const { user } = useAuth();
  const isCustomer = user?.role === 'customer';
  const canViewTechnicianInbox = ['technician', 'admin', 'manager', 'superuser'].includes(user?.role);
  const tenantId = user?.tenant_id;
  const [context, setContext] = useState(null);
  const [contextLoading, setContextLoading] = useState(() => user?.role === 'customer');
  const [contextError, setContextError] = useState('');
  const [mode, setMode] = useState('create');
  const [form, setForm] = useState({
    callType: TICKET_TYPES[0], siteId: '', assetId: '', contactPerson: '', severity: '',
    priority: '', machineStatus: '', complaintType: '', description: '',
  });
  const [files, setFiles] = useState([]);
  const [tickets, setTickets] = useState([]);
  const [selectedTicketId, setSelectedTicketId] = useState('');
  const selectedTicketIdRef = useRef('');
  const [selectedTicketDetails, setSelectedTicketDetails] = useState(null);
  const [ticketLoading, setTicketLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(null);
  const [technicianNotes, setTechnicianNotes] = useState('');

  const loadTickets = useCallback(async () => {
    if (!tenantId) {
      setTickets([]);
      setTicketLoading(false);
      return;
    }
    try {
      const rows = await listLocalTickets(tenantId);
      setTickets(rows.filter((ticket) => String(ticket.tenantId) === String(tenantId)
        && (!isCustomer || String(ticket.customerId) === String(context?.customer?.id))));
      setTicketLoading(false);
    } catch (loadError) {
      setError(loadError.message || 'Could not load the local ticket workspace.');
      setTicketLoading(false);
    }
  }, [context?.customer?.id, isCustomer, tenantId]);

  useEffect(() => {
    if (!isCustomer) {
      return undefined;
    }
    let active = true;
    api.get('/api/customer-support/context/')
      .then((data) => {
        if (!active) return;
        setContext(data);
        setContextError('');
        setForm((current) => {
          const siteId = String(prefill?.site?.id || prefill?.siteId || current.siteId || data.sites?.[0]?.id || '');
          const siteAssets = (data.assets || []).filter((asset) => String(asset.site_id) === siteId);
          const assetId = String(prefill?.asset?.id || prefill?.assetId || current.assetId || siteAssets[0]?.id || '');
          return {
            ...current,
            siteId,
            assetId,
            description: prefill?.description || current.description,
          };
        });
      })
      .catch((loadError) => {
        if (active) setContextError(loadError.detail || loadError.message || 'Could not load authorized customer facilities.');
      })
      .finally(() => { if (active) setContextLoading(false); });
    return () => { active = false; };
  }, [isCustomer, prefill]);

  useEffect(() => {
    loadTickets();
    return subscribeTicketWorkspace(tenantId, loadTickets);
  }, [loadTickets, tenantId]);

  const facilities = context?.sites || [];
  const authorizedAssets = context?.assets || [];
  const facilityAssets = authorizedAssets.filter((asset) => String(asset.site_id) === String(form.siteId));
  const selectedTicket = selectedTicketDetails?.id === selectedTicketId ? selectedTicketDetails : null;

  function updateForm(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function setFacility(siteId) {
    const firstAsset = authorizedAssets.find((asset) => String(asset.site_id) === String(siteId));
    setForm((current) => ({ ...current, siteId, assetId: firstAsset ? String(firstAsset.id) : '' }));
  }

  function addFiles(incoming) {
    const selected = Array.from(incoming || []);
    const oversized = selected.filter((file) => file.size > MAX_FILE_SIZE);
    if (oversized.length) {
      setError(`Each attachment must be 25 MB or smaller: ${oversized.map((file) => file.name).join(', ')}`);
      return;
    }
    setError('');
    setFiles((current) => [...current, ...selected].filter((file, index, list) =>
      list.findIndex((item) => `${item.name}:${item.size}:${item.lastModified}` === `${file.name}:${file.size}:${file.lastModified}`) === index));
  }

  async function submitTicket(event) {
    event.preventDefault();
    setError('');
    setSuccess(null);
    if (!context?.customer?.id || !tenantId) {
      setError('A verified customer and tenant context is required to create this local ticket.');
      return;
    }
    const site = facilities.find((item) => String(item.id) === String(form.siteId));
    const asset = authorizedAssets.find((item) => String(item.id) === String(form.assetId));
    if (!site || !asset || String(asset.site_id) !== String(site.id)) {
      setError('Choose a facility and equipment item from your authorized customer context.');
      return;
    }
    if (!form.complaintType || !form.description.trim()) {
      setError('Select a complaint type and add a description before submitting.');
      return;
    }

    setSubmitting(true);
    try {
      const ticket = await createLocalTicket({
        siteId: String(site.id),
        assetId: String(asset.id),
        callType: form.callType,
        complaintType: form.complaintType,
        complaintText: form.description.trim(),
        contactName: form.contactPerson.trim(),
        severity: form.severity,
        priority: form.priority || 'normal',
        machineStatus: form.machineStatus,
        sourceConversationId: prefill?.conversationUuid || '',
      }, files);
      setSuccess(ticket);
      setFiles([]);
      selectedTicketIdRef.current = ticket.id;
      setSelectedTicketId(ticket.id);
      setSelectedTicketDetails(ticket);
      setMode('my-tickets');
      await loadTickets();
      onClearPrefill?.();
    } catch (submitError) {
      setError(submitError.message || 'Could not save this ticket in the local workspace.');
    } finally {
      setSubmitting(false);
    }
  }

  async function changeStatus(ticket, status) {
    try {
      const updated = await updateLocalTicket(ticket.id, { status });
      setSelectedTicketDetails(updated);
      await loadTickets();
    } catch (updateError) {
      setError(updateError.message || 'Could not update this local ticket.');
    }
  }

  async function saveTechnicianNotes() {
    if (!selectedTicket) return;
    try {
      const updated = await updateLocalTicket(selectedTicket.id, { technicianNotes });
      setSelectedTicketDetails(updated);
      await loadTickets();
    } catch (updateError) {
      setError(updateError.message || 'Could not save technician notes.');
    }
  }

  async function downloadAttachment(attachment) {
    let stored;
    try {
      stored = await getLocalTicketAttachment(selectedTicket.id, attachment.id);
    } catch (downloadError) {
      setError(downloadError.message || 'The attachment could not be opened.');
      return;
    }
    if (!stored?.blob) {
      setError('The attachment is no longer available in this browser workspace.');
      return;
    }
    const url = URL.createObjectURL(stored.blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = stored.name;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function openTicket(ticket) {
    selectedTicketIdRef.current = ticket.id;
    setSelectedTicketId(ticket.id);
    setSelectedTicketDetails(null);
    setTechnicianNotes(ticket.technicianNotes || '');
    setError('');
    try {
      const details = await getLocalTicket(ticket.id);
      if (selectedTicketIdRef.current === ticket.id) {
        setSelectedTicketDetails(details);
        setTechnicianNotes(details.technicianNotes || '');
      }
    } catch (detailError) {
      if (selectedTicketIdRef.current === ticket.id) setError(detailError.detail || detailError.message || 'Could not load ticket details.');
    }
  }

  const ticketForm = (
    <section className="ltw-form-card" aria-label="Create new ticket form">
      <header className="ltw-dialog-heading">
        <div>
          <span className="ltw-eyebrow">SERVICE REQUEST</span>
          <h1>Create New Ticket</h1>
          <p>Fill in the details below to raise a new service ticket.</p>
        </div>
        <button className="ltw-icon-button" type="button" aria-label="Close ticket form" onClick={() => setMode('my-tickets')}>
          <X size={18} />
        </button>
      </header>

      <form className="ltw-form" onSubmit={submitTicket}>
        <div className="ltw-section-title">TICKET DETAILS</div>

        <fieldset className="ltw-type-fieldset">
          <legend>Ticket Type <b>*</b></legend>
          <div className="ltw-type-options" role="group" aria-label="Ticket type">
            {TICKET_TYPES.map((type) => (
              <button key={type} type="button" className={form.callType === type ? 'selected' : ''}
                aria-pressed={form.callType === type} onClick={() => updateForm('callType', type)}>{type}</button>
            ))}
          </div>
        </fieldset>

        <div className="ltw-grid ltw-grid-three">
          <label className="ltw-field"><span>Customer</span>
            <input readOnly value={context?.customer?.name || (contextLoading ? 'Loading customer…' : '')} placeholder="Customer" />
          </label>
          <SelectField label="Facility" required value={form.siteId} placeholder="Select Facility"
            options={facilities.map((site) => ({ value: String(site.id), label: site.name }))}
            onChange={(event) => setFacility(event.target.value)} />
          <label className="ltw-field"><span>Contact Person</span>
            <input maxLength={150} value={form.contactPerson} onChange={(event) => updateForm('contactPerson', event.target.value)} placeholder="Enter contact person" />
          </label>
        </div>

        <SelectField label="Facility Equipment" required value={form.assetId} placeholder="Select Facility Equipment"
          options={facilityAssets.map((asset) => ({ value: String(asset.id), label: `${asset.name}${asset.asset_code ? ` (${asset.asset_code})` : ''}` }))}
          onChange={(event) => updateForm('assetId', event.target.value)} />

        <div className="ltw-grid ltw-grid-two">
          <SelectField label="Severity Level" value={form.severity} placeholder="Select Severity Level"
            options={SEVERITIES} onChange={(event) => updateForm('severity', event.target.value)} />
          <SelectField label="Priority Level" value={form.priority} placeholder="Select Priority Level"
            options={PRIORITIES} onChange={(event) => updateForm('priority', event.target.value)} />
          <SelectField label="Machine Status" value={form.machineStatus} placeholder="Select Machine Status"
            options={MACHINE_STATUSES} onChange={(event) => updateForm('machineStatus', event.target.value)} />
          <SelectField label="Complaint Type" required value={form.complaintType} placeholder="Select Complaint"
            options={COMPLAINT_TYPES} onChange={(event) => updateForm('complaintType', event.target.value)} />
        </div>

        <label className="ltw-field ltw-description-field">
          <span>Description <b>*</b></span>
          <div className="ltw-editor">
            <div className="ltw-editor-toolbar" aria-label="Description formatting toolbar">
              <span>aA⌄</span><i aria-hidden="true">B</i><i aria-hidden="true">I</i><u>U</u><span>≡</span><span>A⌄</span><span>✎⌄</span>
              {prefill?.description && <span className="ltw-autofill-tag"><Check size={13} /> RAG conversation added</span>}
            </div>
            <textarea required rows={9} value={form.description}
              onChange={(event) => updateForm('description', event.target.value)}
              placeholder="Enter the issue description…" />
          </div>
          <small className="ltw-help">The full customer and Servy conversation is included when you create a ticket from AI Self-Service.</small>
        </label>

        <div className="ltw-field">
          <span>Issue Documentation</span>
          <label className="ltw-dropzone" onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => { event.preventDefault(); addFiles(event.dataTransfer.files); }}>
            <input type="file" multiple accept="image/*,video/*,.pdf,.doc,.docx,.xls,.xlsx,.txt,.csv"
              onChange={(event) => { addFiles(event.target.files); event.target.value = ''; }} />
            <FilePlus2 size={24} />
            <strong>Click or drag to upload</strong>
            <span>Images, videos, PDFs, documents and more (max 25 MB each)</span>
            <small><ImagePlus size={14} /> Images <span>·</span> Videos <span>·</span> Docs &amp; more</small>
          </label>
          {files.length > 0 && <ul className="ltw-file-list">
            {files.map((file) => <li key={`${file.name}:${file.size}:${file.lastModified}`}>
              <Paperclip size={15} /><span>{file.name}</span><small>{(file.size / (1024 * 1024)).toFixed(1)} MB</small>
              <button type="button" aria-label={`Remove ${file.name}`} onClick={() => setFiles((current) => current.filter((item) => item !== file))}><X size={14} /></button>
            </li>)}
          </ul>}
        </div>

        {(contextError || error) && <div className="ltw-error" role="alert"><AlertCircle size={17} />{contextError || error}</div>}
        <footer className="ltw-form-footer">
          <span>Local ticket workspace · shared with this tenant’s technician view on this local app</span>
          <button className="ltw-submit" type="submit" disabled={submitting || contextLoading || !context?.customer}>
            {submitting ? <LoaderCircle size={17} className="ltw-spin" /> : <Send size={16} />}
            {submitting ? 'Creating ticket…' : 'Create Ticket'}
          </button>
        </footer>
      </form>
    </section>
  );

  const inbox = (
    <section className="ltw-inbox-shell">
      <header className="ltw-inbox-heading">
        <div><span className="ltw-eyebrow">{isCustomer ? 'CUSTOMER PORTAL' : 'FIELD OPERATIONS'}</span>
          <h1>{isCustomer ? 'My Service Tickets' : 'Technician Ticket Inbox'}</h1>
          <p>{isCustomer ? 'Track requests raised for your equipment.' : 'New customer requests appear here from the connected local ticket workspace.'}</p></div>
        <button className="ltw-refresh" type="button" onClick={loadTickets}><RefreshCw size={16} /> Refresh</button>
      </header>

      {ticketLoading ? <div className="ltw-empty"><LoaderCircle size={20} className="ltw-spin" /> Loading tickets…</div>
        : tickets.length === 0 ? <div className="ltw-empty"><Inbox size={28} /><strong>No tickets yet</strong>
          <span>{isCustomer ? 'Create a ticket from your AI Self-Service conversation.' : 'Customer-submitted local tickets will appear in this inbox.'}</span>
          {isCustomer && <button type="button" className="ltw-submit" onClick={() => { setSuccess(null); setMode('create'); }}>Create New Ticket</button>}
        </div>
        : <div className="ltw-ticket-layout">
          <div className="ltw-ticket-list" aria-label="Tickets">
            {tickets.map((ticket) => <button type="button" key={ticket.id}
              className={`ltw-ticket-row ${selectedTicketId === ticket.id ? 'active' : ''}`}
              onClick={() => openTicket(ticket)}>
              <span className="ltw-ticket-row-top"><strong>{ticket.id}</strong><span className={`ltw-status ltw-status-${ticket.status}`}>{ticket.status.replaceAll('_', ' ')}</span></span>
              <b>{ticket.complaintType}</b>
              <span>{ticket.assetName} · {ticket.customerName}</span>
              <small>{formatDate(ticket.createdAt)}</small>
            </button>)}
          </div>
          {selectedTicketId && !selectedTicket ? <div className="ltw-empty ltw-detail-placeholder"><LoaderCircle size={20} className="ltw-spin" /> Loading ticket details…</div>
            : selectedTicket ? <article className="ltw-ticket-detail">
            <header className="ltw-detail-header"><div><span className="ltw-eyebrow">{selectedTicket.id}</span><h2>{selectedTicket.complaintType}</h2>
              <p>Raised {formatDate(selectedTicket.createdAt)}</p></div>
              {canViewTechnicianInbox
                ? <SelectField label="Status" value={selectedTicket.status}
                  options={TICKET_STATUSES.map(([value, label]) => ({ value, label }))}
                  onChange={(event) => changeStatus(selectedTicket, event.target.value)} />
                : <span className={`ltw-status ltw-status-${selectedTicket.status}`}>{selectedTicket.status.replaceAll('_', ' ')}</span>}
            </header>
            <dl className="ltw-ticket-meta">
              <div><dt>Customer</dt><dd>{selectedTicket.customerName}</dd></div>
              <div><dt>Facility</dt><dd>{selectedTicket.siteName}</dd></div>
              <div><dt>Equipment</dt><dd>{selectedTicket.assetName}{selectedTicket.assetCode ? ` (${selectedTicket.assetCode})` : ''}</dd></div>
              <div><dt>Ticket type</dt><dd>{selectedTicket.callType}</dd></div>
              <div><dt>Priority</dt><dd>{(selectedTicket.priority || 'normal').replace(/^./, (letter) => letter.toUpperCase())}</dd></div>
              <div><dt>Contact person</dt><dd>{selectedTicket.contactName || 'Not provided'}</dd></div>
              {selectedTicket.machineStatus && <div><dt>Machine status</dt><dd>{selectedTicket.machineStatus}</dd></div>}
              {selectedTicket.severity && <div><dt>Severity</dt><dd>{selectedTicket.severity}</dd></div>}
            </dl>
            <section className="ltw-description-view"><h3>Customer description and RAG conversation</h3>
              <pre>{selectedTicket.complaintText}</pre></section>
            {selectedTicket.attachments?.length > 0 && <section className="ltw-attachments"><h3>Issue documentation</h3>
              {selectedTicket.attachments.map((attachment) => <button type="button" key={attachment.id} onClick={() => downloadAttachment(attachment)}>
                <Paperclip size={15} /><span>{attachment.name}</span><Download size={15} /></button>)}</section>}
            {!isCustomer && <section className="ltw-notes"><label htmlFor="ltw-tech-notes">Technician notes</label>
              <textarea id="ltw-tech-notes" rows={3} value={technicianNotes} onChange={(event) => setTechnicianNotes(event.target.value)} placeholder="Add an internal update for this local ticket" />
              <button type="button" className="ltw-submit" onClick={saveTechnicianNotes}>Save Update</button></section>}
          </article> : <div className="ltw-empty ltw-detail-placeholder"><ClipboardList size={26} /><span>Select a ticket to view its details.</span></div>}
        </div>}
    </section>
  );

  if (!tenantId || (!isCustomer && !canViewTechnicianInbox)) return <div className="ltw-page"><div className="ltw-error"><AlertCircle size={17} />Sign in to an authorized customer or technician account to open this workspace.</div></div>;

  return (
    <div className="ltw-page">
      <div className="ltw-modebar">
        <div className="ltw-mode-switch" role="tablist" aria-label="Ticket workspace">
          {isCustomer && <>
            <button type="button" role="tab" aria-selected={mode === 'create'} className={mode === 'create' ? 'active' : ''}
              onClick={() => { setSuccess(null); setMode('create'); }}>Create Ticket</button>
            <button type="button" role="tab" aria-selected={mode === 'my-tickets'} className={mode === 'my-tickets' ? 'active' : ''}
              onClick={() => setMode('my-tickets')}>My Tickets <span>{tickets.length}</span></button>
          </>}
          {!isCustomer && <span><Ticket size={16} /> Technician Workspace</span>}
        </div>
        {!isCustomer && <div className="ltw-local-indicator"><span /> Connected local ticket inbox</div>}
      </div>

      {success && <div className="ltw-success" role="status">
        <span className="ltw-success-icon"><Check size={18} /></span>
        <div><strong>Ticket added to technician inbox</strong><span>Local workspace ID: {success.id}</span></div>
        <button type="button" onClick={() => setSuccess(null)} aria-label="Dismiss confirmation"><X size={16} /></button>
      </div>}
      {contextLoading && isCustomer && mode === 'create' ? <div className="ltw-loading"><LoaderCircle size={20} className="ltw-spin" /> Loading your authorized facilities and equipment…</div>
        : isCustomer && mode === 'create' ? ticketForm : inbox}
    </div>
  );
}
