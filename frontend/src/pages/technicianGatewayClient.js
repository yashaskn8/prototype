// The technician demo reads and updates official ServiceCall records through the
// technician gateway, which serves the official inbox APIs from servy_dev.
export const INBOX_PATH = '/tech-api/technician/inbox/';

export function ticketPath(id, suffix = '') {
  return `${INBOX_PATH}${encodeURIComponent(id)}/${suffix}`;
}

export function attachmentPath(ticketId, attachmentId) {
  return ticketPath(ticketId, `attachments/${encodeURIComponent(attachmentId)}/`);
}

// Statuses a technician may set (the server rejects anything else).
export const TECHNICIAN_STATUS_OPTIONS = [
  ['assigned', 'Assigned'],
  ['in_progress', 'In progress'],
  ['marked_for_closure', 'Marked for closure'],
  ['closed', 'Closed'],
];

export function defaultUpdateStatus(status) {
  return TECHNICIAN_STATUS_OPTIONS.some(([value]) => value === status) ? status : 'assigned';
}

// Official ticket fields -> the names this page renders.
export function normalizeTicket(ticket) {
  if (!ticket || typeof ticket !== 'object') return null;
  return {
    id: ticket.id,
    ref: ticket.ticket_ref || (ticket.servy_id != null ? `SRV-${ticket.servy_id}` : String(ticket.id)),
    status: ticket.status || 'open',
    complaintType: ticket.complaint_type || '',
    priority: ticket.priority || '',
    customerName: ticket.customer?.name || '',
    siteName: ticket.site?.name || '',
    assetName: ticket.asset?.name || '',
    assetCode: ticket.asset?.asset_code || '',
    contactName: ticket.contact?.name || '',
    createdAt: ticket.created_at,
    complaintText: ticket.complaint_text || '',
    description: ticket.description || '',
    attachments: Array.isArray(ticket.attachments) ? ticket.attachments : [],
    conversation: Array.isArray(ticket.conversation) ? ticket.conversation : [],
    supportingDocuments: Array.isArray(ticket.supporting_documents) ? ticket.supporting_documents : [],
    updates: Array.isArray(ticket.updates) ? ticket.updates : [],
    canAcknowledge: ticket.can_acknowledge === true,
  };
}

export function normalizeTickets(data) {
  return (Array.isArray(data?.tickets) ? data.tickets : []).map(normalizeTicket).filter(Boolean);
}
