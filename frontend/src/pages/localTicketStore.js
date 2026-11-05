import { api, request } from '../api';

const CHANNEL_NAME = 'servy-local-ticket-updates';
const WORKSPACE_PATH = '/api/local-ticket-workspace/';

export function notifyTicketWorkspace(tenantId) {
  if (typeof BroadcastChannel !== 'undefined') {
    const channel = new BroadcastChannel(CHANNEL_NAME);
    channel.postMessage({ tenantId: String(tenantId) });
    channel.close();
  }
  window.dispatchEvent(new CustomEvent('servy-local-ticket-update', {
    detail: { tenantId: String(tenantId) },
  }));
}

export function subscribeTicketWorkspace(tenantId, callback) {
  const expectedTenant = String(tenantId);
  const onLocalUpdate = (event) => {
    if (String(event.detail?.tenantId) === expectedTenant) callback();
  };
  window.addEventListener('servy-local-ticket-update', onLocalUpdate);

  let channel;
  if (typeof BroadcastChannel !== 'undefined') {
    channel = new BroadcastChannel(CHANNEL_NAME);
    channel.onmessage = (event) => {
      if (String(event.data?.tenantId) === expectedTenant) callback();
    };
  }

  // Polling connects separate authenticated browser sessions on the same local app.
  const timer = window.setInterval(callback, 4000);
  return () => {
    window.removeEventListener('servy-local-ticket-update', onLocalUpdate);
    window.clearInterval(timer);
    channel?.close();
  };
}

export async function listLocalTickets() {
  const response = await api.get(WORKSPACE_PATH);
  return response.tickets || [];
}

export async function getLocalTicket(ticketId) {
  const response = await api.get(`${WORKSPACE_PATH}${encodeURIComponent(ticketId)}/`);
  return response.ticket;
}

export async function createLocalTicket(ticketFields, files = []) {
  const payload = new FormData();
  payload.append('ticket', JSON.stringify(ticketFields));
  files.forEach((file) => payload.append('attachments', file, file.name));
  const response = await api.post(WORKSPACE_PATH, payload);
  const ticket = response.ticket;
  notifyTicketWorkspace(ticket.tenantId);
  return ticket;
}

export async function updateLocalTicket(ticketId, changes) {
  const response = await request(`${WORKSPACE_PATH}${encodeURIComponent(ticketId)}/`, {
    method: 'PATCH',
    body: changes,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw { status: response.status, ...data };
  notifyTicketWorkspace(data.ticket.tenantId);
  return data.ticket;
}

export async function getLocalTicketAttachment(ticketId, attachmentId) {
  const response = await request(
    `${WORKSPACE_PATH}${encodeURIComponent(ticketId)}/attachments/${encodeURIComponent(attachmentId)}/`,
  );
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || 'The attachment could not be opened.');
  }
  return { blob: await response.blob() };
}
