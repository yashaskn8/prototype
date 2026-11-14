import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const readSource = (path) => readFile(new URL(path, import.meta.url), 'utf8');

test('customer reviews the saved conversation and explicitly confirms the authoritative ticket request', async () => {
  const source = await readSource('../src/pages/CustomerSupportPage.jsx');
  assert.match(source, /conversations\/\$\{conversationUuid\}\/messages\//);
  assert.match(source, /role="dialog"[\s\S]*aria-modal="true"/);
  assert.match(source, /Confirm and create ticket/);
  assert.match(source, /confirm_ticket_creation:\s*true/);
  assert.match(source, /Servy guidance \(not confirmation of completion\)/);
  assert.match(source, /Service Tickets/);
});

test('technician inbox polls the authenticated backend and persists acknowledgement', async () => {
  const [page, app, navigation] = await Promise.all([
    readSource('../src/pages/TechnicianInboxPage.jsx'),
    readSource('../src/App.jsx'),
    readSource('../src/components/Navigation.jsx'),
  ]);
  assert.match(page, /api\.get\('\/api\/technician\/inbox\/'\)/);
  assert.match(page, /setInterval\(poll, POLL_INTERVAL_MS\)/);
  assert.match(page, /\/api\/technician\/inbox\/\$\{selectedTicket\.id\}\/acknowledge\//);
  for (const heading of ['Customer conversation', 'Confirmed troubleshooting', 'Verified supporting references', 'Outstanding information']) {
    assert.ok(page.includes(heading), `missing ticket section: ${heading}`);
  }
  assert.match(app, /TechnicianInboxPage/);
  assert.match(app, /technician\/inbox/);
  assert.match(navigation, /show: isTechnician/);
  assert.match(navigation, /http:\/\/127\.0\.0\.1:5175\//);

  // The SQLite "RAG Ticket Workspace" was retired: RAG tickets are official MySQL
  // ServiceCalls in the Technician Inbox, and old workspace links open that inbox.
  assert.doesNotMatch(app, /TicketWorkspacePage/);
  assert.doesNotMatch(navigation, /rag-ticket-workspace/);
  assert.match(app, /INBOX_LOCATIONS = \['\/technician\/inbox', '\/technician\/rag-tickets'\]/);
  await assert.rejects(readSource('../src/pages/TicketWorkspacePage.jsx'));
  await assert.rejects(readSource('../src/pages/localTicketStore.js'));
});
