import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  applyTicketSubmitError, canConfirmTicket, describeMissingFields,
  interpretTicketResponse, previewIsStale, ticketReference,
} from '../src/pages/ticketPreviewState.js';

const SECTIONS = Object.fromEntries([
  '1. Customer-reported problem', '2. Asset', '3. Symptoms and observations reported by the customer',
  '4. Questions asked during the conversation', '5. Customer responses',
  '6. Troubleshooting discussed (not confirmed as performed)', '7. Approved-manual findings and citations',
  '8. Unresolved', '9. For technician investigation',
].map(title => [title, [`${title} item`]]));
const TRANSCRIPT = [
  { id: 1, sender_role: 'customer', message_text: 'My split AC is running but not cooling.' },
  { id: 2, sender_role: 'assistant', message_text: 'Is the indoor fan running?',
    structured_contract: { clarification_question: 'Is the indoor fan running?' } },
];
const PREVIEW_RESPONSE = {
  ticket: null,
  preview: {
    complaint: 'My split AC is running but not cooling.',
    description: 'Customer conversation ... 9. For technician investigation',
    sections: SECTIONS, message_count: 2, missing_fields: [], can_create: true,
  },
};

test('preview loading uses the server description, all nine sections and the API counts', () => {
  const outcome = interpretTicketResponse('uuid-1', TRANSCRIPT, PREVIEW_RESPONSE);
  assert.equal(outcome.kind, 'preview');
  const p = outcome.preview;
  assert.equal(Object.keys(p.sections).length, 9);
  assert.equal(p.messageCount, 2);
  assert.equal(p.reportedIssue, 'My split AC is running but not cooling.');
  assert.equal(p.outstandingQuestion, 'Is the indoor fan running?');
  assert.deepEqual(p.missingFields, []);
  assert.equal(p.canCreate, true);
  assert.equal(canConfirmTicket(p, false), true);
});

test('an existing ServiceCall is shown instead of a new preview', () => {
  const outcome = interpretTicketResponse('uuid-1', TRANSCRIPT, { ticket: { id: 7, servy_id: 43100 } });
  assert.equal(outcome.kind, 'ticket');
  assert.equal(ticketReference(outcome.ticket), 'SRV-43100');
  assert.equal(ticketReference({ ticket_ref: 'SC-9', servy_id: 1 }), 'SC-9');
});

test('missing contacts from the preview disable confirmation and are described, not filled in', () => {
  const response = { ...PREVIEW_RESPONSE, preview: { ...PREVIEW_RESPONSE.preview, missing_fields: ['phone_or_email'], can_create: false } };
  const p = interpretTicketResponse('uuid-1', TRANSCRIPT, response).preview;
  assert.equal(canConfirmTicket(p, false), false);
  assert.equal(describeMissingFields(p.missingFields), 'a phone number or email address');
  assert.equal(describeMissingFields(['contact_name', 'phone_or_email']), 'contact name, a phone number or email address');
});

test('confirmation is disabled while submitting, without sections, or when the server forbids it', () => {
  const p = interpretTicketResponse('uuid-1', TRANSCRIPT, PREVIEW_RESPONSE).preview;
  assert.equal(canConfirmTicket(p, true), false);                       // repeated clicks
  assert.equal(canConfirmTicket({ ...p, sections: null }, false), false);
  assert.equal(canConfirmTicket({ ...p, canCreate: false }, false), false);
  assert.equal(canConfirmTicket(null, false), false);
  const none = interpretTicketResponse('uuid-1', [], { ticket: null, preview: null,
    preview_detail: 'A customer-reported issue is required before creating a service ticket.' }).preview;
  assert.equal(canConfirmTicket(none, false), false);
  assert.match(none.previewDetail, /customer-reported issue is required/);
});

test('a 400 with missing_fields from POST keeps confirmation disabled', () => {
  const p = interpretTicketResponse('uuid-1', TRANSCRIPT, PREVIEW_RESPONSE).preview;
  const { preview, message } = applyTicketSubmitError(p, {
    status: 400, detail: 'Contact details required to create a service call are missing from the customer record.',
    missing_fields: ['phone_or_email'],
  });
  assert.deepEqual(preview.missingFields, ['phone_or_email']);
  assert.equal(canConfirmTicket(preview, false), false);
  assert.match(message, /Contact details required/);
});

test('other API and network errors keep the dialog and report a retryable message', () => {
  const p = interpretTicketResponse('uuid-1', TRANSCRIPT, PREVIEW_RESPONSE).preview;
  const server = applyTicketSubmitError(p, { status: 500 });
  assert.equal(server.preview, p);
  assert.match(server.message, /not confirmed\. Retry/);
  const network = applyTicketSubmitError(p, new TypeError('Failed to fetch'));
  assert.match(network.message, /Network error/);
  const confirmation = applyTicketSubmitError(p, { status: 400, detail: 'Explicit confirmation is required to create a service ticket.' });
  assert.match(confirmation.message, /Explicit confirmation/);
});

test('the preview is stale when the conversation or its saved message count changes', () => {
  const p = interpretTicketResponse('uuid-1', TRANSCRIPT, PREVIEW_RESPONSE).preview;
  assert.equal(previewIsStale(p, 'uuid-1', 2), false);
  assert.equal(previewIsStale(p, 'uuid-1', 4), true);
  assert.equal(previewIsStale(p, 'uuid-2', 2), true);
});

test('the page uses the shared helpers and still requires explicit confirmation', () => {
  const page = readFileSync(new URL('../src/pages/CustomerSupportPage.jsx', import.meta.url), 'utf8');
  assert.match(page, /interpretTicketResponse\(conversationUuid, transcript, ticketResponse\)/);
  assert.match(page, /disabled=\{!canConfirmTicket\(ticketPreview, ticketSubmitting\)\}/);
  assert.match(page, /if \(!canConfirmTicket\(ticketPreview, ticketSubmitting\)\) return;/);
  assert.match(page, /applyTicketSubmitError\(ticketPreview, submitError\)/);
  assert.match(page, /\{ confirm_ticket_creation: true \}/);
  assert.match(page, /Object\.entries\(ticketPreview\.sections\)/);
  assert.match(page, /\/api\/customer-support\/query-stream\//);           // first message flow unchanged
});
