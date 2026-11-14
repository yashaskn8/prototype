// Pure state helpers for the conversation -> ServiceCall review dialog.
// The description shown is always the server-built preview from
// GET /api/customer-support/conversations/{uuid}/ticket/; nothing here
// summarises messages or supplies contact details.

export const MISSING_FIELD_LABELS = {
  contact_name: 'contact name',
  phone_or_email: 'a phone number or email address',
  asset: 'the selected equipment',
};

export function describeMissingFields(fields = []) {
  return fields.map(field => MISSING_FIELD_LABELS[field] || field).join(', ');
}

/**
 * Interpret the ticket GET response.
 * Returns { kind: 'ticket', ticket } when a ServiceCall already exists, or
 * { kind: 'preview', preview } for the review dialog.
 */
export function interpretTicketResponse(conversationUuid, transcript, response) {
  if (response?.ticket) return { kind: 'ticket', ticket: response.ticket };
  const server = response?.preview || null;
  const lastMessage = transcript[transcript.length - 1];
  return {
    kind: 'preview',
    preview: {
      conversationUuid,
      messages: transcript,
      reportedIssue: server?.complaint || '',
      outstandingQuestion: lastMessage?.sender_role === 'assistant'
        ? (lastMessage.structured_contract?.clarification_question || '') : '',
      description: server?.description || '',
      sections: server?.sections || null,
      messageCount: typeof server?.message_count === 'number' ? server.message_count : null,
      missingFields: Array.isArray(server?.missing_fields) ? server.missing_fields : [],
      canCreate: Boolean(server?.can_create),
      previewDetail: response?.preview_detail || (server ? '' : 'The service call description could not be prepared.'),
      conversationSize: transcript.length,
    },
  };
}

/** Confirmation is allowed only when the server says so and nothing is in flight. */
export function canConfirmTicket(preview, submitting) {
  return Boolean(preview && !submitting && preview.canCreate && preview.sections
    && preview.missingFields.length === 0);
}

/**
 * Apply a failed POST (api.js throws { status, ...body }).
 * Returns { preview, message }.
 */
export function applyTicketSubmitError(preview, error) {
  const missing = Array.isArray(error?.missing_fields) ? error.missing_fields : [];
  const next = preview && missing.length
    ? { ...preview, missingFields: missing, canCreate: false }
    : preview;
  let message = error?.detail;
  if (!message) {
    message = error?.status
      ? 'The service request was not confirmed. Retry to check its status.'
      : 'Network error: the service request may not have been sent. Retry to check its status.';
  }
  return { preview: next, message };
}

/** The ServiceCall reference the backend returned, for display. */
export function ticketReference(ticket) {
  if (!ticket) return '';
  return ticket.ticket_ref || (ticket.servy_id != null ? `SRV-${ticket.servy_id}` : '');
}

/** True when the conversation changed after the preview was built (stale). */
export function previewIsStale(preview, conversationUuid, transcriptLength) {
  if (!preview) return false;
  return preview.conversationUuid !== conversationUuid || preview.conversationSize !== transcriptLength;
}
