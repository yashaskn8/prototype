export const EMPTY_ASSISTANT_RESPONSE = 'Servy did not return a response. Please try again.';

export function createAssistantTypingMessage(id, { senderRole = false, createdAt } = {}) {
  return {
    id,
    [senderRole ? 'sender_role' : 'role']: 'assistant',
    pending: true,
    ...(createdAt ? { created_at: createdAt } : {}),
  };
}

export function replaceMessageById(messages, id, replacement) {
  let replaced = false;
  const next = messages.map(message => {
    if (message.id !== id || replaced) return message;
    replaced = true;
    return replacement;
  });
  return next;
}

export function hasAssistantResponse(message) {
  if (!message || typeof message !== 'object') return false;
  const textFields = [message.text, message.message_text, message.answer,
    message.summary, message.message];
  if (textFields.some(value => typeof value === 'string' && value.trim())) return true;

  const contract = message.structured_contract || message;
  if ([contract.message, contract.answer, contract.summary, contract.clarification_question]
    .some(value => typeof value === 'string' && value.trim())) return true;
  if (Array.isArray(contract.presentation?.sections)
      && contract.presentation.sections.some(section =>
        Array.isArray(section.claims) && section.claims.some(claim =>
          typeof claim?.text === 'string' && claim.text.trim()))) return true;

  const handoff = message.handoff || contract.handoff;
  return Boolean(handoff && (
    handoff.reported_details?.length
    || handoff.checklist?.length
    || (typeof handoff.follow_up_question === 'string' && handoff.follow_up_question.trim())
  ));
}

export function tryBeginRequest(requestRef, requestId) {
  if (requestRef.current !== null) return false;
  requestRef.current = requestId;
  return true;
}

export function finishRequest(requestRef, requestId) {
  if (requestRef.current !== requestId) return false;
  requestRef.current = null;
  return true;
}
