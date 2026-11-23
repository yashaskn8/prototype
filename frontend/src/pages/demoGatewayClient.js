// The demo gateway serves the official conversation and ServiceCall APIs from
// the isolated local development database (servy_dev).
export const CONVERSATIONS_PATH = '/demo-api/customer-support/conversations/';

export function conversationPath(uuid, suffix = '') {
  return `${CONVERSATIONS_PATH}${encodeURIComponent(uuid)}/${suffix}`;
}

// Saved messages -> the chat's message objects (customer text, assistant contract).
export function messagesFromServer(list) {
  return (Array.isArray(list) ? list : [])
    .filter(message => ['customer', 'assistant'].includes(message?.sender_role))
    .map(message => message.sender_role === 'customer'
      ? { id: `server-${message.id}`, role: 'user', text: message.message_text }
      : { id: `server-${message.id}`, role: 'assistant',
        data: { ...(message.structured_contract || { answer: message.message_text }),
          assistant_message_id: message.id, feedback: message.feedback || null } });
}

// Citation links from the official API point at /api/...; the gateway serves them under /demo-api/.
export function gatewaySourceUrl(url) {
  return String(url || '').replace('/api/customer-support/source/', '/demo-api/customer-support/source/');
}

// Turn a gateway response into data or a safe, meaningful error. Never parses a
// non-JSON body (e.g. an HTML error page) as JSON, so the customer sees a clear
// message instead of "Unexpected token '<'".
const STATUS_MESSAGES = {
  401: 'The access code was not accepted. Check it and try again.',
  403: 'This action is not permitted for this access code.',
  404: 'The requested item was not found.',
  429: 'Too many requests. Please wait a moment and try again.',
  503: 'Servy is busy or still starting. Please try again shortly.',
};

export async function parseDemoResponse(response) {
  if (response.status === 204) return {};
  const type = response.headers?.get?.('content-type') || '';
  let data = null;
  if (type.includes('application/json')) {
    try { data = await response.json(); } catch { data = null; }
  }
  if (!response.ok) {
    const detail = data && typeof data.detail === 'string' ? data.detail : '';
    const error = new Error(detail || STATUS_MESSAGES[response.status]
      || 'The Servy server could not complete this request. Please try again.');
    error.status = response.status;
    throw error;
  }
  if (data === null) {
    const error = new Error('The Servy server returned an unexpected response. Please try again.');
    error.status = response.status;
    throw error;
  }
  return data;
}
