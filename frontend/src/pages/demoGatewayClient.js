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
        data: message.structured_contract || { answer: message.message_text } });
}

// Citation links from the official API point at /api/...; the gateway serves them under /demo-api/.
export function gatewaySourceUrl(url) {
  return String(url || '').replace('/api/customer-support/source/', '/demo-api/customer-support/source/');
}
