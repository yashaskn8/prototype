export function makeConversationDescription(messages = []) {
  return messages
    .filter((message) => ['customer', 'assistant'].includes(message?.sender_role) && message?.message_text?.trim())
    .map((message) => `${message.sender_role === 'customer' ? 'Customer' : 'Servy'}:\n${message.message_text.trim()}`)
    .join('\n\n');
}
