import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { AssistantTypingIndicator } from '../src/components/AssistantTypingIndicator.js';
import {
  EMPTY_ASSISTANT_RESPONSE,
  createAssistantTypingMessage,
  finishRequest,
  hasAssistantResponse,
  replaceMessageById,
  tryBeginRequest,
} from '../src/components/assistantRequestLifecycle.js';

test('one three-dot assistant bubble is shared by greetings and document responses', () => {
  const greetingPending = createAssistantTypingMessage('greeting');
  const documentPending = createAssistantTypingMessage('retrieval');
  assert.deepEqual({ ...greetingPending, id: 'same' }, { ...documentPending, id: 'same' });
  const markup = renderToStaticMarkup(React.createElement(AssistantTypingIndicator));
  assert.equal((markup.match(/class="servy-typing-dot"/g) || []).length, 3);
  assert.match(markup, /role="status"/);
  assert.match(markup, /aria-label="Servy is responding"/);
});

test('the typing message stays pending across asynchronous retrieval, reranking and generation', async () => {
  const pending = createAssistantTypingMessage('request-1');
  let messages = [pending];
  let resolveRequest;
  const response = new Promise(resolve => { resolveRequest = resolve; });
  const request = response.then(result => {
    messages = replaceMessageById(messages, pending.id,
      { id: pending.id, role: 'assistant', data: result });
  });
  await Promise.resolve();
  assert.equal(messages[0].pending, true);
  resolveRequest({ response_type: 'DIAGNOSTIC_GUIDANCE' });
  await request;
  assert.equal(messages[0].pending, undefined);
});

test('the first available response replaces the typing message in place', () => {
  const pending = createAssistantTypingMessage('request-2');
  const messages = [{ id: 'user-1', role: 'user' }, pending];
  const settled = replaceMessageById(messages, pending.id,
    { id: pending.id, role: 'assistant', data: { message: 'Hello.' } });
  assert.equal(settled.length, 2);
  assert.equal(settled[1].data.message, 'Hello.');
  assert.equal(settled[1].pending, undefined);
});

test('both customer chat screens render the shared bubble before their request resolves', async () => {
  const [demo, customer] = await Promise.all([
    readFile(new URL('../src/pages/RealDemoPage.jsx', import.meta.url), 'utf8'),
    readFile(new URL('../src/pages/CustomerSupportPage.jsx', import.meta.url), 'utf8'),
  ]);
  for (const source of [demo, customer]) {
    assert.match(source, /import \{ AssistantTypingIndicator \} from '\.\.\/components\/AssistantTypingIndicator'/);
    assert.match(source, /createAssistantTypingMessage\(/);
    assert.match(source, /AssistantTypingIndicator \/>/);
  }
  assert.match(demo, /setMessages\(previous => \[\.\.\.previous,[\s\S]*?createAssistantTypingMessage\(responseId\)[\s\S]*?await callRagStream\(\s*'\/demo-api\/query-stream\/'/);
  assert.match(customer, /setMessages\(prev => \[\.\.\.prev, tempUserMsg, typingMessage\]\)[\s\S]*?await api\.post/);
});

test('errors and empty responses remove pending state and expose a recoverable message', () => {
  const pending = createAssistantTypingMessage('request-3');
  const failure = replaceMessageById([pending], pending.id,
    { id: pending.id, role: 'assistant', error: 'Servy could not complete the response. Please try again.' });
  assert.equal(failure[0].pending, undefined);
  assert.match(failure[0].error, /try again/);
  assert.equal(hasAssistantResponse({ response_type: 'DIAGNOSTIC_GUIDANCE' }), false);
  assert.match(EMPTY_ASSISTANT_RESPONSE, /try again/);
});

test('repeated submissions acquire one request slot and settle without duplicate indicators', () => {
  const requestRef = { current: null };
  assert.equal(tryBeginRequest(requestRef, 'request-4'), true);
  assert.equal(tryBeginRequest(requestRef, 'request-5'), false);
  const pending = createAssistantTypingMessage('request-4');
  const messages = replaceMessageById([pending], pending.id,
    { id: pending.id, role: 'assistant', data: { message: 'Done.' } });
  assert.equal(messages.filter(message => message.pending).length, 0);
  assert.equal(finishRequest(requestRef, 'request-5'), false);
  assert.equal(finishRequest(requestRef, 'request-4'), true);
  assert.equal(tryBeginRequest(requestRef, 'request-6'), true);
});

test('customer chat source does not render internal retrieval progress text', async () => {
  const files = await Promise.all([
    readFile(new URL('../src/pages/RealDemoPage.jsx', import.meta.url), 'utf8'),
    readFile(new URL('../src/pages/CustomerSupportPage.jsx', import.meta.url), 'utf8'),
  ]);
  const banned = /Searching documents|Searching knowledge base|Retrieving documents|Analyzing your query|Processing document chunks|Running semantic search|Checking the approved manuals|Searching…/i;
  assert.equal(files.some(source => banned.test(source)), false);
});

test('reduced-motion users get a static indicator', async () => {
  const css = await readFile(new URL('../src/components/AssistantTypingIndicator.css', import.meta.url), 'utf8');
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /animation:\s*none/);
});
