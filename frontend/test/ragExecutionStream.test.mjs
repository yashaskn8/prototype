import test from 'node:test';
import assert from 'node:assert/strict';
import {
  consumeRagEventStream,
  mergeRagExecutionEvent,
} from '../src/components/ragExecutionStream.js';

function responseFromChunks(chunks, status = 200) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream, {
    status,
    headers: { 'Content-Type': 'text/event-stream; charset=utf-8' },
  });
}

test('POST SSE parser accepts fragmented frames and returns the final payload', async () => {
  const frames = [
    ': servy-rag-stream\n\n',
    'event: log\ndata: {"type":"stage","event_id":"a1","stage":"retrieval","status":"started","timestamp":"2026-11-13T00:00:00Z","metrics":{}}\n\n',
    'event: log\ndata: {"type":"stage","event_id":"a1","stage":"retrieval","status":"completed","timestamp":"2026-11-13T00:00:01Z","duration_ms":100,"metrics":{}}\n\n',
    'event: result\ndata: {"answer":"synthetic answer"}\n\n',
  ].join('');
  const chunks = [frames.slice(0, 17), frames.slice(17, 94), frames.slice(94, 181), frames.slice(181)];
  const logs = [];
  const result = await consumeRagEventStream(responseFromChunks(chunks), event => logs.push(event));
  assert.deepEqual(result, { answer: 'synthetic answer' });
  assert.equal(logs.length, 2);
  assert.equal(logs[0].status, 'started');
  assert.equal(logs[1].status, 'completed');
});

test('SSE error frame rejects with its safe detail', async () => {
  const response = responseFromChunks([
    'event: error\ndata: {"detail":"RAG request failed. Please retry."}\n\n',
  ]);
  await assert.rejects(
    consumeRagEventStream(response),
    /RAG request failed/,
  );
});

test('HTTP errors are decoded as normal JSON errors', async () => {
  const response = new Response(JSON.stringify({ detail: 'Service staff only' }), {
    status: 403,
    headers: { 'Content-Type': 'application/json' },
  });
  await assert.rejects(
    consumeRagEventStream(response),
    error => error.status === 403 && error.message === 'Service staff only',
  );
});

test('stream ending without a result is reported as incomplete', async () => {
  const response = responseFromChunks([
    'event: log\ndata: {"type":"stage","event_id":"a1","stage":"retrieval","status":"started","metrics":{}}\n\n',
  ]);
  await assert.rejects(consumeRagEventStream(response), /ended before an answer/);
});

test('execution log updates one started event rather than duplicating it', () => {
  const started = {
    type: 'stage', event_id: 'same-stage', stage: 'retrieval',
    status: 'started', metrics: {},
  };
  const completed = {
    ...started, status: 'completed', duration_ms: 23,
  };
  const first = mergeRagExecutionEvent([], started);
  const second = mergeRagExecutionEvent(first, completed);
  assert.equal(second.length, 1);
  assert.equal(second[0].status, 'completed');
  assert.equal(second[0].duration_ms, 23);
});

test('non-stage events are ignored by the stage reducer', () => {
  const original = [{ event_id: 'a', status: 'completed' }];
  assert.equal(
    mergeRagExecutionEvent(original, { type: 'result', data: { answer: 'ok' } }),
    original,
  );
});
