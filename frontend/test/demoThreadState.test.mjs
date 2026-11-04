import test from 'node:test';
import assert from 'node:assert/strict';
import { chatTitle, appendDemoThread, canOpenDemoThread, MAX_DEMO_THREADS }
  from '../src/pages/demoThreadState.js';

test('thread memory is capped at eight conversations', () => {
  let threads = [];
  for (let i = 0; i < 18; i++) {
    threads = appendDemoThread(threads, { id: 'thread-' + i, assetId: '60', title: 'Synthetic' });
  }
  assert.equal(threads.length, MAX_DEMO_THREADS);
  assert.equal(threads.at(-1).id, 'thread-17');
  assert.equal(threads[0].id, 'thread-10');
});

test('asset switching discards unrelated thread metadata', () => {
  const old = [{ id: 'old', assetId: '60', title: 'Airflow' }];
  const next = appendDemoThread(old, { id: 'new', assetId: '61', title: 'Pressure' });
  assert.deepEqual(next.map(t => t.id), ['new']);
  assert.equal(canOpenDemoThread(old[0], '61'), false);
  assert.equal(canOpenDemoThread(next[0], '61'), true);
});

test('chat title is short, normalized and text-only', () => {
  assert.equal(chatTitle(''), 'New chat');
  assert.equal(chatTitle('  Split AC\n not cooling  '), 'Split AC not cooling');
  assert.ok(chatTitle('x'.repeat(200)).length <= 60);
  assert.equal(chatTitle('\u0000What\u0001 about the fan?'), 'What about the fan?');
});

test('invalid and duplicate thread indexes fail safely', () => {
  assert.throws(() => appendDemoThread(null, { id: 'a', assetId: '60' }));
  assert.throws(() => appendDemoThread([], { id: 'a' }));
  const next = appendDemoThread([{ id: 'a', assetId: '60', title: 'Old' }],
    { id: 'a', assetId: '60', title: 'New' });
  assert.equal(next.length, 1);
  assert.equal(next[0].title, 'New');
});
