import assert from 'node:assert/strict';
import test from 'node:test';

import {
  mergeSavedThreads,
  removeDemoThread,
} from '../src/pages/demoThreadState.js';

const saved = (uuid, title = uuid) => ({
  id: `conv-${uuid}`,
  assetId: '1',
  conversationUuid: uuid,
  title,
});

test('stale history responses cannot restore UUIDs deleted in this session', () => {
  const existing = [saved('deleted-1'), saved('keep-1'), {
    id: 'draft-1', assetId: '1', title: 'New chat',
  }];
  const response = [saved('deleted-1', 'stale title'), saved('keep-1', 'fresh title')];

  const merged = mergeSavedThreads(existing, response, new Set(['deleted-1']));

  assert.deepEqual(merged.map(item => item.conversationUuid || item.id), [
    'draft-1',
    'keep-1',
  ]);
  assert.equal(merged.find(item => item.conversationUuid === 'keep-1').title, 'fresh title');
});

test('history merge preserves saved chats omitted from a partial page and local drafts', () => {
  const existing = [saved('older-chat'), { id: 'draft-1', assetId: '1', title: 'New chat' }];
  const response = [saved('newer-chat')];

  const merged = mergeSavedThreads(existing, response);

  assert.deepEqual(merged.map(item => item.conversationUuid || item.id), [
    'older-chat',
    'draft-1',
    'newer-chat',
  ]);
});

test('merging a refreshed row replaces the stale copy without duplicating it', () => {
  const existing = [saved('chat-1', 'Old title'), saved('chat-2')];
  const response = [saved('chat-1', 'Updated title')];

  const merged = mergeSavedThreads(existing, response);

  assert.equal(merged.filter(item => item.conversationUuid === 'chat-1').length, 1);
  assert.equal(merged.find(item => item.conversationUuid === 'chat-1').title, 'Updated title');
  assert.ok(merged.some(item => item.conversationUuid === 'chat-2'));
});

test('sequential concurrent-delete completions remove from the latest list', () => {
  let current = [saved('chat-a'), saved('chat-b'), saved('chat-c')];

  current = removeDemoThread(current, 'conv-chat-a');
  current = removeDemoThread(current, 'conv-chat-b');

  assert.deepEqual(current.map(item => item.conversationUuid), ['chat-c']);
});

test('removing one chat does not affect another asset or draft', () => {
  const current = [
    saved('asset-1-chat'),
    { id: 'asset-2-draft', assetId: '2', title: 'New chat' },
  ];

  const remaining = removeDemoThread(current, 'conv-asset-1-chat');

  assert.deepEqual(remaining, [{ id: 'asset-2-draft', assetId: '2', title: 'New chat' }]);
});

test('refresh keeps the active local chat identity for the same server UUID', () => {
  const local = { ...saved('chat-1'), id: 'local-thread-1' };
  const result = mergeSavedThreads([local], [saved('chat-1', 'Updated')]);
  assert.equal(result.length, 1);
  assert.equal(result[0].id, 'local-thread-1');
  assert.equal(result[0].title, 'Updated');
});

test('delete removes all aliases of a saved conversation UUID', () => {
  const result = removeDemoThread([
    { ...saved('chat-1'), id: 'local-thread-1' },
    saved('chat-1'),
    saved('chat-2'),
  ], 'local-thread-1');
  assert.deepEqual(result, [saved('chat-2')]);
});
