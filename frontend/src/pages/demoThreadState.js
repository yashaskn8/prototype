// The Real-data demo keeps conversations only in this browser tab's RAM.
// No localStorage, IndexedDB, cookies, database writes or data export.
export const MAX_DEMO_THREADS = 8;

export function chatTitle(input) {
  const cleaned = String(input ?? '').normalize('NFKC')
    .replace(/[\u0000-\u001f\u007f]/g, ' ').replace(/\s+/g, ' ').trim();
  if (!cleaned) return 'New chat';
  return cleaned.length > 60 ? cleaned.slice(0, 57) + '…' : cleaned;
}

export function appendDemoThread(current, thread) {
  if (!Array.isArray(current) || !thread || !thread.id || !thread.assetId) {
    throw new Error('Invalid conversation index');
  }
  const scoped = current.filter(t => t.assetId === thread.assetId && t.id !== thread.id);
  return [...scoped, thread].slice(-MAX_DEMO_THREADS);
}

export function canOpenDemoThread(thread, selectedAssetId) {
  return Boolean(thread && selectedAssetId
    && String(thread.assetId) === String(selectedAssetId));
}

export function isDemoAssetThreadBound(threads, activeThreadId, selectedAssetId) {
  if (!selectedAssetId || !activeThreadId || !Array.isArray(threads)) return false;
  const active = threads.find(thread => thread?.id === activeThreadId);
  return Boolean(active && String(active.assetId) === String(selectedAssetId));
}

/**
 * Merge a customer-history response without restoring UUIDs deleted during
 * this browser session. Existing local drafts and unreturned saved rows remain.
 */
export function mergeSavedThreads(current, saved, deletedUuids = new Set()) {
  const deleted = deletedUuids instanceof Set ? deletedUuids : new Set(deletedUuids || []);
  const visibleCurrent = (Array.isArray(current) ? current : [])
    .filter(item => !item?.conversationUuid || !deleted.has(item.conversationUuid));
  const visibleSaved = (Array.isArray(saved) ? saved : [])
    .filter(item => item?.conversationUuid && !deleted.has(item.conversationUuid));
  const savedUuids = new Set(visibleSaved.map(item => item.conversationUuid));
  return [
    ...visibleCurrent.filter(item => !item.conversationUuid || !savedUuids.has(item.conversationUuid)),
    ...visibleSaved,
  ];
}

export function removeDemoThread(current, threadId) {
  return (Array.isArray(current) ? current : []).filter(item => item?.id !== threadId);
}
