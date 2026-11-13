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
