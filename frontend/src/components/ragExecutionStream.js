/** Parse a POST-based Server-Sent Events response used by Servy RAG. */
export async function consumeRagEventStream(response, onLog = () => {}) {
  if (!response?.ok) {
    const payload = await response?.json?.().catch(() => ({}));
    const error = new Error(payload?.detail || "RAG request failed.");
    error.status = response?.status;
    throw error;
  }
  if (!response.body?.getReader) {
    throw new Error("Live execution streaming is unavailable in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  let terminalError = null;

  const dispatch = (frame) => {
    if (!frame.trim() || frame.startsWith(":")) return;
    let eventName = "message";
    const dataLines = [];
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    let payload;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      throw new Error("Servy returned an invalid execution event.");
    }

    if (eventName === "log" && payload?.type === "stage") {
      onLog(payload);
    } else if (eventName === "result") {
      result = payload;
    } else if (eventName === "error") {
      terminalError = new Error(payload?.detail || "RAG request failed.");
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let boundary;
      while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const frame = buffer.slice(0, boundary);
        const separator = buffer.slice(boundary).match(/^\r?\n\r?\n/)[0];
        buffer = buffer.slice(boundary + separator.length);
        dispatch(frame);
        if (result || terminalError) {
          await reader.cancel().catch(() => {});
          break;
        }
      }
      if (result || terminalError || done) break;
    }
  } finally {
    reader.releaseLock();
  }
  if (terminalError) throw terminalError;
  if (!result) throw new Error("The RAG stream ended before an answer was received.");
  return result;
}

export function mergeRagExecutionEvent(previous = [], event) {
  if (!event || event.type !== "stage" || typeof event.event_id !== "string") {
    return previous;
  }
  const index = previous.findIndex(item => item.event_id === event.event_id);
  if (index < 0) return [...previous, event].slice(-80);
  const next = previous.slice();
  next[index] = { ...next[index], ...event };
  return next;
}
