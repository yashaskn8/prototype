import re


def chunk_markdownish(text, max_chars=1100, overlap=140):
    """Chunk text while preserving simple section headings for citations."""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    lines = text.split("\n")
    sections = []
    heading = ""
    buffer = []
    for line in lines:
        stripped = line.strip()
        is_heading = bool(re.match(r"^#{1,4}\s+", stripped)) or (
            stripped and len(stripped) < 90 and stripped.endswith(":")
        )
        if is_heading:
            if buffer:
                sections.append((heading, "\n".join(buffer).strip()))
                buffer = []
            heading = re.sub(r"^#{1,4}\s+", "", stripped).rstrip(":")
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))
    if not sections:
        sections = [("", text)]

    chunks = []
    for heading, body in sections:
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not body:
            continue
        if len(body) <= max_chars:
            chunks.append((heading, body))
            continue
        start = 0
        while start < len(body):
            end = min(len(body), start + max_chars)
            cut = body[start:end]
            if end < len(body):
                split_at = max(cut.rfind("\n\n"), cut.rfind(". "))
                if split_at > max_chars * 0.55:
                    end = start + split_at + 1
                    cut = body[start:end]
            chunks.append((heading, cut.strip()))
            if end >= len(body):
                break
            start = max(start + 1, end - overlap)
    return chunks
