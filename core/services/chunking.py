import re


def _find_split_point(cut, min_pos):
    # 1. Paragraph boundary
    p_split = cut.rfind("\n\n")
    if p_split > min_pos:
        return p_split + 2
    # 2. Numbered step, bullet, or table row boundary (\n followed by list marker or table pipe)
    list_matches = list(re.finditer(r"\n(?=(?:\d+[\.\)]|\*|-)\s|\|)", cut))
    if list_matches:
        last_match_start = list_matches[-1].start()
        if last_match_start > min_pos:
            return last_match_start + 1
    # 3. Standard line break (preserves table rows and single lines)
    nl_split = cut.rfind("\n")
    if nl_split > min_pos:
        return nl_split + 1
    # 4. Sentence boundary
    s_split = cut.rfind(". ")
    if s_split > min_pos:
        return s_split + 2
    return -1


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
        min_split_ratio = 0.55
        while start < len(body):
            end = min(len(body), start + max_chars)
            cut = body[start:end]
            if end < len(body):
                split_at = _find_split_point(cut, int(max_chars * min_split_ratio))
                if split_at != -1:
                    end = start + split_at
                    cut = body[start:end]
            chunks.append((heading, cut.strip()))
            if end >= len(body):
                break
            start = max(start + 1, end - overlap)
            # Align start forward to the next line boundary if close
            if start < end:
                next_nl = body.find("\n", start, min(len(body), start + max(1, overlap // 2) + 1))
                if next_nl != -1 and next_nl + 1 < end:
                    start = next_nl + 1
    return chunks
