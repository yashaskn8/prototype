"""Canonical passage rendering for the legacy Op27 *research* pipeline.

Both training examples and Op27's isolated Chroma index MUST call this helper.
The pipeline is separate from the modern BGE/Qwen research export and production
retriever. It does not modify any source documents or collections.
"""


def op27_passage_text(chunk, *, content_first=False):
    heading = (chunk.heading or "").strip()
    body = chunk.text or ""
    if content_first:
        return f"{heading}\n{body}".strip() if heading and heading not in body else body

    document = chunk.document
    parts = filter(None, (
        document.title, document.description, document.tags, document.doc_type,
        getattr(document.domain, "name", "") if document.domain else "",
        getattr(document.category, "name", "") if document.category else "",
        getattr(document.product, "name", "") if document.product else "",
        getattr(document.asset, "name", "") if document.asset else "",
        getattr(document.asset, "model_number", "") if document.asset else "",
    ))
    metadata = " ".join(str(p) for p in parts)
    return f"{metadata} {heading}\n{body}".strip()
