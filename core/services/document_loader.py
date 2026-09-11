from pathlib import Path
import csv
import zipfile

from django.conf import settings

MAX_TEXT_CHARS = getattr(settings, "SERVY_MAX_DOCUMENT_TEXT_CHARS", 500_000)
MAX_PDF_PAGES = getattr(settings, "SERVY_MAX_PDF_PAGES", 120)
MAX_CSV_ROWS = getattr(settings, "SERVY_MAX_CSV_ROWS", 10_000)
MAX_DOCX_UNCOMPRESSED = getattr(settings, "SERVY_MAX_DOCX_UNCOMPRESSED_BYTES", 40 * 1024 * 1024)


class UnsafeDocumentError(ValueError):
    pass


def _limit(text):
    return (text or "")[:MAX_TEXT_CHARS]


def _validate_docx_archive(path):
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if len(infos) > 500:
                raise UnsafeDocumentError("The DOCX archive contains too many internal files.")
            total = sum(i.file_size for i in infos)
            if total > MAX_DOCX_UNCOMPRESSED:
                raise UnsafeDocumentError("DOCX expands beyond the configured safety limit.")
            for item in infos:
                if item.filename.startswith("/") or ".." in Path(item.filename).parts:
                    raise UnsafeDocumentError("Unsafe DOCX archive path detected.")
    except UnsafeDocumentError:
        raise
    except Exception as exc:
        raise UnsafeDocumentError("Invalid or corrupted DOCX archive.") from exc


def extract_document_text(document):
    """Extract bounded text locally. No remote URL/video fetching is performed."""
    if document.content_text and document.content_text.strip():
        return _limit(document.content_text.strip())
    if not document.file:
        return ""

    path = Path(document.file.path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".log"}:
        return _limit(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            for i, row in enumerate(csv.reader(f)):
                if i >= MAX_CSV_ROWS:
                    break
                rows.append(" | ".join(str(c)[:2000] for c in row))
        return _limit("\n".join(rows))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        try:
            reader = PdfReader(str(path), strict=False)
            if len(reader.pages) > MAX_PDF_PAGES:
                raise UnsafeDocumentError(f"PDF has more than {MAX_PDF_PAGES} pages.")
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
                if sum(len(x) for x in text) >= MAX_TEXT_CHARS:
                    break
            return _limit("\n\n".join(text))
        except UnsafeDocumentError:
            raise
        except Exception as exc:
            raise UnsafeDocumentError("Failed to parse PDF document.") from exc
    if suffix == ".docx":
        _validate_docx_archive(path)
        try:
            from docx import Document
        except ImportError:
            return ""
        try:
            doc = Document(str(path))
            parts = []
            for p in doc.paragraphs:
                parts.append(p.text)
                if sum(len(x) for x in parts) >= MAX_TEXT_CHARS:
                    break
            return _limit("\n".join(parts))
        except Exception as exc:
            raise UnsafeDocumentError("Failed to parse DOCX document.") from exc
    return ""
