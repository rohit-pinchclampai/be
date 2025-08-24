from typing import Tuple
from pypdf import PdfReader
import docx

def load_pdf(fp) -> str:
    reader = PdfReader(fp)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_docx(fp) -> str:
    d = docx.Document(fp)
    return "\n".join(p.text for p in d.paragraphs)

def sniff_and_load(filename: str, file_bytes: bytes) -> Tuple[str, str]:
    name = filename.lower()
    if name.endswith(".pdf"):
        from io import BytesIO
        return load_pdf(BytesIO(file_bytes)), "pdf"
    if name.endswith(".docx"):
        from io import BytesIO
        return load_docx(BytesIO(file_bytes)), "docx"
    # treat as plain text
    return file_bytes.decode("utf-8", errors="ignore"), "txt"
