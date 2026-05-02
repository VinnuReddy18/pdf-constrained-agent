from __future__ import annotations

import re
from pathlib import Path
from typing import List

from pypdf import PdfReader

from .types import Chunk


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def _detect_section(page_text: str) -> str | None:
    patterns = [
        r"\b(Topic\s+\d+\s+[A-Z][A-Za-z0-9 &:/,\-]+?)(?=\s+\d+\s+[A-Z]|\s*$)",
        r"\b(Chapter\s+\d+\s+[A-Z][A-Za-z0-9 &:/,\-]+?)(?=\s+\d+\s+[A-Z]|\s*$)",
        r"\b(Section\s+\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 &:/,\-]+?)(?=\s+\d+\s+[A-Z]|\s*$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, page_text)
        if match:
            return match.group(1).strip()
    return None


def load_pdf_text_per_page(pdf_path: str) -> List[str]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(_normalize_text(text))
    return pages


def build_chunks(pages: List[str], chunk_size: int = 900, overlap: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        section = _detect_section(page_text)
        start = 0
        part = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            text = page_text[start:end].strip()
            if text:
                chunk_id = f"p{page_idx}_c{part}"
                chunks.append(Chunk(chunk_id=chunk_id, page=page_idx, text=text, section=section))
            if end == len(page_text):
                break
            start = max(end - overlap, start + 1)
            part += 1
    return chunks
