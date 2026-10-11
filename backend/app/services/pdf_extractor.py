"""
Extracts text from a PDF, page by page, using PyMuPDF (fitz).

We keep page boundaries because ContextGuard needs to attribute every
retrieved chunk back to a page number for source citation later.
"""
import logging
from pathlib import Path
from typing import List, TypedDict

import pymupdf  # PyMuPDF (the `fitz` import name is deprecated)

logger = logging.getLogger(__name__)


class PageText(TypedDict):
    page_number: int  # 1-indexed, matches what a human sees in a PDF viewer
    text: str


def extract_text_by_page(file_path: str | Path) -> List[PageText]:
    """
    Open a PDF and return a list of {page_number, text} dicts, one per page.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        ValueError: if the file can't be opened as a PDF or has no pages.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    try:
        doc = pymupdf.open(file_path)
    except Exception as exc:
        raise ValueError(f"Could not open '{file_path}' as a PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF '{file_path}' has no pages.")

    pages: List[PageText] = []
    for i, page in enumerate(doc):
        raw_text = page.get_text("text")
        cleaned = _clean_text(raw_text)
        pages.append({"page_number": i + 1, "text": cleaned})

    doc.close()
    logger.info("Extracted %d pages from %s", len(pages), file_path.name)
    return pages


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace left behind by PDF extraction."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
