"""
Splits extracted page text into overlapping word-based chunks.

Each chunk remembers which page it started on. This is a simple, explainable
chunker on purpose -- ContextGuard's research value is in the evaluation
metrics, not in a fancy chunking strategy. We can swap this out later
without touching anything downstream, since every chunk is just
{chunk_id, text, page_number}.
"""
import logging
from typing import List, TypedDict

from app.services.pdf_extractor import PageText

logger = logging.getLogger(__name__)


class Chunk(TypedDict):
    chunk_id: str
    text: str
    page_number: int
    word_count: int


def chunk_document(
    pages: List[PageText],
    doc_id: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[Chunk]:
    """
    Turn a list of per-page text into overlapping chunks.

    Args:
        pages: output of pdf_extractor.extract_text_by_page
        doc_id: id of the parent document, used to build chunk_id
        chunk_size: target words per chunk
        overlap: words repeated between consecutive chunks, so an answer
                 near a chunk boundary doesn't lose context

    Returns:
        List of Chunk dicts, each tagged with its source page number.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[Chunk] = []
    chunk_index = 0

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                {
                    "chunk_id": f"{doc_id}_p{page['page_number']}_c{chunk_index}",
                    "text": chunk_text,
                    "page_number": page["page_number"],
                    "word_count": len(chunk_words),
                }
            )
            chunk_index += 1

            if end == len(words):
                break
            start = end - overlap

    logger.info("Created %d chunks for doc_id=%s", len(chunks), doc_id)
    return chunks
