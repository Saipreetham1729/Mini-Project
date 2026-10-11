"""
Wraps a sentence-transformers model to turn text into vectors.

The model loads once (lru_cache) and stays in memory. Embeddings are
L2-normalized so cosine similarity can be computed as a plain dot product,
which is what ChromaDB's "cosine" space expects.
"""
import logging
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s (first load may take a moment)", settings.embedding_model_name)
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Returns one vector per input text, same order."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
