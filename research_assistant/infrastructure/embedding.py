"""Local text embedding adapter."""

import logging
from functools import lru_cache

import numpy as np
from research_assistant import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model '%s'", config.EMBEDDING_MODEL)
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Return float32 vectors, one per supplied text."""
    vectors = get_embedder().encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype(np.float32)
