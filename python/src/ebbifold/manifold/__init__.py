"""Concept manifold: embedding inputs, offline disk cache, cosine neighbourhoods.

Embeddings are inputs. Any provider that satisfies ``EmbeddingProvider`` (a
hosted API, a local model, or a precomputed artifact) can supply them; nothing
here assumes a particular vendor. Once vectors have been supplied, the disk
cache serves them offline and neighbourhood lookup is pure NumPy.
"""

from .cache import (
    CACHE_DIR_ENV,
    CachedEmbedder,
    EmbeddingCache,
    EmbeddingCacheMiss,
    cache_key,
    default_cache_dir,
)
from .embeddings import (
    EMBEDDING_DIM,
    EmbeddingError,
    EmbeddingProvider,
    PrecomputedEmbeddingProvider,
    validate_embeddings,
)
from .neighbourhood import ConceptManifold, Neighbour

__all__ = [
    "CACHE_DIR_ENV",
    "EMBEDDING_DIM",
    "CachedEmbedder",
    "ConceptManifold",
    "EmbeddingCache",
    "EmbeddingCacheMiss",
    "EmbeddingError",
    "EmbeddingProvider",
    "Neighbour",
    "PrecomputedEmbeddingProvider",
    "cache_key",
    "default_cache_dir",
    "validate_embeddings",
]
