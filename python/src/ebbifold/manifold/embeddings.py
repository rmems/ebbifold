"""Provider-agnostic embedding interface.

The scheduler never calls a vendor directly. A provider is anything with a
stable ``model_id``, a fixed ``dimensions`` and an ``embed`` method; swapping
providers (or replaying a precomputed artifact) changes no other code.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Concept vectors are 3,072-dimensional, matching the upstream corpus.
EMBEDDING_DIM = 3072


class EmbeddingError(ValueError):
    """Raised when a provider returns vectors that violate the contract."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Source of embedding vectors for text.

    ``model_id`` must change whenever the vectors would change (model, version,
    preprocessing): it is part of the cache key.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> NDArray[np.floating]:
        """Return one row per text, shape ``(len(texts), dimensions)``."""
        ...


def validate_embeddings(
    vectors: object, *, count: int, dimensions: int
) -> NDArray[np.float32]:
    """Coerce provider output to float32 and reject malformed vectors."""
    array = np.asarray(vectors, dtype=np.float64)
    if array.shape != (count, dimensions):
        raise EmbeddingError(
            f"expected embeddings of shape ({count}, {dimensions}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise EmbeddingError("embeddings contain NaN or infinite values")
    if count and np.any(np.linalg.norm(array, axis=1) <= 0.0):
        raise EmbeddingError("embeddings contain a zero vector; cosine is undefined")
    return array.astype(np.float32)


class PrecomputedEmbeddingProvider:
    """Serve vectors from a precomputed text-to-vector artifact.

    Useful for replaying exported embeddings and for tests: no network, no
    credentials. Unknown texts raise ``KeyError`` rather than inventing a
    vector.
    """

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float] | NDArray[np.floating]],
        *,
        model_id: str,
        dimensions: int = EMBEDDING_DIM,
    ) -> None:
        if not model_id:
            raise EmbeddingError("model_id must be non-empty")
        texts = list(vectors)
        matrix = validate_embeddings(
            [vectors[text] for text in texts] if texts else np.empty((0, dimensions)),
            count=len(texts),
            dimensions=dimensions,
        )
        self._rows = {text: matrix[i] for i, text in enumerate(texts)}
        self._model_id = model_id
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        missing = [text for text in texts if text not in self._rows]
        if missing:
            raise KeyError(f"no precomputed embedding for {len(missing)} text(s)")
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)
        return np.stack([self._rows[text] for text in texts])
