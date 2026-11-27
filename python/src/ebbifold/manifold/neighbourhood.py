"""Cosine neighbourhood lookup over concept vectors in NumPy.

Replaces pgvector for the local stack. Rows are L2-normalised once in
float64, so a query is a single matrix-vector product. Ties are broken by
concept id so results are deterministic.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .embeddings import (
    EMBEDDING_DIM,
    EmbeddingError,
    EmbeddingProvider,
    validate_embeddings,
)


@dataclass(frozen=True)
class Neighbour:
    concept_id: str
    similarity: float


class ConceptManifold:
    """Immutable set of concept vectors supporting top-k cosine lookup."""

    def __init__(
        self,
        concept_ids: Sequence[str],
        vectors: NDArray[np.floating] | Sequence[Sequence[float]],
        *,
        dimensions: int = EMBEDDING_DIM,
    ) -> None:
        ids = list(concept_ids)
        if len(set(ids)) != len(ids):
            raise EmbeddingError("concept ids must be unique")
        unit = validate_embeddings(
            vectors, count=len(ids), dimensions=dimensions
        ).astype(np.float64)
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        unit.setflags(write=False)
        rank = np.empty(len(ids), dtype=np.int64)
        rank[sorted(range(len(ids)), key=ids.__getitem__)] = np.arange(len(ids))
        self._ids = tuple(ids)
        self._index = {concept_id: i for i, concept_id in enumerate(ids)}
        self._unit = unit
        groups: dict[bytes, list[int]] = {}
        for i, row in enumerate(unit):
            groups.setdefault(row.tobytes(), []).append(i)
        self._row_groups = tuple(tuple(indices) for indices in groups.values())
        self._id_rank = rank
        self.dimensions = dimensions

    @classmethod
    def from_provider(
        cls, concepts: Mapping[str, str], provider: EmbeddingProvider
    ) -> "ConceptManifold":
        """Embed ``{concept_id: text}`` via any provider (e.g. ``CachedEmbedder``)."""
        ids = list(concepts)
        vectors = provider.embed([concepts[concept_id] for concept_id in ids])
        return cls(ids, vectors, dimensions=provider.dimensions)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, concept_id: object) -> bool:
        return concept_id in self._index

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return self._ids

    def vector(self, concept_id: str) -> NDArray[np.float64]:
        """Unit-normalised vector for ``concept_id``."""
        return self._unit[self._index[concept_id]]

    def similarities(
        self, query: NDArray[np.floating] | Sequence[float]
    ) -> NDArray[np.float64]:
        """Cosine similarity of ``query`` against every concept, in input order."""
        q = np.asarray(query, dtype=np.float64).reshape(-1)
        if q.shape != (self.dimensions,):
            raise EmbeddingError(
                f"query must have shape ({self.dimensions},), got {q.shape}"
            )
        if not np.all(np.isfinite(q)):
            raise EmbeddingError("query contains NaN or infinite values")
        norm = np.linalg.norm(q)
        if norm <= 0.0:
            raise EmbeddingError("query is a zero vector; cosine is undefined")
        unit_query = q / norm
        result = np.empty(len(self._ids), dtype=np.float64)
        for indices in self._row_groups:
            score = self._unit[indices[0]] @ unit_query
            result[list(indices)] = score
        return result

    def neighbours(
        self,
        query: NDArray[np.floating] | Sequence[float],
        k: int,
        *,
        exclude: Collection[str] = (),
    ) -> list[Neighbour]:
        """Top-``k`` concepts by cosine similarity to ``query``, best first."""
        if k < 0:
            raise ValueError("k must be non-negative")
        sims = self.similarities(query)
        skip = set(exclude)
        result: list[Neighbour] = []
        for i in np.lexsort((self._id_rank, -sims)):
            if len(result) == k:
                break
            concept_id = self._ids[i]
            if concept_id not in skip:
                result.append(Neighbour(concept_id, float(sims[i])))
        return result

    def neighbours_of(self, concept_id: str, k: int) -> list[Neighbour]:
        """Top-``k`` neighbours of a known concept, excluding itself."""
        if concept_id not in self._index:
            raise KeyError(concept_id)
        return self.neighbours(self.vector(concept_id), k, exclude={concept_id})
