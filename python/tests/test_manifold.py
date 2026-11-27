"""Phase 3b manifold gate tests.

All vectors are synthetic (seeded NumPy RNG): no embedding provider,
credential, or network is involved. Neighbourhoods are checked against a
pure-Python cosine reference, and the disk cache is checked for byte-level
determinism and offline operation after first use.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ebbifold.manifold import (
    CACHE_DIR_ENV,
    EMBEDDING_DIM,
    CachedEmbedder,
    ConceptManifold,
    EmbeddingCache,
    EmbeddingCacheMiss,
    EmbeddingError,
    EmbeddingProvider,
    PrecomputedEmbeddingProvider,
    cache_key,
    default_cache_dir,
)

MODEL = "synthetic-test-model@1"
TOL = 1e-9


def synthetic_vectors(
    count: int, seed: int = 0, dim: int = EMBEDDING_DIM
) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((count, dim)).astype(np.float32)


def reference_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    return dot / (na * nb)


def reference_neighbours(
    ids: Sequence[str], vectors: np.ndarray, query: Sequence[float], k: int, exclude=()
) -> list[tuple[str, float]]:
    q = [float(x) for x in query]
    scored = [
        (concept_id, reference_cosine([float(x) for x in row], q))
        for concept_id, row in zip(ids, vectors, strict=True)
        if concept_id not in exclude
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:k]


class CountingProvider:
    """Deterministic fake provider: vector seeded from the text's cache key."""

    def __init__(self, model_id: str = MODEL, dimensions: int = EMBEDDING_DIM) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = [
            np.random.default_rng(
                int(cache_key(self.model_id, self.dimensions, t)[:8], 16)
            ).standard_normal(self.dimensions)
            for t in texts
        ]
        return np.array(rows).reshape(len(texts), self.dimensions)


class ExplodingProvider(CountingProvider):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise AssertionError("provider must not be called when the cache is warm")


# --- neighbourhoods -------------------------------------------------------


@pytest.fixture(scope="module")
def manifold_data() -> tuple[list[str], np.ndarray]:
    ids = [f"concept-{i:03d}" for i in range(64)]
    return ids, synthetic_vectors(len(ids), seed=3)


def test_dimension_is_3072() -> None:
    assert EMBEDDING_DIM == 3072


@pytest.mark.parametrize("k", [0, 1, 5, 64, 100])
def test_neighbours_match_reference_cosine(manifold_data, k: int) -> None:
    ids, vectors = manifold_data
    manifold = ConceptManifold(ids, vectors)
    query = synthetic_vectors(1, seed=99)[0]
    got = manifold.neighbours(query, k)
    expected = reference_neighbours(ids, vectors, query, k)
    assert [n.concept_id for n in got] == [cid for cid, _ in expected]
    for neighbour, (_, sim) in zip(got, expected, strict=True):
        assert neighbour.similarity == pytest.approx(sim, abs=TOL)


def test_neighbours_of_excludes_self_and_matches_reference(manifold_data) -> None:
    ids, vectors = manifold_data
    manifold = ConceptManifold(ids, vectors)
    for concept_id in ids[:8]:
        got = manifold.neighbours_of(concept_id, 10)
        row = vectors[ids.index(concept_id)]
        expected = reference_neighbours(ids, vectors, row, 10, exclude={concept_id})
        assert concept_id not in [n.concept_id for n in got]
        assert [n.concept_id for n in got] == [cid for cid, _ in expected]
        assert [n.similarity for n in got] == pytest.approx(
            [sim for _, sim in expected], abs=TOL
        )


def test_similarities_match_reference_for_all_concepts(manifold_data) -> None:
    ids, vectors = manifold_data
    manifold = ConceptManifold(ids, vectors)
    query = synthetic_vectors(1, seed=7)[0]
    expected = [reference_cosine(row.tolist(), query.tolist()) for row in vectors]
    np.testing.assert_allclose(manifold.similarities(query), expected, atol=TOL, rtol=0)


def test_scale_invariance_and_self_similarity(manifold_data) -> None:
    ids, vectors = manifold_data
    manifold = ConceptManifold(ids, vectors)
    scaled = ConceptManifold(ids, vectors * 17.5)
    query = vectors[5]
    top, top_scaled = (
        manifold.neighbours(query, 1)[0],
        scaled.neighbours(query * 0.01, 1)[0],
    )
    assert top.concept_id == top_scaled.concept_id
    assert top.similarity == pytest.approx(top_scaled.similarity, abs=TOL)
    assert manifold.neighbours(query, 1)[0].concept_id == ids[5]
    assert manifold.neighbours(query, 1)[0].similarity == pytest.approx(1.0, abs=TOL)


def test_ties_break_by_concept_id_deterministically() -> None:
    dim = 4
    base = np.array([1.0, 0.0, 0.0, 0.0])
    ids = ["zeta", "alpha", "mu", "beta"]
    vectors = np.stack([base, base * 2, np.array([0.0, 1.0, 0.0, 0.0]), base * 3])
    manifold = ConceptManifold(ids, vectors, dimensions=dim)
    got = manifold.neighbours(base, 4)
    assert [n.concept_id for n in got] == ["alpha", "beta", "zeta", "mu"]
    assert manifold.neighbours(base, 2, exclude={"alpha"})[0].concept_id == "beta"
    reversed_manifold = ConceptManifold(list(reversed(ids)), vectors[::-1].copy(), dimensions=dim)
    assert [n.concept_id for n in reversed_manifold.neighbours(base, 4)] == [
        n.concept_id for n in got
    ]


def test_distinct_vectors_with_equal_cosine_keep_input_order_independent_ties() -> None:
    ids = ["zeta", "alpha"]
    vectors = np.array([[1.0, 1.0], [1.0, -1.0]])
    query = np.array([1.0, 0.0])
    forward = ConceptManifold(ids, vectors, dimensions=2)
    backward = ConceptManifold(ids[::-1], vectors[::-1].copy(), dimensions=2)
    assert [n.concept_id for n in forward.neighbours(query, 2)] == ["alpha", "zeta"]
    assert [n.concept_id for n in backward.neighbours(query, 2)] == ["alpha", "zeta"]


def test_manifold_accessors(manifold_data) -> None:
    ids, vectors = manifold_data
    manifold = ConceptManifold(ids, vectors)
    assert len(manifold) == 64
    assert ids[0] in manifold and "nope" not in manifold
    assert manifold.concept_ids == tuple(ids)
    assert np.linalg.norm(manifold.vector(ids[0])) == pytest.approx(1.0, abs=TOL)
    with pytest.raises(KeyError):
        manifold.neighbours_of("nope", 3)
    with pytest.raises(ValueError):
        manifold.neighbours(vectors[0], -1)


@pytest.mark.parametrize(
    ("ids", "vectors", "match"),
    [
        (["a", "a"], np.ones((2, 4)), "unique"),
        (["a"], np.ones((1, 3)), "shape"),
        (["a"], np.array([[1.0, np.nan, 0.0, 0.0]]), "NaN"),
        (["a"], np.array([[1e39, 0.0, 0.0, 0.0]]), "float32 range"),
        (["a"], np.array([[1e-50, 0.0, 0.0, 0.0]]), "zero vector"),
        (["a"], np.zeros((1, 4)), "zero vector"),
    ],
)
def test_manifold_rejects_bad_vectors(ids, vectors, match: str) -> None:
    with pytest.raises(EmbeddingError, match=match):
        ConceptManifold(ids, vectors, dimensions=4)


@pytest.mark.parametrize(
    ("query", "match"),
    [([1.0, 0.0], "shape"), ([np.inf, 0, 0, 0], "NaN"), ([0.0] * 4, "zero vector")],
)
def test_query_validation(query, match: str) -> None:
    manifold = ConceptManifold(["a"], np.ones((1, 4)), dimensions=4)
    with pytest.raises(EmbeddingError, match=match):
        manifold.similarities(query)


# --- providers ------------------------------------------------------------


def test_precomputed_provider_round_trip() -> None:
    vectors = synthetic_vectors(3, seed=1)
    provider = PrecomputedEmbeddingProvider(
        {"a": vectors[0], "b": vectors[1], "c": vectors[2]}, model_id=MODEL
    )
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_id == MODEL and provider.dimensions == EMBEDDING_DIM
    np.testing.assert_array_equal(provider.embed(["c", "a"]), vectors[[2, 0]])
    assert provider.embed([]).shape == (0, EMBEDDING_DIM)
    with pytest.raises(KeyError):
        provider.embed(["missing"])


def test_precomputed_provider_validation() -> None:
    with pytest.raises(EmbeddingError, match="model_id"):
        PrecomputedEmbeddingProvider({}, model_id="")
    with pytest.raises(EmbeddingError, match="shape"):
        PrecomputedEmbeddingProvider({"a": [1.0, 2.0]}, model_id=MODEL)
    empty = PrecomputedEmbeddingProvider({}, model_id=MODEL, dimensions=4)
    assert empty.embed([]).shape == (0, 4)


# --- disk cache -----------------------------------------------------------


def test_cache_is_offline_after_first_use(tmp_path: Path) -> None:
    texts = ["Krebs cycle", "beta-oxidation", "Krebs cycle", "urea cycle"]
    provider = CountingProvider()
    cache = EmbeddingCache(tmp_path, model_id=MODEL, dimensions=EMBEDDING_DIM)
    warm = CachedEmbedder(cache, provider).embed(texts)
    assert provider.calls == [["Krebs cycle", "beta-oxidation", "urea cycle"]]
    np.testing.assert_array_equal(warm[0], warm[2])

    # Fresh objects (as in a new process) with no usable provider.
    cache = EmbeddingCache(tmp_path, model_id=MODEL, dimensions=EMBEDDING_DIM)
    offline = CachedEmbedder(cache, None).embed(texts)
    assert offline.tobytes() == warm.tobytes()
    flagged = CachedEmbedder(cache, ExplodingProvider(), offline=True).embed(texts)
    assert flagged.tobytes() == warm.tobytes()
    assert (
        CachedEmbedder(cache, ExplodingProvider()).embed(texts).tobytes()
        == warm.tobytes()
    )


def test_cache_miss_offline_raises(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model_id=MODEL, dimensions=EMBEDDING_DIM)
    with pytest.raises(EmbeddingCacheMiss):
        CachedEmbedder(cache, None).embed(["never seen"])
    provider = CountingProvider()
    with pytest.raises(EmbeddingCacheMiss):
        CachedEmbedder(cache, provider, offline=True).embed(["never seen"])
    assert provider.calls == []


def test_cache_bytes_are_deterministic(tmp_path: Path) -> None:
    vector = synthetic_vectors(1, seed=5)[0]
    a = EmbeddingCache(tmp_path / "a", model_id=MODEL, dimensions=EMBEDDING_DIM)
    b = EmbeddingCache(tmp_path / "b", model_id=MODEL, dimensions=EMBEDDING_DIM)
    path_a, path_b = a.put("text", vector), b.put("text", vector)
    assert path_a.relative_to(tmp_path / "a") == path_b.relative_to(tmp_path / "b")
    assert path_a.read_bytes() == path_b.read_bytes()
    manifest_a = (a.directory / "manifest.json").read_bytes()
    assert manifest_a == (b.directory / "manifest.json").read_bytes()
    a.put("text", vector)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert not list(a.directory.rglob(".tmp-*"))


def test_cache_key_separates_models_dimensions_and_texts() -> None:
    keys = {
        cache_key(MODEL, 3072, "x"),
        cache_key("other-model", 3072, "x"),
        cache_key(MODEL, 1024, "x"),
        cache_key(MODEL, 3072, "y"),
    }
    assert len(keys) == 4
    first = cache_key(MODEL, 3072, "x")
    assert first == cache_key(MODEL, 3072, "x")


def test_cache_spaces_are_isolated(tmp_path: Path) -> None:
    vector = synthetic_vectors(1, seed=2)[0]
    EmbeddingCache(tmp_path, model_id=MODEL, dimensions=EMBEDDING_DIM).put("t", vector)
    other = EmbeddingCache(tmp_path, model_id="other-model", dimensions=EMBEDDING_DIM)
    assert other.get("t") is None


def test_cache_rejects_tampered_manifest_and_entries(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model_id=MODEL, dimensions=4)
    path = cache.put("t", [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        cache.get("t"), np.array([1, 2, 3, 4], dtype=np.float32)
    )
    np.save(path, np.ones(3, dtype="<f4"))
    with pytest.raises(EmbeddingError, match="corrupt"):
        cache.get("t")
    np.save(path, np.array([np.nan, 0, 0, 0], dtype="<f4"))
    with pytest.raises(EmbeddingError, match="corrupt"):
        cache.get("t")
    np.save(path, np.zeros(4, dtype="<f4"))
    with pytest.raises(EmbeddingError, match="corrupt"):
        cache.get("t")
    (cache.directory / "manifest.json").write_text(
        '{"model_id": "x"}', encoding="utf-8"
    )
    with pytest.raises(EmbeddingError, match="manifest"):
        cache.get("t")
    with pytest.raises(EmbeddingError, match="manifest"):
        cache.put("u", [1.0, 0.0, 0.0, 0.0])


def test_cache_and_embedder_validation(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="model_id"):
        EmbeddingCache(tmp_path, model_id="", dimensions=4)
    with pytest.raises(EmbeddingError, match="dimensions"):
        EmbeddingCache(tmp_path, model_id=MODEL, dimensions=0)
    cache = EmbeddingCache(tmp_path, model_id=MODEL, dimensions=EMBEDDING_DIM)
    with pytest.raises(EmbeddingError, match="disagree"):
        CachedEmbedder(cache, CountingProvider(model_id="other"))
    with pytest.raises(EmbeddingError, match="zero vector"):
        cache.put("z", np.zeros(EMBEDDING_DIM))

    class WrongShape(CountingProvider):
        def embed(self, texts: Sequence[str]) -> np.ndarray:
            return np.ones((len(texts), 3))

    with pytest.raises(EmbeddingError, match="shape"):
        CachedEmbedder(cache, WrongShape()).embed(["a"])
    assert cache.get("a") is None


def test_default_cache_dir_is_outside_repo(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "override"))
    assert default_cache_dir() == tmp_path / "override"
    monkeypatch.delenv(CACHE_DIR_ENV)
    repo_root = Path(__file__).resolve().parents[2]
    assert not default_cache_dir().resolve().is_relative_to(repo_root)


def test_manifold_from_cached_provider_offline(tmp_path: Path) -> None:
    concepts = {f"c{i}": f"concept text {i}" for i in range(12)}
    cache = EmbeddingCache(tmp_path, model_id=MODEL, dimensions=EMBEDDING_DIM)
    online = ConceptManifold.from_provider(
        concepts, CachedEmbedder(cache, CountingProvider())
    )
    offline = ConceptManifold.from_provider(concepts, CachedEmbedder(cache, None))
    assert online.neighbours_of("c0", 5) == offline.neighbours_of("c0", 5)
    vectors = CachedEmbedder(cache).embed(list(concepts.values()))
    expected = reference_neighbours(
        list(concepts), vectors, vectors[0], 5, exclude={"c0"}
    )
    assert [n.concept_id for n in offline.neighbours_of("c0", 5)] == [
        c for c, _ in expected
    ]


def test_failed_cache_write_leaves_no_partial_files(
    tmp_path: Path, monkeypatch
) -> None:
    cache = EmbeddingCache(tmp_path, model_id=MODEL, dimensions=4)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", boom)
    with pytest.raises(OSError, match="disk full"):
        cache.put("t", [1.0, 0.0, 0.0, 0.0])
    assert cache.get("t") is None
    assert not list(cache.directory.rglob(".tmp-*"))
