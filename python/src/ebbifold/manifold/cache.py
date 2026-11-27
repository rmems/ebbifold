"""Content-addressed on-disk embedding cache.

Each vector lives in its own ``.npy`` file keyed by SHA-256 of
``(model_id, dimensions, text)``, so the same input always maps to the same
path and the same bytes. After a vector has been supplied once, lookups need
neither the provider nor the network.

The default location is a per-user cache directory, never the repository:
cached vectors and learner memory are runtime artifacts, not source.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from .embeddings import EmbeddingError, EmbeddingProvider, validate_embeddings

CACHE_DIR_ENV = "EBBIFOLD_CACHE_DIR"
CACHE_FORMAT_VERSION = 1
# Little-endian float32 so cache bytes are identical across platforms.
_DTYPE = np.dtype("<f4")


class EmbeddingCacheMiss(LookupError):
    """Raised offline when a text has no cached vector and no provider may run."""


def default_cache_dir() -> Path:
    """``$EBBIFOLD_CACHE_DIR``, else the platform's per-user cache directory."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "ebbifold" / "cache" / "embeddings"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ebbifold" / "embeddings"


def cache_key(model_id: str, dimensions: int, text: str) -> str:
    """Stable hex digest identifying one (model, dimensions, text) vector."""
    payload = json.dumps(
        [CACHE_FORMAT_VERSION, model_id, dimensions, text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_dir_name(model_id: str, dimensions: int) -> str:
    digest = hashlib.sha256(f"{model_id}\0{dimensions}".encode()).hexdigest()
    return digest[:16]


class EmbeddingCache:
    """Read/write vectors for a single ``(model_id, dimensions)`` space."""

    def __init__(self, root: Path | str, *, model_id: str, dimensions: int) -> None:
        if not model_id:
            raise EmbeddingError("model_id must be non-empty")
        if dimensions <= 0:
            raise EmbeddingError("dimensions must be positive")
        self.model_id = model_id
        self.dimensions = dimensions
        self.directory = Path(root) / _model_dir_name(model_id, dimensions)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "model_id": self.model_id,
            "dimensions": self.dimensions,
            "dtype": _DTYPE.str,
        }

    def _manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def path_for(self, text: str) -> Path:
        key = cache_key(self.model_id, self.dimensions, text)
        return self.directory / key[:2] / f"{key}.npy"

    def _check_manifest(self) -> None:
        path = self._manifest_path()
        if not path.exists():
            return
        found = json.loads(path.read_text(encoding="utf-8"))
        if found != self.manifest:
            raise EmbeddingError(f"cache manifest mismatch at {path}: {found!r}")

    def get(self, text: str) -> NDArray[np.float32] | None:
        path = self.path_for(text)
        if not path.exists():
            return None
        self._check_manifest()
        vector = np.load(path, allow_pickle=False)
        if vector.dtype != _DTYPE or vector.shape != (self.dimensions,):
            raise EmbeddingError(f"corrupt cache entry {path}")
        return vector.astype(np.float32, copy=False)

    def put(self, text: str, vector: NDArray[np.floating]) -> Path:
        row = validate_embeddings(
            np.asarray(vector).reshape(1, -1), count=1, dimensions=self.dimensions
        )[0].astype(_DTYPE)
        self._ensure_manifest()
        path = self.path_for(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, lambda handle: np.save(handle, row, allow_pickle=False))
        return path

    def _ensure_manifest(self) -> None:
        path = self._manifest_path()
        if path.exists():
            self._check_manifest()
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        body = json.dumps(self.manifest, sort_keys=True, indent=2) + "\n"
        _atomic_write(path, lambda handle: handle.write(body.encode("utf-8")))


def _atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class CachedEmbedder:
    """Cache-first embedding lookup with an optional provider for misses.

    Satisfies ``EmbeddingProvider`` itself. With ``provider=None`` or
    ``offline=True`` it never calls out: every text must already be cached,
    otherwise ``EmbeddingCacheMiss``.
    """

    def __init__(
        self,
        cache: EmbeddingCache,
        provider: EmbeddingProvider | None = None,
        *,
        offline: bool = False,
    ) -> None:
        if provider is not None and (
            provider.model_id != cache.model_id
            or provider.dimensions != cache.dimensions
        ):
            raise EmbeddingError(
                "provider and cache disagree on model_id/dimensions: "
                f"{provider.model_id}/{provider.dimensions} vs "
                f"{cache.model_id}/{cache.dimensions}"
            )
        self.cache = cache
        self.provider = provider
        self.offline = offline

    @property
    def model_id(self) -> str:
        return self.cache.model_id

    @property
    def dimensions(self) -> int:
        return self.cache.dimensions

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        cached = {text: self.cache.get(text) for text in dict.fromkeys(texts)}
        missing = [text for text, vector in cached.items() if vector is None]
        if missing:
            cached.update(zip(missing, self._fetch(missing), strict=True))
        result = np.empty((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            result[i] = cached[text]
        return result

    def _fetch(self, missing: list[str]) -> list[NDArray[np.float32]]:
        if self.provider is None or self.offline:
            raise EmbeddingCacheMiss(
                f"{len(missing)} text(s) not cached and no provider may run offline"
            )
        vectors = validate_embeddings(
            self.provider.embed(missing), count=len(missing), dimensions=self.dimensions
        )
        for text, vector in zip(missing, vectors, strict=True):
            self.cache.put(text, vector)
        # Return what was written so first-use results equal later cache hits.
        return [vector.astype("<f4").astype(np.float32) for vector in vectors]
