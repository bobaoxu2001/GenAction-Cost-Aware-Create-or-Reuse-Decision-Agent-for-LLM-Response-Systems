"""Lightweight, reproducible text embeddings.

Two backends are supported:

* ``sentence-transformers`` (if installed *and* a model can be loaded) -- better
  semantic similarity, especially for paraphrases.
* ``tfidf`` -- a scikit-learn TF-IDF fallback that needs no downloads, no API
  keys, and is fully deterministic. This is the default so the project runs
  out-of-the-box and offline.

The ``"auto"`` backend prefers sentence-transformers and silently falls back to
TF-IDF if it is unavailable.

Note on the TF-IDF backend: the vectoriser is *fit once* on the full known
corpus (FAQ + incoming stream). This makes the embedding space fixed and
reproducible. It is a deliberate simplification of a fully online setting
(vocabulary is assumed known up front); created actions reuse query text that is
already in the corpus, so no out-of-vocabulary drift occurs during a run.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

_DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation with a guard against zero vectors."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class Embedder:
    """Encode text into fixed, L2-normalised vectors.

    Parameters
    ----------
    backend:
        ``"auto"`` (default), ``"tfidf"`` or ``"sentence-transformers"``.
    seed:
        Kept for API symmetry / future stochastic backends. TF-IDF is
        deterministic and does not use it.
    st_model:
        Sentence-transformers model name (only used by that backend).
    """

    def __init__(
        self,
        backend: str = "auto",
        seed: int = 0,
        st_model: str = _DEFAULT_ST_MODEL,
    ) -> None:
        self.seed = seed
        self.st_model_name = st_model
        self._fitted = False
        self._vectorizer = None  # TF-IDF
        self._st_model = None  # sentence-transformers
        self.backend = self._resolve_backend(backend)

    # ------------------------------------------------------------------ #
    # backend resolution
    # ------------------------------------------------------------------ #
    def _resolve_backend(self, backend: str) -> str:
        if backend == "tfidf":
            return "tfidf"
        if backend in ("auto", "sentence-transformers"):
            model = self._try_load_st()
            if model is not None:
                self._st_model = model
                return "sentence-transformers"
            if backend == "sentence-transformers":
                raise RuntimeError(
                    "sentence-transformers backend requested but the package or "
                    "model could not be loaded."
                )
            return "tfidf"
        raise ValueError(f"Unknown embedding backend: {backend!r}")

    def _try_load_st(self):
        try:  # pragma: no cover - exercised only when the package is present
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self.st_model_name)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def fit(self, corpus: Iterable[str]) -> "Embedder":
        """Fit the embedder on a text corpus (no-op for sentence-transformers)."""
        corpus = list(corpus)
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
                norm="l2",
            )
            self._vectorizer.fit(corpus)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        """Return an ``(n, dim)`` array of L2-normalised embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)

        if self.backend == "tfidf":
            if not self._fitted or self._vectorizer is None:
                raise RuntimeError("Embedder.fit must be called before encode().")
            mat = self._vectorizer.transform(texts).toarray()
            return _l2_normalize(mat)

        # sentence-transformers
        emb = self._st_model.encode(texts, convert_to_numpy=True)
        return _l2_normalize(emb)

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience: encode a single string into a 1-D vector."""
        return self.encode([text])[0]

    @property
    def is_fitted(self) -> bool:
        return self._fitted or self.backend == "sentence-transformers"
