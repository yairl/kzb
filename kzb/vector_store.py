"""Vector store helpers for retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .data_processing import Page


@dataclass
class RetrievalResult:
    page: Page
    score: float


class InMemoryVectorStore:
    """Simple in-memory vector store supporting cosine similarity."""

    def __init__(
        self,
        pages: Iterable[Page],
        *,
        use_english: bool,
        embeddings_matrix: np.ndarray | None = None,
    ) -> None:
        self.pages: List[Page] = list(pages)
        self.use_english = use_english
        self._matrix = (
            embeddings_matrix.astype(np.float32)
            if embeddings_matrix is not None
            else self._build_matrix()
        )

    def _build_matrix(self) -> np.ndarray:
        embeddings: List[Sequence[float]] = []
        for page in self.pages:
            embedding = (
                page.english_embedding if self.use_english else page.hebrew_embedding
            )
            embeddings.append(np.asarray(embedding, dtype=np.float32))
        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.vstack(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def search(self, query_embedding: Sequence[float], top_k: int = 5) -> List[RetrievalResult]:
        if self._matrix.size == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        normalized_query = query / query_norm
        scores = self._matrix @ normalized_query
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [RetrievalResult(self.pages[i], float(scores[i])) for i in top_indices]


def save_vector_index(pages: Sequence[Page], path: Path) -> None:
    """Persist normalized embedding matrices for later reuse."""

    hebrew_store = InMemoryVectorStore(pages, use_english=False)
    english_store = InMemoryVectorStore(pages, use_english=True)

    identifiers = np.array([page.identifier for page in hebrew_store.pages], dtype="U")

    np.savez_compressed(
        path,
        identifiers=identifiers,
        hebrew_embeddings=hebrew_store.matrix,
        english_embeddings=english_store.matrix,
    )


def load_vector_index(pages: Sequence[Page], path: Path) -> Tuple[InMemoryVectorStore, InMemoryVectorStore]:
    """Load precomputed embedding matrices and bind them to the provided pages."""

    data = np.load(path, allow_pickle=False)

    try:
        identifiers = data["identifiers"].astype(str).tolist()
        hebrew_matrix = data["hebrew_embeddings"].astype(np.float32)
        english_matrix = data["english_embeddings"].astype(np.float32)
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError("Embedding index is missing required keys") from exc

    identifier_to_page = {page.identifier: page for page in pages}
    ordered_pages: List[Page] = []
    missing_pages: List[str] = []
    for identifier in identifiers:
        page = identifier_to_page.get(identifier)
        if page is None:
            missing_pages.append(identifier)
        else:
            ordered_pages.append(page)

    if missing_pages:
        raise ValueError(
            "Embedding index references identifiers not present in the processed data: "
            + ", ".join(sorted(missing_pages))
        )

    if len(ordered_pages) != len(pages):
        extra_identifiers = sorted(
            set(identifier_to_page) - set(identifiers)
        )
        if extra_identifiers:
            raise ValueError(
                "Processed data includes identifiers missing from the embedding index: "
                + ", ".join(extra_identifiers)
            )

    if hebrew_matrix.shape[0] != len(ordered_pages) or english_matrix.shape[0] != len(
        ordered_pages
    ):
        raise ValueError("Embedding matrix row count does not match processed pages")

    hebrew_store = InMemoryVectorStore(
        ordered_pages, use_english=False, embeddings_matrix=hebrew_matrix
    )
    english_store = InMemoryVectorStore(
        ordered_pages, use_english=True, embeddings_matrix=english_matrix
    )
    return hebrew_store, english_store
