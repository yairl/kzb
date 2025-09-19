"""Core RAG pipeline for the Kol-Zchut bot."""
from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .data_processing import Page, load_processed_pages
from .openai_utils import (
    embed_text,
    generate_contextual_answer,
    translate_hebrew_to_english,
)
from .vector_store import (
    InMemoryVectorStore,
    RetrievalResult,
    load_vector_index,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievedAnswer:
    results: List[RetrievalResult]
    query_embedding: Sequence[float]


class KolZchutBot:
    """Main interface for preparing data and answering queries."""

    def __init__(self, processed_data_path: Path, *, embeddings_path: Path | None = None) -> None:
        self.processed_data_path = processed_data_path
        self.embeddings_path = embeddings_path
        self._pages: List[Page] | None = None
        self._hebrew_store: InMemoryVectorStore | None = None
        self._english_store: InMemoryVectorStore | None = None
        self._attempted_index_load = False

    @property
    def pages(self) -> List[Page]:
        if self._pages is None:
            LOGGER.info("Loading processed pages from %s", self.processed_data_path)
            self._pages = load_processed_pages(self.processed_data_path)
        return self._pages

    @property
    def hebrew_store(self) -> InMemoryVectorStore:
        if self._hebrew_store is None:
            self._ensure_vector_indexes_loaded()
            if self._hebrew_store is None:
                self._hebrew_store = InMemoryVectorStore(self.pages, use_english=False)
        return self._hebrew_store

    @property
    def english_store(self) -> InMemoryVectorStore:
        if self._english_store is None:
            self._ensure_vector_indexes_loaded()
            if self._english_store is None:
                self._english_store = InMemoryVectorStore(self.pages, use_english=True)
        return self._english_store

    def _ensure_vector_indexes_loaded(self) -> None:
        if self.embeddings_path is None or self._attempted_index_load:
            return

        self._attempted_index_load = True
        if not self.embeddings_path.exists():
            LOGGER.warning(
                "Embedding index path %s does not exist; falling back to in-memory build",
                self.embeddings_path,
            )
            return

        try:
            hebrew_store, english_store = load_vector_index(self.pages, self.embeddings_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Failed to load embedding index from %s: %s", self.embeddings_path, exc)
            return

        self._hebrew_store = hebrew_store
        self._english_store = english_store

    def _retrieve(
        self, store: InMemoryVectorStore, query_embedding: Sequence[float], top_k: int
    ) -> RetrievedAnswer:
        return RetrievedAnswer(
            results=store.search(query_embedding, top_k=top_k),
            query_embedding=query_embedding,
        )

    def retrieve_relevant_pages(
        self, query_hebrew: str, *, top_k: int = 5
    ) -> tuple[RetrievedAnswer, RetrievedAnswer, str]:
        english_query = translate_hebrew_to_english(query_hebrew)
        hebrew_embedding = embed_text(query_hebrew)
        english_embedding = embed_text(english_query)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            hebrew_future = executor.submit(
                self._retrieve, self.hebrew_store, hebrew_embedding, top_k
            )
            english_future = executor.submit(
                self._retrieve, self.english_store, english_embedding, top_k
            )
            hebrew_results = hebrew_future.result()
            english_results = english_future.result()

        return hebrew_results, english_results, english_query

    def build_answer(
        self,
        query_hebrew: str,
        query_english: str,
        hebrew_results: RetrievedAnswer,
        english_results: RetrievedAnswer,
    ) -> str:
        context_blocks: List[str] = []
        combined_results: dict[str, RetrievalResult] = {}
        for result in hebrew_results.results:
            combined_results[result.page.identifier] = result
        for result in english_results.results:
            existing = combined_results.get(result.page.identifier)
            if existing is None or result.score > existing.score:
                combined_results[result.page.identifier] = result

        sorted_results = sorted(
            combined_results.values(), key=lambda item: item.score, reverse=True
        )[:5]

        for idx, result in enumerate(sorted_results, start=1):
            page = result.page
            context_blocks.append(
                f"### מקור {idx}: {page.title}\n"
                f"קישור: {page.url}\n"
                f"תוכן עברית:\n{page.hebrew_text[:1200]}\n\n"
                f"Translation:\n{page.english_text[:1200]}"
            )

        context_prompt = "\n\n".join(context_blocks)
        LOGGER.debug("Constructed context prompt with %d sources", len(sorted_results))

        return generate_contextual_answer(
            query_hebrew=query_hebrew,
            query_english=query_english,
            context_prompt=context_prompt,
        )

    def answer_query(self, query_hebrew: str, *, top_k: int = 5) -> str:
        hebrew_results, english_results, english_query = self.retrieve_relevant_pages(
            query_hebrew, top_k=top_k
        )
        return self.build_answer(
            query_hebrew, english_query, hebrew_results, english_results
        )
