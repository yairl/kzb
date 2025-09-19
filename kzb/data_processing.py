"""Data processing pipeline for the Kol-Zchut dataset."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm

from .cleaning import clean_page_text
from .openai_utils import embed_text, translate_hebrew_to_english

LOGGER = logging.getLogger(__name__)


@dataclass
class Page:
    """Representation of a Kol-Zchut page."""

    identifier: str
    title: str
    url: str
    hebrew_text: str
    english_text: str
    hebrew_embedding: Sequence[float]
    english_embedding: Sequence[float]

    def to_serializable(self) -> Dict[str, object]:
        data = asdict(self)
        data["hebrew_embedding"] = list(self.hebrew_embedding)
        data["english_embedding"] = list(self.english_embedding)
        return data


def _load_raw_pages(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "pages" in data and isinstance(data["pages"], list):
            return data["pages"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("Unsupported JSON structure for Kol-Zchut pages")

    if isinstance(data, list):
        return data

    raise TypeError("Unexpected JSON data structure")


def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def translate_long_text(text: str) -> str:
    chunks = chunk_text(text)
    translated_chunks = [translate_hebrew_to_english(chunk) for chunk in chunks]
    return "\n\n".join(translated_chunks)


def _extract_identifier(page: Dict[str, object], index: int) -> str:
    for key in ("id", "identifier", "slug"):
        if key in page and isinstance(page[key], str):
            return page[key]
    return str(index)


def _extract_text(page: Dict[str, object]) -> str:
    for key in ("content", "body", "text", "article"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Unable to locate primary text field in page")


def _extract_title(page: Dict[str, object], identifier: str) -> str:
    for key in ("title", "name", "label"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"Kol-Zchut Page {identifier}"


def _extract_url(page: Dict[str, object], identifier: str) -> str:
    for key in ("url", "link", "href"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"https://www.kolzchut.org.il/{identifier}"


def process_dataset(input_path: Path, output_path: Path) -> None:
    raw_pages = list(_load_raw_pages(input_path))
    LOGGER.info("Loaded %d raw pages", len(raw_pages))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, raw_page in enumerate(tqdm(raw_pages, desc="Processing pages")):
            identifier = _extract_identifier(raw_page, index)
            title = _extract_title(raw_page, identifier)
            url = _extract_url(raw_page, identifier)
            raw_text = _extract_text(raw_page)
            hebrew_text = clean_page_text(raw_text)
            if not hebrew_text:
                LOGGER.warning("Skipping page %s due to empty cleaned text", identifier)
                continue
            english_text = translate_long_text(hebrew_text)
            hebrew_embedding = embed_text(hebrew_text)
            english_embedding = embed_text(english_text)
            page = Page(
                identifier=identifier,
                title=title,
                url=url,
                hebrew_text=hebrew_text,
                english_text=english_text,
                hebrew_embedding=hebrew_embedding,
                english_embedding=english_embedding,
            )
            output_file.write(json.dumps(page.to_serializable(), ensure_ascii=False) + "\n")

    LOGGER.info("Finished processing %d pages", len(raw_pages))


def load_processed_pages(path: Path) -> List[Page]:
    pages: List[Page] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            pages.append(
                Page(
                    identifier=data["identifier"],
                    title=data["title"],
                    url=data["url"],
                    hebrew_text=data["hebrew_text"],
                    english_text=data["english_text"],
                    hebrew_embedding=data["hebrew_embedding"],
                    english_embedding=data["english_embedding"],
                )
            )
    return pages
