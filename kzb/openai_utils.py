"""Utilities for working with the local gpt-oss-20b model."""
from __future__ import annotations

import logging
from functools import lru_cache
from textwrap import dedent
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "openaccess-ai-collective/gpt-oss-20b"
DEFAULT_MAX_NEW_TOKENS = 512
EMBEDDING_MAX_LENGTH = 2048


def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def get_model() -> AutoModelForCausalLM:
    LOGGER.info("Loading model %s", MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=_select_dtype(),
        trust_remote_code=True,
    )
    model.eval()
    return model


def _generate_text(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
    tokenizer = get_tokenizer()
    model = get_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5) if temperature > 0 else 1.0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def translate_hebrew_to_english(text: str, *, temperature: float = 0.0) -> str:
    """Translate Hebrew text to English using the local model."""
    prompt = dedent(
        f"""
        You are a precise translator that converts Hebrew text to English while preserving the factual
        meaning and formatting. Translate the following Hebrew passage to natural English. Do not add
        explanations or remove structure.

        Hebrew text:
        {text}

        English translation:
        """
    ).strip()
    LOGGER.debug("Translating text of length %d", len(text))
    translation = _generate_text(
        prompt,
        temperature=temperature,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    )
    LOGGER.debug("Finished translation; output length %d", len(translation))
    return translation


def generate_contextual_answer(
    *,
    query_hebrew: str,
    query_english: str,
    context_prompt: str,
    temperature: float = 0.2,
) -> str:
    """Generate an answer in Hebrew using contextual information."""
    prompt = dedent(
        f"""
        You are a knowledgeable assistant for the Kol-Zchut knowledge base. Answer the user's question in
        Hebrew using only the provided context. Cite relevant context titles inline when helpful and avoid
        inventing facts.

        שאלה בעברית: {query_hebrew}
        English translation of the question: {query_english}

        Context from Kol-Zchut pages:
        {context_prompt}

        Provide the answer only in Hebrew, summarizing the relevant information.
        """
    ).strip()
    LOGGER.debug("Generating contextual answer for query length %d", len(query_hebrew))
    return _generate_text(
        prompt,
        temperature=temperature,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    )


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """Return embeddings for the provided texts using the local model's hidden states."""
    tokenizer = get_tokenizer()
    model = get_model()

    embeddings = []
    hidden_size = model.config.hidden_size
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=EMBEDDING_MAX_LENGTH,
            padding="max_length",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embedding = summed / counts
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        embeddings.append(embedding.squeeze(0).to("cpu", dtype=torch.float32).numpy())

    if embeddings:
        result = np.stack(embeddings, axis=0)
    else:
        result = np.zeros((0, hidden_size), dtype=np.float32)
    LOGGER.debug("Computed embeddings with shape %s", result.shape)
    return result


def embed_text(text: str) -> np.ndarray:
    """Return the embedding vector for a single text."""
    return embed_texts([text])[0]
