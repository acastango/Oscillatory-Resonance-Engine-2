# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT INTERFACES
# Design: A3 (ML Integration) + I1 (Systems Architect)
# Implementation: I4 (Integration)
# ═══════════════════════════════════════════════════════════════════════════════

"""
A3: "ORE needs to talk to real LLMs. But we can't hardcode Claude or Ollama.
Abstract the client interface so any backend plugs in."

I1: "ABC for the contract, concrete clients for the popular providers,
and a mock client for testing without API keys."
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate completion.

        Args:
            prompt: The current user message.
            system_prompt: System-level instructions.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            messages: Optional conversation history as a list of
                {"role": "user"|"assistant", "content": "..."} dicts.
                When provided, prompt is appended as the final user message.
                When None, only prompt is sent (single-turn).
        """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding."""


class ClaudeClient(LLMClient):
    """
    Anthropic Claude client.

    Requires: pip install anthropic
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            ) from exc

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        # Build messages list: history + current prompt
        if messages:
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]
            api_messages.append({"role": "user", "content": prompt})
        else:
            api_messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=api_messages,
        )
        return response.content[0].text

    def embed(self, text: str) -> np.ndarray:
        # Claude doesn't have native embeddings; use a separate service
        raise NotImplementedError(
            "Claude does not provide embeddings. Use a separate embedding "
            "service (e.g. voyage-ai, OpenAI ada-002)."
        )


class OllamaClient(LLMClient):
    """
    Local Ollama client.

    Requires: Ollama running at base_url, plus the requests package.
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        import requests

        if messages:
            # Use /api/chat for multi-turn conversations
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            for m in messages:
                chat_messages.append({"role": m["role"], "content": m["content"]})
            chat_messages.append({"role": "user", "content": prompt})

            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": chat_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        else:
            # Single-turn: use /api/generate
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["response"]

    def embed(self, text: str) -> np.ndarray:
        import requests

        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"])


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API keys.

    - complete() returns deterministic responses based on input hash.
    - embed() returns deterministic 1536-dim unit vectors from text hash.
    - All calls are recorded for test inspection.
    """

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.call_log: list = []

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        self.call_log.append({
            "method": "complete",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        })

        # Deterministic response based on prompt + conversation context
        hash_input = prompt
        if messages:
            hash_input = '|'.join(m['content'] for m in messages) + '|' + prompt
        seed = int(hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16)
        word_count = max(5, min(max_tokens // 10, 50))
        rng = np.random.RandomState(seed % (2**31))

        # Generate plausible-length response
        words = [
            "The", "analysis", "suggests", "that", "this", "approach",
            "could", "work", "well", "given", "the", "current", "context",
            "and", "available", "evidence", "from", "multiple", "sources",
            "indicating", "a", "positive", "outcome", "for", "consideration",
        ]
        response_words = [words[rng.randint(len(words))] for _ in range(word_count)]
        return " ".join(response_words)

    def embed(self, text: str) -> np.ndarray:
        self.call_log.append({
            "method": "embed",
            "text": text,
        })

        # Deterministic embedding from text hash
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)
        emb = rng.randn(self.embedding_dim)
        return emb / np.linalg.norm(emb)
