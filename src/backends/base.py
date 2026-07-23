"""Inference backend abstraction for local Gemma and NVIDIA NIM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InferenceResult:
    """Normalized inference output consumed by the WebSocket handler."""

    text: str
    transcription: str | None = None
    ttft: float | None = None
    decode_tokens: int | None = None
    decode_time: float | None = None
    tok_per_sec: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ConversationSession(ABC):
    """Per-connection conversation handle."""

    @abstractmethod
    def send_message(self, message: dict) -> InferenceResult:
        raise NotImplementedError

    def __enter__(self) -> ConversationSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class InferenceBackend(ABC):
    """Swappable vision+language backend."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_conversation(
        self,
        messages: list[dict],
        tools: list[Callable] | None = None,
    ) -> ConversationSession:
        raise NotImplementedError

    @abstractmethod
    def supports_native_audio(self) -> bool:
        """True when the backend ingests raw audio without a separate STT step."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
