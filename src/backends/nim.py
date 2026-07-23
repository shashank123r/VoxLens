"""NVIDIA NIM OpenAI-compatible backend using Nemotron vision-language models.

STT via Sarvam AI cloud API (accurate, fast).
TTS via Sarvam AI cloud API (fast ~1s).
Zero local GPU required — PC will not crash.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import httpx

from backends.base import ConversationSession, InferenceBackend, InferenceResult

NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_NIM_MODEL = os.environ.get(
    "NIM_MODEL",
    "nvidia/nemotron-nano-12b-v2-vl",
)

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user may show you camera images "
    "and ask questions. Reply naturally in a few short sentences. Be direct and helpful."
)


# ---------------------------------------------------------------------------
# NIM conversation (OpenAI-compatible streaming)
# ---------------------------------------------------------------------------


class NIMConversation(ConversationSession):
    """Stateful chat session against NIM chat/completions."""

    def __init__(
        self,
        client: httpx.Client,
        model: str,
        messages: list[dict[str, Any]],
    ) -> None:
        self._client = client
        self._model = model
        self._messages = list(messages)

    # -- API key helper ------------------------------------------------

    @staticmethod
    def _api_key() -> str:
        key = os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            raise RuntimeError(
                "NVIDIA_API_KEY is required when MODEL_BACKEND=nim. "
                "Get a key at https://build.nvidia.com/"
            )
        return key

    @staticmethod
    def _strip_data_url(blob: str) -> str:
        if blob.startswith("data:"):
            return blob.split(",", 1)[1]
        return blob

    # -- Send message --------------------------------------------------

    def send_message(self, message: dict) -> InferenceResult:
        user_content = self._to_openai_content(message)
        self._messages.append({"role": "user", "content": user_content})

        t0 = time.perf_counter()
        ttft: float | None = None
        chunks: list[str] = []

        with self._client.stream(
            "POST",
            f"{NIM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": self._messages,
                "max_tokens": 256,
                "temperature": 0.4,
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                delta = data.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(piece)

        decode_time = time.perf_counter() - t0
        text = "".join(chunks).strip()
        self._messages.append({"role": "assistant", "content": text})

        # Extract transcription hint injected by server
        transcription = None
        if isinstance(user_content, list):
            for part in user_content:
                t = part.get("text", "")
                if t.startswith("User said:"):
                    transcription = t.replace("User said:", "", 1).strip()
                    break

        return InferenceResult(
            text=text,
            transcription=transcription,
            ttft=round(ttft, 3) if ttft is not None else None,
            decode_time=round(decode_time, 3),
            raw={"model": self._model},
        )

    # -- Build OpenAI content ------------------------------------------

    def _to_openai_content(self, message: dict) -> str | list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        transcription_hint = message.get("_transcription")

        if transcription_hint:
            parts.append({"type": "text", "text": f"User said: {transcription_hint}"})

        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image":
                    blob = self._strip_data_url(item["blob"])
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{blob}"},
                        }
                    )
                elif item.get("type") == "audio":
                    if not transcription_hint:
                        parts.append(
                            {
                                "type": "text",
                                "text": (
                                    "The user spoke via microphone but audio could not be "
                                    "transcribed."
                                ),
                            }
                        )
        elif isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif message.get("text"):
            parts.append({"type": "text", "text": message["text"]})

        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts


# ---------------------------------------------------------------------------
# NIM backend (zero local GPU)
# ---------------------------------------------------------------------------


class NIMBackend(InferenceBackend):
    name = "nim"

    def __init__(self) -> None:
        self._client: httpx.Client | None = None
        self._model = DEFAULT_NIM_MODEL

    def load(self) -> None:
        key = os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            raise RuntimeError(
                "NVIDIA_API_KEY is required when MODEL_BACKEND=nim. "
                "Get a key at https://build.nvidia.com/"
            )
        self._client = httpx.Client(timeout=60.0)
        print(f"NIM backend ready: {self._model} @ {NIM_BASE_URL}")

    def transcribe_audio(self, audio_blob: str) -> str | None:
        """Transcribe WAV audio using Sarvam AI cloud STT API."""
        import sarvam

        return sarvam.transcribe(audio_blob, language_code="en-IN")

    def create_conversation(
        self,
        messages: list[dict],
        tools: list[Callable] | None = None,
    ) -> ConversationSession:
        if self._client is None:
            raise RuntimeError("NIM backend not loaded")
        return NIMConversation(self._client, self._model, messages)

    def supports_native_audio(self) -> bool:
        return False

    def describe(self) -> str:
        return f"NVIDIA NIM {self._model} (Sarvam STT)"
