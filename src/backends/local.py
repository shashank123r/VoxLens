"""Local on-device Gemma 4 E2B via LiteRT-LM."""

from __future__ import annotations

import os
import re
from typing import Callable

import litert_lm

from backends.base import ConversationSession, InferenceBackend, InferenceResult

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"

STRIP_RE = re.compile(r'<\|"\|>')


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def _strip(text: str) -> str:
    return STRIP_RE.sub("", text).strip()


class LocalConversation(ConversationSession):
    def __init__(self, engine: litert_lm.Engine, messages: list[dict], tools: list[Callable]):
        self._tool_result: dict[str, str] = {}
        self._respond = self._make_responder()
        tool_list = [self._respond, *(tools or [])]
        self._conversation = engine.create_conversation(messages=messages, tools=tool_list)
        self._conversation.__enter__()

    def _make_responder(self) -> Callable:
        backend = self

        def respond_to_user(transcription: str, response: str) -> str:
            backend._tool_result["transcription"] = transcription
            backend._tool_result["response"] = response
            return "OK"

        respond_to_user.__doc__ = (
            "Respond to the user's voice message.\n\n"
            "Args:\n"
            "    transcription: Exact transcription of what the user said in the audio.\n"
            "    response: Your conversational response to the user. Keep it to 1-4 short sentences."
        )
        return respond_to_user

    def send_message(self, message: dict) -> InferenceResult:
        self._tool_result.clear()
        response = self._conversation.send_message(message)

        if self._tool_result:
            return InferenceResult(
                text=_strip(self._tool_result.get("response", "")),
                transcription=_strip(self._tool_result.get("transcription", "")) or None,
                raw={"tool": True},
            )

        text = response["content"][0]["text"]
        return InferenceResult(text=_strip(text), raw={"tool": False, "response": response})

    def __exit__(self, exc_type, exc, tb) -> None:
        self._conversation.__exit__(exc_type, exc, tb)


class LocalGemmaBackend(InferenceBackend):
    name = "local"

    def __init__(self) -> None:
        self._engine: litert_lm.Engine | None = None
        self._model_path: str | None = None

    @property
    def model_path(self) -> str:
        if self._model_path is None:
            self._model_path = resolve_model_path()
        return self._model_path

    def load(self) -> None:
        print(f"Loading Gemma 4 E2B from {self.model_path}...")
        self._engine = litert_lm.Engine(
            self.model_path,
            backend=litert_lm.Backend.GPU(),
            vision_backend=litert_lm.Backend.GPU(),
            audio_backend=litert_lm.Backend.CPU(),
        )
        self._engine.__enter__()
        print("Local Gemma engine loaded.")

    def create_conversation(
        self,
        messages: list[dict],
        tools: list[Callable] | None = None,
    ) -> ConversationSession:
        if self._engine is None:
            raise RuntimeError("Local backend not loaded")
        return LocalConversation(self._engine, messages, tools or [])

    def supports_native_audio(self) -> bool:
        return True

    def describe(self) -> str:
        return f"Gemma 4 E2B (LiteRT-LM) @ {self.model_path}"
