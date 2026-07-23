"""VoxLens -- real-time multimodal AI (voice + vision).

Pipeline:
  Speech -> Sarvam STT (~1s) -> NIM LLM (~1s) -> Sarvam TTS (~1s) = ~3s E2E
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import tts
from backends import create_backend
from backends.base import InferenceBackend
from backends.nim import NIMBackend
from latency import LatencyRecord, LatencyTracker, new_request_id
from narrate import SceneChangeDetector

load_dotenv()

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "Reply naturally in a few short sentences. Be direct and helpful."
)

NARRATE_PROMPT = (
    "You are monitoring a live camera feed. Describe what changed since the last frame "
    "in 1-2 short sentences suitable for spoken narration. Focus on meaningful changes only."
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

inference_backend: InferenceBackend | None = None
tts_backend = None
latency_tracker = LatencyTracker()
narrate_interval_s = float(os.environ.get("NARRATE_INTERVAL_S", "5"))
narrate_ssim_threshold = float(os.environ.get("NARRATE_SSIM_THRESHOLD", "0.85"))


# ---------------------------------------------------------------------------
# TTS Engine -- Sarvam cloud (fast) or Kokoro local (fallback)
# ---------------------------------------------------------------------------

class _TTSEngine:
    """Wrapper around active TTS backend."""

    def __init__(self) -> None:
        self._kokoro = None
        self.sample_rate: int = 24000
        self._use_sarvam = bool(os.environ.get("SARVAM_API_KEY", ""))

    def load(self) -> None:
        if self._use_sarvam:
            import sarvam
            _ = sarvam.get_api_key()  # validate early
            self.sample_rate = 22050
            print("TTS: Sarvam AI cloud (22050 Hz, ~1s latency)")
        else:
            self._kokoro = tts.load()
            self.sample_rate = self._kokoro.sample_rate
            print(f"TTS: Kokoro CPU ({self.sample_rate} Hz)")

    def generate(self, text: str, voice: str = "anushka") -> np.ndarray:
        if self._use_sarvam:
            import sarvam
            result = sarvam.synthesize(text, speaker=voice, pace=1.0)
            if result is not None:
                audio, _ = result
                return audio
            return np.zeros(int(self.sample_rate * 2), dtype=np.float32)
        if self._kokoro is not None:
            return self._kokoro.generate(text)
        return np.zeros(int(self.sample_rate * 2), dtype=np.float32)

    @property
    def name(self) -> str:
        return "sarvam" if self._use_sarvam else "kokoro"


def load_models() -> None:
    global inference_backend, tts_backend
    inference_backend = create_backend()
    inference_backend.load()
    engine = _TTSEngine()
    engine.load()
    tts_backend = engine
    print(f"Inference: {inference_backend.describe()}")
    print(f"TTS: {engine.name} @ {tts_backend.sample_rate} Hz")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.get("/health")
async def health():
    tts_name = tts_backend.name if tts_backend else None
    return JSONResponse(
        {
            "status": "ok" if inference_backend and tts_backend else "loading",
            "backend": inference_backend.name if inference_backend else None,
            "backend_description": inference_backend.describe() if inference_backend else None,
            "tts": tts_name,
            "tts_sample_rate": tts_backend.sample_rate if tts_backend else None,
            "narrate_interval_s": narrate_interval_s,
        }
    )


# ---------------------------------------------------------------------------
# Build message content for LLM
# ---------------------------------------------------------------------------

def build_user_content(msg: dict, transcription: str | None = None) -> list[dict]:
    content: list[dict] = []
    if msg.get("audio"):
        content.append({"type": "audio", "blob": msg["audio"]})
    if msg.get("image"):
        content.append({"type": "image", "blob": msg["image"]})

    if msg.get("narrate"):
        content.append(
            {
                "type": "text",
                "text": (
                    "The camera scene changed. Describe what changed in natural language "
                    "for spoken narration."
                ),
            }
        )
    elif msg.get("audio") and msg.get("image"):
        content.append(
            {
                "type": "text",
                "text": (
                    "The user just spoke to you (audio) while showing their camera (image). "
                    "Respond to what they said, referencing what you see if relevant."
                ),
            }
        )
    elif msg.get("audio"):
        content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
    elif msg.get("image"):
        content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
    else:
        content.append({"type": "text", "text": msg.get("text", "Hello!")})

    if transcription:
        content.append({"type": "text", "text": f"(Transcribed speech: {transcription})"})
    return content


# ---------------------------------------------------------------------------
# Inference + TTS streaming
# ---------------------------------------------------------------------------

async def run_inference(session, msg: dict, record: LatencyRecord):
    record.inference_start_ts = time.perf_counter()

    user_message = {"role": "user", "content": build_user_content(msg)}
    if isinstance(inference_backend, NIMBackend) and msg.get("audio"):
        transcription = await asyncio.get_event_loop().run_in_executor(
            None, inference_backend.transcribe_audio, msg["audio"]
        )
        if transcription:
            user_message = {
                "role": "user",
                "content": build_user_content(msg, transcription=transcription),
                "_transcription": transcription,
            }

    t0 = time.perf_counter()
    result = await asyncio.get_event_loop().run_in_executor(
        None, session.send_message, user_message
    )
    llm_time = time.perf_counter() - t0
    record.llm_time_s = round(llm_time, 3)
    if result.ttft is not None:
        record.first_token_ts = record.inference_start_ts + result.ttft
    else:
        record.first_token_ts = time.perf_counter()
    record.text_complete_ts = time.perf_counter()
    record.response_chars = len(result.text)
    return result


async def stream_tts(ws: WebSocket, text: str, record: LatencyRecord, interrupted: asyncio.Event):
    """Stream TTS audio to the client. Uses Sarvam (fast) or Kokoro (fallback)."""
    sentences = split_sentences(text) or [text]
    tts_start = time.perf_counter()

    await ws.send_text(
        json.dumps(
            {
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
                "tts_engine": tts_backend.name,
            }
        )
    )

    first_audio = True
    for i, sentence in enumerate(sentences):
        if interrupted.is_set():
            break
        try:
            pcm = await asyncio.get_event_loop().run_in_executor(
                None, lambda s=sentence: tts_backend.generate(s)
            )
        except Exception as e:
            print(f"TTS generation error for sentence {i}: {e}")
            break
        if interrupted.is_set():
            break
        if first_audio:
            record.first_audio_ts = time.perf_counter()
            first_audio = False
        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
        try:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                        "index": i,
                    }
                )
            )
        except Exception as e:
            print(f"WebSocket send error during TTS streaming: {e}")
            break

    tts_time = time.perf_counter() - tts_start
    record.tts_time_s = round(tts_time, 3)
    if not interrupted.is_set():
        await ws.send_text(json.dumps({"type": "audio_end", "tts_time": round(tts_time, 2)}))


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    session = inference_backend.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
    )
    session.__enter__()

    interrupted = asyncio.Event()
    msg_queue: asyncio.Queue = asyncio.Queue()
    narrate_enabled = False
    scene_detector = SceneChangeDetector(threshold=narrate_ssim_threshold)

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                elif msg.get("type") == "narrate_config":
                    nonlocal narrate_enabled
                    narrate_enabled = bool(msg.get("enabled", False))
                    if not narrate_enabled:
                        scene_detector.reset()
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "narrate_status",
                                "enabled": narrate_enabled,
                                "interval_s": narrate_interval_s,
                            }
                        )
                    )
                elif msg.get("type") == "narrate_tick" and narrate_enabled:
                    image = msg.get("image")
                    if not image:
                        continue
                    changed, score = scene_detector.should_narrate(image)
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "narrate_probe",
                                "changed": changed,
                                "ssim": round(score, 4),
                            }
                        )
                    )
                    if changed:
                        await msg_queue.put({"image": image, "narrate": True})
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()
            record = LatencyRecord(
                request_id=new_request_id(),
                backend=inference_backend.name,
                mode="narrate" if msg.get("narrate") else "interactive",
                capture_ts=time.perf_counter(),
                has_audio=bool(msg.get("audio")),
                has_image=bool(msg.get("image")),
                narrate=bool(msg.get("narrate")),
            )

            if msg.get("narrate"):
                session.__exit__(None, None, None)
                session = inference_backend.create_conversation(
                    messages=[{"role": "system", "content": NARRATE_PROMPT}]
                )
                session.__enter__()

            if tts_backend is None:
                print("TTS backend not loaded, skipping audio response")
                record.complete_ts = time.perf_counter()
                latency_tracker.persist(record)
                continue

            result = await run_inference(session, msg, record)

            print(
                f"LLM ({record.llm_time_s}s) [{inference_backend.name}] "
                f"heard: {result.transcription!r} -> {result.text}"
            )

            if interrupted.is_set():
                continue

            reply = {
                "type": "text",
                "text": result.text,
                "llm_time": record.llm_time_s,
                "backend": inference_backend.name,
                "narrate": msg.get("narrate", False),
                "tts": tts_backend.name,
            }
            if result.transcription:
                reply["transcription"] = result.transcription
            if result.ttft is not None:
                reply["ttft"] = result.ttft
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                continue

            await stream_tts(ws, result.text, record, interrupted)
            record.complete_ts = time.perf_counter()
            latency_tracker.persist(record)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        session.__exit__(None, None, None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
