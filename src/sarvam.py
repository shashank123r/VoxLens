"""Sarvam AI API client for Speech-to-Text and Text-to-Speech.

Uses the same API key for both services (api-subscription-key header).
STT: https://api.sarvam.ai/speech-to-text
TTS: https://api.sarvam.ai/text-to-speech (bulbul:v2, speaker: anushka)
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import httpx
import numpy as np


def get_api_key() -> str:
    """Get Sarvam API key from environment."""
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        raise RuntimeError("SARVAM_API_KEY environment variable is required")
    return key


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

def transcribe(audio_blob: str, language_code: str = "en-IN") -> str | None:
    """Transcribe a base64 WAV audio blob using Sarvam STT API.

    Args:
        audio_blob: Base64-encoded WAV audio (16 kHz mono, as sent by frontend).
        language_code: Language code (e.g. 'en-IN', 'hi-IN').

    Returns:
        Transcribed text, or None on failure.
    """
    key = get_api_key()
    headers = {"api-subscription-key": key}

    try:
        raw = base64.b64decode(audio_blob)
        buf = io.BytesIO(raw)

        files = {"file": ("audio.wav", buf, "audio/wav")}
        data = {"language_code": language_code}

        t0 = time.perf_counter()
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers=headers,
                files=files,
                data=data,
            )
        stt_time = time.perf_counter() - t0

        if r.status_code != 200:
            print(f"Sarvam STT error ({r.status_code}): {r.text[:200]}")
            return None

        result = r.json()
        transcript = result.get("transcript", "") or result.get("text", "")
        print(f"Sarvam STT ({stt_time:.2f}s): {transcript!r}")
        return transcript.strip() or None

    except Exception as exc:
        print(f"Sarvam STT exception: {exc}")
        return None


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

# Valid speakers: anushka, abhilash, manisha, vidya, arya, karun, hitesh, aditya,
# ritu, priya, neha, rahul, pooja, rohan, simran, kavya, amit, dev, ishita,
# shreya, ratan, varun, manan, sumit, roopa, kabir, aayan, shubh, ashutosh,
# advait, anand, tanya, tarun, sunny, mani, gokul, vijay, shruti, suhani,
# mohit, kavitha, rehan, soham, rupali

DEFAULT_SPEAKER = "anushka"
DEFAULT_MODEL = "bulbul:v2"
TARGET_SAMPLE_RATE = 22050  # Sarvam TTS outputs 22050 Hz WAV


def synthesize(
    text: str,
    speaker: str = DEFAULT_SPEAKER,
    pace: float = 1.0,
) -> tuple[np.ndarray, int] | None:
    """Synthesize speech from text using Sarvam TTS API.

    Args:
        text: Text to speak.
        speaker: Voice name (default: 'anushka').
        pace: Speaking pace (1.0 = normal).

    Returns:
        Tuple of (audio_samples as float32 numpy array, sample_rate).
        Returns None on failure.
    """
    key = get_api_key()
    headers = {"api-subscription-key": key}

    try:
        t0 = time.perf_counter()
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers=headers,
                json={
                    "inputs": [text],
                    "target_language_code": "en-IN",
                    "speaker": speaker,
                    "pace": pace,
                    "model": DEFAULT_MODEL,
                },
            )
        tts_time = time.perf_counter() - t0

        if r.status_code != 200:
            print(f"Sarvam TTS error ({r.status_code}): {r.text[:200]}")
            return None

        data = r.json()
        audios = data.get("audios", [])
        if not audios:
            print("Sarvam TTS: no audio returned")
            return None

        audio_bytes = base64.b64decode(audios[0])
        audio = (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )
        sr = TARGET_SAMPLE_RATE

        print(f"Sarvam TTS ({tts_time:.2f}s): {len(audio) / sr:.1f}s audio")
        return audio, sr

    except Exception as exc:
        print(f"Sarvam TTS exception: {exc}")
        return None
