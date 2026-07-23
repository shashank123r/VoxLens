"""Per-request latency instrumentation and persistence."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LatencyRecord:
    request_id: str
    backend: str
    mode: str = "interactive"
    capture_ts: float | None = None
    inference_start_ts: float | None = None
    first_token_ts: float | None = None
    text_complete_ts: float | None = None
    first_audio_ts: float | None = None
    complete_ts: float | None = None
    llm_time_s: float | None = None
    tts_time_s: float | None = None
    ttft_s: float | None = None
    ttfa_s: float | None = None
    total_s: float | None = None
    has_audio: bool = False
    has_image: bool = False
    narrate: bool = False
    response_chars: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        if self.capture_ts and self.complete_ts:
            self.total_s = round(self.complete_ts - self.capture_ts, 3)
        if self.inference_start_ts and self.first_token_ts:
            self.ttft_s = round(self.first_token_ts - self.inference_start_ts, 3)
        if self.capture_ts and self.first_audio_ts:
            self.ttfa_s = round(self.first_audio_ts - self.capture_ts, 3)

    def log_line(self) -> str:
        self.finalize()
        return (
            f"latency backend={self.backend} mode={self.mode} "
            f"ttft={self.ttft_s}s ttfa={self.ttfa_s}s llm={self.llm_time_s}s "
            f"tts={self.tts_time_s}s total={self.total_s}s "
            f"audio={self.has_audio} image={self.has_image} narrate={self.narrate} "
            f"chars={self.response_chars}"
        )


class LatencyTracker:
    def __init__(self) -> None:
        out_dir = Path(os.environ.get("LATENCY_DIR", "latency_logs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "latency.csv"
        self.jsonl_path = out_dir / "latency.jsonl"
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "request_id",
                    "backend",
                    "mode",
                    "ttft_s",
                    "ttfa_s",
                    "llm_time_s",
                    "tts_time_s",
                    "total_s",
                    "has_audio",
                    "has_image",
                    "narrate",
                    "response_chars",
                ],
            )
            writer.writeheader()

    def persist(self, record: LatencyRecord) -> None:
        record.finalize()
        row = {
            "request_id": record.request_id,
            "backend": record.backend,
            "mode": record.mode,
            "ttft_s": record.ttft_s,
            "ttfa_s": record.ttfa_s,
            "llm_time_s": record.llm_time_s,
            "tts_time_s": record.tts_time_s,
            "total_s": record.total_s,
            "has_audio": record.has_audio,
            "has_image": record.has_image,
            "narrate": record.narrate,
            "response_chars": record.response_chars,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, **record.extra}) + "\n")
        print(record.log_line())


def new_request_id() -> str:
    return f"{int(time.time() * 1000)}"
