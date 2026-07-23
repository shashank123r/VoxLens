"""Comprehensive latency benchmark — runs 20+ iterations per backend and produces statistics.

Usage:
    # Local backend (server must be running with MODEL_BACKEND=local):
    uv run python benchmarks/latency_bench.py --backend local --runs 20

    # NIM backend (server must be running with MODEL_BACKEND=nim):
    uv run python benchmarks/latency_bench.py --backend nim --runs 20

    # Both backends sequentially:
    uv run python benchmarks/latency_bench.py --runs 20

Measures: TTFT, TTFA, LLM time, TTS time, total time.
Produces: Mean, median, p95, min, max per metric + comparison table.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import websockets


SERVER_URL = os.environ.get("SERVER_URL", "ws://localhost:8000/ws")


@dataclass
class BenchResult:
    backend: str
    mode: str = "interactive"
    ttft_s: float | None = None
    ttfa_s: float | None = None
    llm_time_s: float = 0.0
    tts_time_s: float | None = None
    total_s: float = 0.0
    text_recv_s: float = 0.0
    response_chars: int = 0
    audio_kb: float = 0.0
    text: str = ""
    transcription: str | None = None
    has_audio: bool = False
    has_image: bool = False
    error: str | None = None


# ── Test fixtures ──────────────────────────────────────────────────────────

def make_wav_b64(duration_s: float, sample_rate: int = 16000) -> str:
    """Create a WAV file as base64 string (440Hz sine tone)."""
    samples = np.sin(2 * np.pi * 440 * np.arange(int(sample_rate * duration_s)) / sample_rate)
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def make_jpg_b64(width: int = 320, height: int = 240) -> str:
    """Create a JPEG image as base64 string."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Test payloads ──────────────────────────────────────────────────────────

FIXTURES = {
    "text_only": {"text": "Tell me a fun fact about the ocean."},
    "text_question": {"text": "What is the capital of France and what is its most famous landmark?"},
    "text_creative": {"text": "Write a short poem about artificial intelligence."},
    "text_joke": {"text": "Tell me a short joke about computers."},
    "text_fact": {"text": "What is the speed of light and who discovered it?"},
    "image_only": {},  # image added per-iteration
    "image_describe": {},  # image added per-iteration
    "image_text": {},  # image + text
    "audio_short": {},  # audio added per-iteration
    "audio_image": {},  # audio + image
}

TEXT_PROMPTS = [
    "Tell me a fun fact about the ocean.",
    "What is the capital of France and what is its most famous landmark?",
    "Write a short poem about artificial intelligence.",
    "Tell me a short joke about computers.",
    "What is the speed of light and who discovered it?",
    "Explain quantum computing in simple terms.",
    "What are the health benefits of drinking water?",
    "Tell me about the history of the internet.",
    "What is machine learning?",
    "How do batteries work?",
]


# ── WebSocket client ──────────────────────────────────────────────────────

async def send_and_receive(ws, payload: dict) -> dict:
    """Send a message and collect all responses until audio_end."""
    t0 = time.perf_counter()
    await ws.send(json.dumps(payload))

    text_msg = None
    text_recv_ts = None
    audio_bytes = 0
    first_audio_ts = None
    tts_time = None
    completed = False

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=60)
        msg = json.loads(raw)

        if msg["type"] == "text":
            text_msg = msg
            text_recv_ts = time.perf_counter()
        elif msg["type"] == "audio_chunk":
            if first_audio_ts is None:
                first_audio_ts = time.perf_counter()
            audio_bytes += len(msg.get("audio", ""))
        elif msg["type"] == "audio_start":
            pass
        elif msg["type"] == "audio_end":
            tts_time = msg.get("tts_time")
            completed = True
            break

    total_s = time.perf_counter() - t0
    text_recv_s = (text_recv_ts - t0) if text_recv_ts else total_s
    ttfa_s = (first_audio_ts - t0) if first_audio_ts else None

    return {
        "text": text_msg.get("text", "") if text_msg else "",
        "transcription": text_msg.get("transcription") if text_msg else None,
        "llm_time": text_msg.get("llm_time", 0) if text_msg else 0,
        "ttft": text_msg.get("ttft"),
        "ttfa": ttfa_s,
        "total_s": round(total_s, 4),
        "text_recv_s": round(text_recv_s, 4),
        "tts_time": tts_time,
        "completed": completed,
        "audio_kb": round(audio_bytes * 3 / 4 / 1024, 1),
    }


# ── Statistics ────────────────────────────────────────────────────────────

def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "count": 0}
    sorted_v = sorted(values)
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(sorted_v[int(len(sorted_v) * 0.95)], 3) if len(values) > 1 else round(sorted_v[-1], 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "count": len(values),
    }


def print_stats(name: str, stats_dict: dict, unit: str = "s"):
    if stats_dict["count"] == 0:
        print(f"  {name:<20}  no data")
        return
    print(
        f"  {name:<20}  mean={stats_dict['mean']:<6.3f}{unit}  "
        f"median={stats_dict['median']:<6.3f}{unit}  "
        f"p95={stats_dict['p95']:<6.3f}{unit}  "
        f"min={stats_dict['min']:<6.3f}{unit}  "
        f"max={stats_dict['max']:<6.3f}{unit}  "
        f"(n={stats_dict['count']})"
    )


# ── Benchmark runner ──────────────────────────────────────────────────────

async def run_benchmark(backend: str, num_runs: int = 20) -> list[BenchResult]:
    """Run num_runs iterations against the running server."""
    image = make_jpg_b64()
    results: list[BenchResult] = []

    print(f"\n{'='*70}")
    print(f"  BENCHMARK: backend={backend}, iterations={num_runs}")
    print(f"{'='*70}")
    print(f"  Server: {SERVER_URL}")

    for i in range(num_runs):
        prompt = TEXT_PROMPTS[i % len(TEXT_PROMPTS)]

        # Alternate between text-only and text+image every 3 iterations
        if i % 3 == 0:
            payload = {"text": prompt, "image": image}
            has_image = True
            has_audio = False
            mode = "text_image"
        elif i % 3 == 1:
            payload = {"text": prompt}
            has_image = False
            has_audio = False
            mode = "text_only"
        else:
            payload = {"text": prompt, "image": image}
            has_image = True
            has_audio = False
            mode = "text_image"

        result = BenchResult(
            backend=backend,
            mode=mode,
            has_audio=has_audio,
            has_image=has_image,
        )

        try:
            async with websockets.connect(SERVER_URL, ping_timeout=30) as ws:
                r = await send_and_receive(ws, payload)
                result.ttft_s = r.get("ttft")
                result.ttfa_s = r.get("ttfa")
                result.llm_time_s = r.get("llm_time", 0)
                result.tts_time_s = r.get("tts_time")
                result.total_s = r.get("total_s", 0)
                result.text_recv_s = r.get("text_recv_s", 0)
                result.response_chars = len(r.get("text", ""))
                result.audio_kb = r.get("audio_kb", 0)
                result.text = r.get("text", "")[:80]
                result.transcription = r.get("transcription")

                if r.get("completed"):
                    status = "OK"
                else:
                    status = "NO_AUDIO"

                llm_str = f"LLM={r['llm_time']:.2f}s"
                ttft_str = f"TTFT={r['ttft']}s" if r['ttft'] else "TTFT=N/A"
                ttfa_str = f"TTFA={r['ttfa']:.2f}s" if r['ttfa'] else "TTFA=N/A"
                total_str = f"TOTAL={r['total_s']:.2f}s"
                print(f"  [{i+1:>2}/{num_runs}] {status:<8} {mode:<14} {llm_str} {ttft_str} {ttfa_str} {total_str}  '{r['text'][:40]}'")

        except Exception as e:
            result.error = str(e)
            print(f"  [{i+1:>2}/{num_runs}] ERROR: {str(e)}")

        results.append(result)

    return results


def aggregate_results(results: list[BenchResult]) -> dict:
    """Aggregate statistics from benchmark results."""
    ttft_vals = [r.ttft_s for r in results if r.ttft_s is not None and r.error is None]
    ttfa_vals = [r.ttfa_s for r in results if r.ttfa_s is not None and r.error is None]
    llm_vals = [r.llm_time_s for r in results if r.error is None]
    tts_vals = [r.tts_time_s for r in results if r.tts_time_s is not None and r.error is None]
    total_vals = [r.total_s for r in results if r.error is None]
    text_recv_vals = [r.text_recv_s for r in results if r.error is None]

    return {
        "ttft": compute_stats(ttft_vals),
        "ttfa": compute_stats(ttfa_vals),
        "llm_time": compute_stats(llm_vals),
        "tts_time": compute_stats(tts_vals),
        "total": compute_stats(total_vals),
        "text_recv": compute_stats(text_recv_vals),
        "total_runs": len(results),
        "successful_runs": sum(1 for r in results if r.error is None),
        "failed_runs": sum(1 for r in results if r.error is not None),
    }


def print_aggregate(backend: str, agg: dict):
    print(f"\n{'-'*70}")
    print(f"  RESULTS: backend={backend}")
    print(f"  Total runs: {agg['total_runs']}, Successful: {agg['successful_runs']}, Failed: {agg['failed_runs']}")
    print(f"{'-'*70}")
    print_stats("TTFT (first token)", agg["ttft"])
    print_stats("TTFA (first audio)", agg["ttfa"])
    print_stats("LLM time", agg["llm_time"])
    print_stats("TTS time", agg["tts_time"])
    print_stats("Text recv time", agg["text_recv"])
    print_stats("Total E2E time", agg["total"])


def print_comparison(local_agg: dict | None, nim_agg: dict | None):
    """Print a markdown comparison table."""
    print(f"\n{'='*70}")
    print("  LOCAL vs NIM LATENCY COMPARISON")
    print(f"{'='*70}")

    metrics = [
        ("TTFT (s)", "ttft"),
        ("TTFA (s)", "ttfa"),
        ("LLM Time (s)", "llm_time"),
        ("TTS Time (s)", "tts_time"),
        ("Total E2E (s)", "total"),
    ]

    header = f"| {'Metric':<20} | {'Local Mean':<12} | {'Local p95':<12} | {'NIM Mean':<12} | {'NIM p95':<12} | {'Speedup':<10} |"
    sep = f"| {'-'*20} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10} |"
    print(f"\n{header}\n{sep}")

    for label, key in metrics:
        l = local_agg.get(key, {}) if local_agg else {}
        n = nim_agg.get(key, {}) if nim_agg else {}
        lm = f"{l.get('mean', 'N/A'):<6.3f}" if l.get('count', 0) > 0 else "N/A"
        lp = f"{l.get('p95', 'N/A'):<6.3f}" if l.get('count', 0) > 0 else "N/A"
        nm = f"{n.get('mean', 'N/A'):<6.3f}" if n and n.get('count', 0) > 0 else "N/A"
        np95 = f"{n.get('p95', 'N/A'):<6.3f}" if n and n.get('count', 0) > 0 else "N/A"

        if l and n and l.get('count', 0) > 0 and n.get('count', 0) > 0:
            speedup = f"{l['mean'] / n['mean']:.2f}x" if n['mean'] > 0 else "N/A"
        else:
            speedup = "N/A"
        print(f"| {label:<20} | {lm:<12} | {lp:<12} | {nm:<12} | {np95:<12} | {speedup:<10} |")

    print(f"\n  {'*'*60}")
    print(f"  * Speedup > 1.0 means LOCAL is faster (lower latency)")
    print(f"  * Speedup < 1.0 means NIM is faster (lower latency)")
    print(f"  {'*'*60}")


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="VoxLens latency benchmark")
    parser.add_argument("--backend", choices=["local", "nim", "both"], default="both",
                       help="Backend to benchmark")
    parser.add_argument("--runs", type=int, default=20,
                       help="Number of iterations per backend")
    parser.add_argument("--server", default="ws://localhost:8000/ws",
                       help="WebSocket server URL")
    args = parser.parse_args()

    global SERVER_URL
    SERVER_URL = args.server

    print(f"{'#'*70}")
    print(f"  VOXLENS LATENCY BENCHMARK")
    print(f"  Backend(s): {args.backend}")
    print(f"  Iterations per backend: {args.runs}")
    print(f"  Server: {SERVER_URL}")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Platform: {sys.platform}")
    print(f"{'#'*70}")

    local_agg = None
    nim_agg = None

    if args.backend in ("local", "both"):
        print(f"\n{'#'*70}")
        print(f"  PHASE 1: LOCAL BACKEND (Gemma 4 E2B via LiteRT-LM)")
        print(f"  Make sure server is running with: MODEL_BACKEND=local")
        print(f"{'#'*70}")
        results = await run_benchmark("local", args.runs)
        local_agg = aggregate_results(results)
        print_aggregate("local", local_agg)

    if args.backend in ("nim", "both"):
        print(f"\n{'#'*70}")
        print(f"  PHASE 2: NIM BACKEND (Nemotron via NVIDIA API)")
        print(f"  Make sure server is running with: MODEL_BACKEND=nim")
        print(f"{'#'*70}")
        results = await run_benchmark("nim", args.runs)
        nim_agg = aggregate_results(results)
        print_aggregate("nim", nim_agg)

    if local_agg and nim_agg:
        print_comparison(local_agg, nim_agg)

    print(f"\n{'#'*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'#'*70}")


if __name__ == "__main__":
    asyncio.run(main())
