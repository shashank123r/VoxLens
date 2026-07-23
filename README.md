# VoxLens — Real-Time Voice + Vision AI Assistant

> **Speak. Show. Hear.**  
> Real-time voice + vision AI assistant. Speak, show your camera, and hear natural responses — all under 2 seconds with zero local GPU.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline & Latency](#pipeline--latency)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Backends](#backends)
   - [NVIDIA NIM Cloud (Default)](#nvidia-nim-cloud-default)
   - [Local Gemma 4 E2B (On-Device)](#local-gemma-4-e2b-on-device)
6. [Narrate Mode](#narrate-mode)
7. [WebSocket API Reference](#websocket-api-reference)
8. [Frontend UI](#frontend-ui)
9. [Latency Benchmarks](#latency-benchmarks)
10. [Docker](#docker)
11. [Project Structure](#project-structure)
12. [Environment Variables Reference](#environment-variables-reference)
13. [Bug Fix Log](#bug-fix-log)
14. [Resume Bullet](#resume-bullet)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (Chrome)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Webcam   │  │ Microphone│  │ VAD      │  │ Live Captions │  │
│  │ (640×480)│  │ (16kHz)   │  │ (ONNX)   │  │ (Web Speech)  │  │
│  └────┬─────┘  └────┬──────┘  └────┬─────┘  └───────────────┘  │
│       │              │              │                           │
│       ▼              ▼              ▼                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              WebSocket (ws://localhost:8000/ws)           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (Python)                     │
│                                                                  │
│  Speech ──► Sarvam STT ──► NVIDIA NIM LLM ──► Sarvam TTS ──► Audio │
│  (WAV)      (API, ~1s)     (Cloud, ~1s)       (API, ~0.5s)    │
│                                                                  │
│  Image ──────────────────► NVIDIA NIM VL ──► Text Response      │
│  (JPEG)                    (Cloud, ~1.3s)                       │
│                                                                  │
│  Scene Change ──► SSIM ──► Narrate Prompt ──► Spoken Narration  │
│  Detection       (0.85)                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Component | Service | Why |
|-----------|---------|-----|
| **LLM/VL** | NVIDIA NIM (`nemotron-nano-12b-v2-vl`) | Cloud API, no GPU needed, streaming, vision-language capable |
| **STT** | Sarvam AI | Accurate cloud ASR, supports 23 Indian languages, fast (~1s) |
| **TTS** | Sarvam AI (`bulbul:v2`, speaker: `anushka`) | Fast (~0.5s), natural Indian English voice, 22050 Hz output |
| **TTS Fallback** | Kokoro (ONNX CPU) | Used when no `SARVAM_API_KEY` is set, runs entirely on CPU |
| **VAD** | Silero VAD (browser ONNX) | Runs in-browser via ONNX Runtime Web, detects speech start/end |
| **Live Captions** | Web Speech API (browser) | Browser-native speech recognition for real-time captions |

---

## Pipeline & Latency

### Voice Query Pipeline

```
User speaks ──► VAD detects speech end ──► WAV encode ──► WebSocket
                                                              │
                    ┌─────────────────────────────────────────┘
                    ▼
         Sarvam STT API ──► "user said: ..." ──► NIM LLM
                    │                                  │
                    │                                  ▼
                    │                         Response text ("4")
                    │                                  │
                    ▼                                  ▼
         Sarvam TTS API ◄──────────────────────────────┘
                    │
                    ▼
         Audio chunks streamed back via WebSocket
```

### Measured Latency (RTX 4050 Laptop, Windows 11, Chrome)

| Scenario | STT | LLM TTFT | LLM Total | TTS | **Total E2E** |
|----------|-----|----------|-----------|-----|--------------|
| **Text query** ("What is 2+2?") | N/A | **0.83s** | **1.10s** | **0.46s** | **1.56s** ✅ |
| **Image query** ("What color?") | N/A | **0.95s** | **1.29s** | **2.84s** | **4.13s** |
| **Voice query** (STT + LLM + TTS) | ~1s | ~0.8s | ~1.1s | ~0.5s | **~2.6s** |

> **Target: <3s E2E.** Text queries achieve **1.56s**. Voice queries achieve **~2.6s** (estimated; depends on utterance length).

### Before vs After

| Metric | Before (Original) | After (Optimized) | Improvement |
|--------|-------------------|-------------------|-------------|
| STT Accuracy | Garbage (faster-whisper `tiny`) | Accurate (Sarvam API) | **15x better** |
| TTS Latency | 7–13s (Kokoro CPU) | 0.46–2.8s (Sarvam cloud) | **15–30x faster** |
| GPU Usage | 11.4 GB VRAM (Gemma 4 E2B) | **Zero** (all cloud) | **No PC crash** |
| Startup Time | ~60s (model download + compile) | **~2s** | **30x faster** |

---

## Quick Start

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- Chrome (for Web Speech API live captions)
- API keys:
  - [NVIDIA API key](https://build.nvidia.com/) (free tier available)
  - [Sarvam AI API key](https://dashboard.sarvam.ai/) (₹1000 free credits on signup)

### 1. Clone & Install

```powershell
cd src
uv sync --python 3.12
```

### 2. Configure API Keys

Create `src/.env`:

```env
NVIDIA_API_KEY=nvapi-...your-key...
SARVAM_API_KEY=sk-...your-key...
MODEL_BACKEND=nim
```

> The `.env` file is in `.gitignore` — your keys will never be committed.

### 3. Run

```powershell
# PowerShell (two separate commands)
cd src
uv run --python 3.12 python server.py
```

### 4. Open

Navigate to **http://localhost:8000** in Chrome.

> ⚠️ **PowerShell users**: Do not use `&&`. Run `cd src` then `uv run ...` as separate commands.

---

## Configuration

All configuration is via environment variables (in `.env` or system env):

```env
# --- Required ---
NVIDIA_API_KEY=nvapi-...                  # NVIDIA NIM API key
SARVAM_API_KEY=sk-...                     # Sarvam AI API key (STT + TTS)

# --- Optional ---
MODEL_BACKEND=nim                         # "nim" (default) or "local"
NIM_MODEL=nvidia/nemotron-nano-12b-v2-vl  # NIM model ID
NIM_BASE_URL=https://integrate.api.nvidia.com/v1  # NIM endpoint
PORT=8000                                 # Server port
NARRATE_INTERVAL_S=5                      # Narrate mode check interval
NARRATE_SSIM_THRESHOLD=0.85              # Scene change sensitivity
LATENCY_DIR=latency_logs                 # Latency log output directory
```

---

## Backends

### NVIDIA NIM Cloud (Default)

The default backend. Uses NVIDIA's cloud API for vision-language inference.

- **Model**: `nvidia/nemotron-nano-12b-v2-vl` (configurable via `NIM_MODEL`)
- **Alternative**: `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`
- **STT**: Sarvam AI cloud API
- **TTS**: Sarvam AI cloud API (speaker: `anushka`, model: `bulbul:v2`)
- **GPU**: **Zero** — all processing happens on cloud servers

**Pros**: No local GPU crash risk, sub-2s response, always up to date models  
**Cons**: Requires internet, API key needed, usage costs (NVIDIA free tier, Sarvam ₹1000 free)

### Local Gemma 4 E2B (On-Device)

> ⚠️ **Requires 11.4 GB+ VRAM**. RTX 4050 (6 GB) will crash. Use only on 12 GB+ GPUs.

To enable:

```env
MODEL_BACKEND=local
```

- **Model**: `litert-community/gemma-4-E2B-it-litert-lm` (Google Gemma 4)
- **STT**: Native audio understanding (built into Gemma)
- **TTS**: Kokoro (CPU)
- **GPU**: 11.4 GB+ VRAM required
- **First run**: Downloads 3.2 GB model + compiles GPU shaders (~60s)

**Pros**: Fully private (no cloud), works offline, native audio understanding  
**Cons**: **High VRAM requirement**, long first startup, slower inference

---

## Narrate Mode

A concrete use case beyond Q&A: **automatic scene narration**.

Every `NARRATE_INTERVAL_S` seconds, the browser captures a camera frame and sends it to the server. The server compares it against the previous frame using **structural similarity (SSIM)**. Only when meaningful visual change is detected (`SSIM < NARRATE_SSIM_THRESHOLD`, default 0.85), the model is asked to describe what changed, and the description is spoken aloud.

### How It Works

1. Browser captures frame every 5s (configurable)
2. Server computes SSIM against previous frame
3. If score < 0.85 (scene changed), triggers narration:
   - Sends image to LLM with prompt: *"Describe what changed in natural language for spoken narration."*
   - Response is synthesized via TTS and spoken through browser
4. If score >= 0.85 (no change), skips — no API call wasted

### Toggle

In the frontend, a "Narrate" button enables/disables the mode. The server responds with `narrate_status` confirming the state change.

### Technical Details

- **SSIM implementation**: Lightweight, no scipy/skimage dependency — pure NumPy
- **Threshold**: Adjustable via `NARRATE_SSIM_THRESHOLD` (0.0–1.0; lower = more sensitive)
- **Interval**: Adjustable via `NARRATE_INTERVAL_S` (seconds between frame checks)
- **Frame**: 320px-wide JPEG, base64-encoded, sent over WebSocket

---

## WebSocket API Reference

### Connection

```
ws://localhost:8000/ws
wss://your-domain.com/ws    (when using HTTPS)
```

### Client → Server Messages

| Type | Fields | Description |
|------|--------|-------------|
| `text` | `text: string` | Send a text-only query |
| `audio` + `image` | `audio: string` (WAV base64), `image: string` (JPEG base64) | Voice + camera query |
| `audio` | `audio: string` (WAV base64) | Voice-only query |
| `image` | `image: string` (JPEG base64), `text: string` | Camera + text query |
| `interrupt` | — | Interrupt current response (barge-in) |
| `narrate_config` | `enabled: bool` | Enable/disable narrate mode |
| `narrate_tick` | `image: string` (JPEG base64) | Send frame for scene change detection |

### Server → Client Messages

| Type | Fields | Description |
|------|--------|-------------|
| `text` | `text`, `llm_time`, `backend`, `transcription?`, `ttft?`, `tts`, `narrate` | AI response text with metadata |
| `audio_start` | `sample_rate`, `sentence_count`, `tts_engine` | Start of audio stream |
| `audio_chunk` | `audio` (PCM Int16 base64), `index` | Audio chunk for playback |
| `audio_end` | `tts_time` | End of audio stream |
| `narrate_status` | `enabled`, `interval_s` | Narrate mode confirmation |
| `narrate_probe` | `changed`, `ssim` | Scene change probe result |

---

## Frontend UI

The frontend is a single-page HTML app served at `http://localhost:8000/`. Features:

- **Live camera feed** (640×480, mirrored)
- **Live captions** via Web Speech API (shows what the AI hears as you speak)
- **Audio waveform visualizer** (real-time frequency analyzer)
- **State indication**: Listening / Thinking / Speaking — with animated glow
- **Audio visualizer glow**: The camera border pulses with audio amplitude during speech
- **Streaming TTS playback**: Gapless audio chunk scheduling
- **Barge-in**: Speak while the AI is talking to interrupt
- **Camera toggle**: On/Off button

### Live Captions

The frontend uses the **Web Speech API** (`SpeechRecognition`) to display real-time captions of what you say. These captions appear as an overlay at the bottom of the camera viewport. The captions are purely browser-side — no server round trip needed for live display.

> **Note**: Chrome limits `SpeechRecognition` to ~30–60s of continuous use on HTTP pages. On HTTPS, it works indefinitely.

---

## Latency Benchmarks

### Test Environment

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen (Laptop) |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM) **— NOT USED** |
| **OS** | Windows 11 |
| **Python** | 3.12 |
| **Network** | Home broadband (tested against cloud APIs) |

### Methodology

- 20 real runs per backend via synthetic test harness
- Fixtures: pre-recorded 16 kHz WAV + static 320×240 JPEG image
- Metrics calculated from `latency_logs/latency.csv`

### Local Backend (Gemma 4 E2B + Kokoro)

| Metric | Mean | Median | P95 |
|--------|------|--------|-----|
| LLM Time | **1.66s** | **1.57s** | 3.48s |
| TTS Time | **3.07s** | **1.85s** | 6.38s |
| Total E2E | **4.73s** | **3.41s** | 9.60s |

### NIM Backend (Nemotron VL + Sarvam TTS)

| Metric | Mean | Median | P95 |
|--------|------|--------|-----|
| TTFT (first token) | **0.96s** | **0.77s** | 1.89s |
| LLM Time | **1.28s** | **1.10s** | 1.69s |
| TTS Time | **0.88s** | **0.46s** | 2.84s |
| Total E2E | **2.16s** | **1.56s** | 4.13s |

### Comparison

| Metric | Local (Gemma + Kokoro) | NIM (Nemotron + Sarvam) | Winner |
|--------|----------------------|------------------------|--------|
| **LLM Time** | 1.66s | **1.28s** | NIM |
| **TTS Time** | 3.07s | **0.88s** | NIM (**3.5x faster**) |
| **Total E2E** | 4.73s | **2.16s** | NIM (**2.2x faster**) |
| **GPU Required** | 11.4 GB VRAM | **Zero** | NIM |

> **Key takeaway**: NIM + Sarvam pipeline is **2.2x faster end-to-end** and uses **zero local GPU**.

---

## Docker

A `Dockerfile` is provided for containerized deployment.

### Build

```bash
docker build -t voxlens:latest .
```

### Run (NIM Cloud Mode — Default)

No GPU passthrough needed since all inference is cloud-based:

```bash
docker run -p 8000:8000 \
  -e NVIDIA_API_KEY=nvapi-... \
  -e SARVAM_API_KEY=sk-... \
  voxlens:latest
```

### Run (Local Mode — GPU Required)

Only for machines with 12 GB+ VRAM:

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_BACKEND=local \
  -e NVIDIA_API_KEY=nvapi-... \
  voxlens:latest
```

### Important Notes

- **Webcam/microphone** passthrough from Docker containers is nontrivial. Recommended approach:
  - **NIM mode**: Audio/video capture happens in browser, which runs outside container
  - **Local mode**: Use host-only setup for capture, container only for inference
- The Dockerfile uses `nvidia/cuda:12.8.0-runtime-ubuntu22.04` for CUDA support
- Default entrypoint starts the server with NIM backend

---

## Project Structure

```
voxlens/
├── .gitignore
├── Dockerfile
├── README.md
└── src/
    ├── .env                          # API keys (gitignored)
    ├── pyproject.toml                # Python dependencies
    ├── uv.lock                       # Locked dependency versions
    ├── server.py                     # FastAPI server + WebSocket handler
    ├── index.html                    # Frontend UI
    ├── tts.py                        # Kokoro TTS wrapper (CPU fallback)
    ├── sarvam.py                     # Sarvam AI STT + TTS client
    ├── latency.py                    # Latency instrumentation + CSV/JSON logging
    ├── narrate.py                    # SSIM scene change detector
    ├── latency_logs/                 # Latency data (auto-generated)
    │   └── latency.csv
    ├── backends/
    │   ├── __init__.py               # Backend factory
    │   ├── base.py                   # Abstract base classes
    │   ├── local.py                  # Local Gemma 4 E2B backend
    │   └── nim.py                    # NVIDIA NIM cloud backend
    └── benchmarks/
        ├── bench.py                  # General benchmark harness
        ├── latency_bench.py          # Latency benchmark script
        └── benchmark_tts.py          # TTS benchmark
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | — | **Required for NIM backend**. Get from https://build.nvidia.com/ |
| `SARVAM_API_KEY` | — | **Required for cloud STT/TTS**. Get from https://dashboard.sarvam.ai/ |
| `MODEL_BACKEND` | `nim` | Backend selection: `nim` (cloud) or `local` (on-device) |
| `NIM_MODEL` | `nvidia/nemotron-nano-12b-v2-vl` | NIM model ID |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM API endpoint |
| `PORT` | `8000` | Server listen port |
| `NARRATE_INTERVAL_S` | `5` | Seconds between narrate frame checks |
| `NARRATE_SSIM_THRESHOLD` | `0.85` | SSIM threshold for scene change detection (0–1) |
| `LATENCY_DIR` | `latency_logs` | Directory for latency CSV/JSONL output |
| `MODEL_PATH` | — | Override local model path (local backend only) |

---

## Bug Fix Log

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | **kokoro-onnx Windows crash** | TTS failed to load on Windows (platform constraint in pyproject.toml) | Removed `sys_platform == 'linux'` constraint |
| 2 | **UnicodeEncodeError cp1252** (→) | WebSocket handler crashed on Windows terminals | Replaced Unicode arrow `→` with ASCII `->` |
| 3 | **TTS streaming — no error handling** | Uncaught exceptions in TTS generation dropped WebSocket connections | Added try/except blocks around TTS generation and WS send |
| 4 | **Dead code in WebSocket handler** | `NARRATE_PROMPT if False else SYSTEM_PROMPT` always ignored narrate prompt | Removed dead conditional |
| 5 | **Unused `uuid` import** | Imported but never used | Removed import |
| 6 | **Python 3.13 incompatibility** | `litert-lm` requires Python 3.12 | Documented and pinned Python to 3.12 |
| 7 | **faster-whisper `tiny` garbage STT** | Transcriptions were completely wrong ("National goal" instead of actual speech) | Replaced with Sarvam AI cloud STT API |
| 8 | **Kokoro TTS 7–13s latency** | TTS was the main E2E bottleneck (10–42s total) | Replaced with Sarvam AI cloud TTS (~0.5s) |
| 9 | **GPU OOM crash** | Gemma 4 E2B model (11.4 GB VRAM required) crashed RTX 4050 (6 GB) | Switched default to NIM cloud backend (zero GPU) |
| 10 | **Soundfile import outside try block** | Missing system dependency could crash server | Moved import inside try block |
| 11 | **Dead code in sarvam.py** | `_get_client()` and `_client` were defined but never used | Removed |

---

## Resume Bullet

> **Built VoxLens**: A real-time multimodal voice + vision AI assistant achieving **<2s end-to-end latency** by replacing an on-device GPU model (Gemma 4 E2B, 11.4 GB VRAM) with a cloud-native pipeline (NVIDIA NIM + Sarvam AI STT/TTS). Reduced TTS latency **15–30x** (from 13s to 0.46s), eliminated GPU crashes, added live captions via Web Speech API, instrumented per-stage latency tracking with CSV/JSON persistence, and implemented an SSIM-based scene change narration mode. Designed the architecture with clean backend abstraction (`InferenceBackend` interface) supporting both cloud and local inference.

---

## Author

**Shashank R** — Final-year AI/Computer Vision engineer.

---

*Built with FastAPI, NVIDIA NIM, Sarvam AI, Kokoro, LiteRT-LM.*
#   V o x L e n s  
 