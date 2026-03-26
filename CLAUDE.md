# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Queue-based asynchronous audio transcription system using **Huey** (SQLite-backed task queue) with **WhisperX** as the transcription engine. Targets both Linux (Docker/script) and Windows (PyInstaller executables).

## Common Commands

```bash
# Install dependencies
uv sync

# Run direct transcription (no queue)
uv run python main.py <audio_file> --model <model> [--device cuda|cpu]

# Start Huey worker (processes queued tasks)
uv run python huey_worker.py

# Queue a transcription task
uv run python transcriber_huey.py --payload <payload.json> --model <model> [--wait]

# Queue tasks for testing (lightweight, no ML imports)
uv run python queue_simulator.py --audio-file <file> --model turbo --device cpu [--use-diarization --hf-token <token>]

# Check system resources
uv run python resource_tracker_main.py

# Check model compatibility
uv run python model_profile_main.py

# Build Windows executables
pyinstaller huey_worker_windows.spec --clean --noconfirm
pyinstaller model_profile_windows.spec --clean --noconfirm
```

## Architecture

```
Queue Producers                  Queue (SQLite)              Queue Consumer
─────────────                    ──────────────              ──────────────
transcriber_huey.py  ──→  transcribe_queue_[ENV].db  ──→  huey_worker.py
queue_simulator.py   ──→                              ──→    consumer.py
                                                              ↓
                                                      transcriber/
                                                      (WhisperX engine)
                                                              ↓
                                                      output/{process_id}/
                                                      {process_id}_transcript.json
```

**Key design decisions:**
- `transcriber_huey_queue.py` is a lightweight stub that registers the Huey task without importing ML libraries (torch, whisperx). This allows `queue_simulator.py` to enqueue tasks fast. The real implementation in `transcriber_huey.py` runs inside the worker.
- `huey_worker.py` pre-parses `--ffmpeg-path` before any ML imports to inject ffmpeg into PATH early, avoiding library warnings.
- `torch.load` is monkey-patched early to disable `weights_only` for compatibility.
- `sys.modules['transcriber_huey']` is explicitly set in frozen mode to keep task names consistent between queuer and worker.
- PyInstaller uses **onedir mode** (not onefile) to avoid the 4GB single-file limit and antivirus false positives.

**Module structure:**
- `transcriber/` — Core transcription: WhisperX wrapper, schemas (Pydantic), segment merging, SRT/VTT conversion, metrics
- `model_manager/` — Model downloading and caching (WhisperX models)
- `resource_tracker/` — System resource detection (RAM, CPU, GPU via psutil/pynvml), model profile matching via `model_profiles.json`

## Queue Database

- SQLite file: `transcribe_queue_[ENV].db` (auto-created by Huey)
- Environment set via `ENV` or `ENVIRONMENT` env var (default: `"default"`)
- External data source: `salina_vad.db` contains VAD/language detection results from upstream pipeline

## Environment

- **Python 3.11** required (`.python-version`)
- **uv** for dependency management
- Key env vars: `ENV`/`ENVIRONMENT`, `HF_TOKEN` (diarization), `MODELS_DIR`, `FFMPEG_PATH`
- CUDA 12.1 used in CI builds

## CI/CD

GitHub Actions builds Windows executables:
- `build-windows-huey-worker.yml` — Builds `huey_worker.exe` (CUDA PyTorch, 60min timeout)
- `build-windows-model-profile.yml` — Builds `model_profile.exe` (lightweight, 30min timeout)
- Triggered on push to `main`, `master`, `staging`, `face-recog/queue`
