# Queue Transcriber - Documentation

## Overview

The queue transcriber system allows you to queue transcription tasks to a SQLite-based task queue. Tasks are processed asynchronously by the huey worker.

There are two ways to queue tasks:

1. **`transcriber_huey` executable** - Main production queuer
2. **`queue_simulator.py` script** - Testing/simulation tool for queuing multiple tasks

## Prerequisites

- Python 3.11+ (if using scripts)
- Built executables in `dist/` directory (if using executables)
- Audio files to transcribe
- JSON payload files with transcription metadata

## Using the transcriber_huey Executable

### Basic Usage

Queue a single transcription task:

```bash
./dist/transcriber_huey --payload sample_payload.json --model turbo
```

### Command-Line Options

```bash
./dist/transcriber_huey \
  --payload <payload_file.json> \
  --model <whisper_model> \
  [--device cuda|cpu] \
  [--compute-type float16|float32|int8] \
  [--batch-size 16] \
  [--use-diarization] \
  [--hf-token <token>] \
  [--output-dir <dir>] \
  [--models-dir <dir>] \
  [--wait]
```

**Required Arguments:**
- `--payload`: Path to JSON payload file (or `-` for stdin)
- `--model`: Whisper model name (e.g., `turbo`, `large-v3`, `medium`, `base`, `small`)

**Optional Arguments:**
- `--device`: Device to use (`cuda` or `cpu`). Auto-detects if not specified
- `--compute-type`: Compute type (`float16` for GPU, `float32` for CPU, `int8`). Auto-detects if not specified
- `--batch-size`: Batch size for transcription (default: 16)
- `--use-diarization`: Enable speaker diarization (requires `--hf-token` or `HF_TOKEN` env var)
- `--hf-token`: HuggingFace token for diarization models
- `--output-dir`: Output directory for transcript files (default: `output`)
- `--models-dir`: Directory for Whisper models (default: `resources/models` or `MODELS_DIR` env var)
- `--wait`: Wait for task to complete synchronously (blocking mode)

### Examples

**Queue a single task:**
```bash
./dist/transcriber_huey --payload sample_payload.json --model large-v3
```

**Queue with custom output directory:**
```bash
./dist/transcriber_huey --payload sample_payload.json --model turbo --output-dir transcripts
```

**Queue with diarization:**
```bash
./dist/transcriber_huey --payload sample_payload.json --model turbo --use-diarization --hf-token YOUR_TOKEN
```

**Queue and wait for completion:**
```bash
./dist/transcriber_huey --payload sample_payload.json --model turbo --wait
```

**Queue from stdin:**
```bash
cat payload.json | ./dist/transcriber_huey --payload - --model turbo
```

## Using queue_simulator.py Script

The `queue_simulator.py` script is useful for testing and simulating multiple tasks.

### Basic Usage

**Queue a single task from audio file:**
```bash
uv run queue_simulator.py --audio-file resources/videos/5min.mp4 --model turbo
```

**Queue multiple tasks:**
```bash
uv run queue_simulator.py --audio-file resources/videos/5min.mp4 --count 10 --model turbo
```

### Command-Line Options

```bash
uv run queue_simulator.py \
  [--payload PAYLOAD] \
  [--payload-json PAYLOAD_JSON] \
  [--process-id PROCESS_ID] \
  [--audio-file AUDIO_FILE] \
  [--count COUNT] \
  [--model MODEL] \
  [--device {cuda,cpu}] \
  [--compute-type {float16,float32,int8}] \
  [--batch-size BATCH_SIZE] \
  [--output-dir OUTPUT_DIR] \
  [--models-dir MODELS_DIR] \
  [--delay DELAY]
```

**Options:**
- `--payload`: Path to JSON payload file
- `--payload-json`: JSON payload as string
- `--process-id`: Process ID for test payload (auto-generated if not provided)
- `--audio-file`: Audio file path (required if creating test payload)
- `--count`: Number of tasks to queue (default: 1)
- `--model`: Whisper model to use (default: turbo)
- `--device`: Device to use (auto-detects if not specified)
- `--compute-type`: Compute type (auto-detects if not specified)
- `--batch-size`: Batch size for transcription (default: 16)
- `--output-dir`: Output directory for transcript files (default: output)
- `--models-dir`: Directory for model files (default: resources/models)
- `--delay`: Delay between queuing tasks in seconds (default: 0.0)

### Examples

**Queue 5 tasks with 1 second delay:**
```bash
uv run queue_simulator.py --audio-file resources/videos/5min.mp4 --count 5 --delay 1.0 --model turbo
```

**Queue tasks from existing payload:**
```bash
uv run queue_simulator.py --payload sample_payload.json --count 3 --model large-v3
```

**Queue with custom process ID:**
```bash
uv run queue_simulator.py --audio-file resources/videos/5min.mp4 --process-id my_test_001 --model turbo
```

## Payload Format

The payload JSON must match the `TranscriptionPayloadSchema`:

```json
{
  "language_stats": {
    "major_language_percentage": 100,
    "language_breakdown": {
      "counts": {"en": 1},
      "durations": {"en": 60.0},
      "percentages": {"en": 100}
    }
  },
  "language_code": "en",
  "process_id": "unique_process_id",
  "language_classification": "single_language",
  "merged_mappings": [
    {
      "failed": false,
      "audio_file": "/path/to/audio.mp4",
      "language_code": "en",
      "text": "",
      "confidence": 0.9,
      "language_confidence": 0.92,
      "transcription_backend": "sieve",
      "segments": [],
      "words": [],
      "start": 0.0,
      "end": 60.0,
      "duration": 60.0
    }
  ]
}
```

**Key Fields:**
- `process_id`: Unique identifier for this transcription job (used as output filename)
- `merged_mappings`: Array of audio chunks to transcribe
  - `audio_file`: Full path to audio file
  - `start`: Start time in seconds
  - `end`: End time in seconds
  - `language_code`: Language code (e.g., "en", "es", "fr")

## Queue Database

Tasks are stored in a SQLite database:

- **Location**: `transcribe_queue_{env}.db` in the current directory
- **Environment**: Set via `ENV` or `ENVIRONMENT` environment variable (default: `default`)
- **Database name**: `transcribe_queue_default.db` (if ENV not set)

### Managing the Queue Database

**Clear the queue:**
```bash
rm -f transcribe_queue_*.db
```

**Inspect the queue (using sqlite3):**
```bash
sqlite3 transcribe_queue_default.db "SELECT COUNT(*) FROM task;"
```

**View queued tasks:**
```bash
sqlite3 transcribe_queue_default.db "SELECT id, name FROM task LIMIT 10;"
```

## Environment Variables

- `ENV` or `ENVIRONMENT`: Environment name for queue database (default: `default`)
  - Queue database: `transcribe_queue_{env}.db`
- `HF_TOKEN`: HuggingFace token for diarization models
- `MODELS_DIR`: Directory for Whisper models (default: `resources/models`)

## Output

Transcripts are saved to:
- **Directory**: `{output_dir}/`
- **Filename**: `{process_id}_transcript.json`

The output format matches the reference transcript format with:
- Machine name
- Video file reference
- Process ID
- Timestamp
- Full transcript with segments and words
- Language statistics

## Workflow Example

1. **Start the worker** (in one terminal):
   ```bash
   ./dist/huey_worker
   ```

2. **Queue tasks** (in another terminal):
   ```bash
   ./dist/transcriber_huey --payload sample_payload.json --model turbo
   ```

3. **Monitor progress**: The worker will process tasks and print progress to stdout

4. **Check results**: Transcripts will be saved in the output directory

## Troubleshooting

### Task Not Found Error

If you see `__main__.transcribe_payload_task not found in TaskRegistry`:
1. Clear the queue database: `rm -f transcribe_queue_default.db`
2. Restart the worker
3. Queue a new task

### Invalid Model Error

If you see `'turbo' is not a valid WhisperModel`:
- Use a valid model name: `large-v3`, `medium`, `base`, `small`, `tiny`
- Check the model name spelling

### Permission Denied Errors

If you see permission errors when writing to model directories:
- Ensure the `resources/models` directory is writable
- Check filesystem permissions

### Queue Database Locked

If tasks aren't being processed:
- Ensure only one worker is running per queue database
- Check if the database file is locked by another process
- Restart the worker if needed





