# Huey Transcriber Usage Guide

## Overview
The Huey-based transcriber processes audio chunks asynchronously using a task queue.

## Starting the Worker

Start the Huey worker to process queued tasks:

```bash
python huey_worker.py
```

Or using `uv`:

```bash
uv run huey_worker.py
```

The worker will:
- Listen for queued transcription tasks
- Process them sequentially (or in parallel in future versions)
- Save transcripts to the output directory

Press `Ctrl+C` to stop the worker.

## Queuing Tasks

### Basic Usage

Queue a transcription task from a JSON payload file:

```bash
python transcriber_huey.py --payload sample_payload.json --model turbo
```

Or using `uv`:

```bash
uv run transcriber_huey.py --payload sample_payload.json --model turbo
```

### Command Line Options

```bash
python transcriber_huey.py \
  --payload <payload_file.json> \
  --model <whisper_model> \
  [--device cuda|cpu] \
  [--compute-type float16|float32] \
  [--batch-size 16] \
  [--use-diarization] \
  [--hf-token <token>] \
  [--output-dir <dir>] \
  [--models-dir <dir>] \
  [--wait]
```

**Required:**
- `--payload`: Path to JSON file with transcription payload
- `--model`: Whisper model name (e.g., `turbo`, `large-v3`, `medium`)

**Optional:**
- `--device`: Device to use (`cuda` or `cpu`, defaults to auto-detect)
- `--compute-type`: Compute type (`float16` for GPU, `float32` for CPU, defaults to auto)
- `--batch-size`: Batch size for processing (default: 16)
- `--use-diarization`: Enable speaker diarization (requires `--hf-token`)
- `--hf-token`: HuggingFace token for diarization models
- `--output-dir`: Output directory for transcripts (default: `./transcripts`)
- `--models-dir`: Directory for Whisper models (default: `./models`)
- `--wait`: Wait for task to complete (blocking mode)

### Payload Format

The payload JSON should match the `TranscriptionPayloadSchema`:

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
      "start": 0.0,
      "end": 60.0,
      "duration": 60.0
    }
  ]
}
```

### Queue from STDIN

You can also pipe JSON from stdin:

```bash
cat payload.json | python transcriber_huey.py --payload - --model turbo
```

### Wait for Completion

To wait for the task to complete (blocking):

```bash
python transcriber_huey.py --payload sample_payload.json --model turbo --wait
```

## Environment Variables

- `ENV` or `ENVIRONMENT`: Environment name for queue database (default: `default`)
  - Queue database: `transcribe_queue_{env}.db`
- `HF_TOKEN`: HuggingFace token for diarization (can also use `--hf-token`)

## Output

Transcripts are saved to:
- Directory: `{output_dir}/{process_id}/`
- Filename: `{process_id}_transcript.json`

The output format matches the reference transcript format (e.g., `rog_turbo_6mins_transcript.json`).

## Workflow Example

1. **Start the worker** (in one terminal):
   ```bash
   python huey_worker.py
   ```

2. **Queue tasks** (in another terminal or from your application):
   ```bash
   python transcriber_huey.py --payload sample_payload.json --model turbo
   python transcriber_huey.py --payload another_payload.json --model large-v3
   ```

3. **Monitor progress**: The worker will process tasks and print progress to stdout.

4. **Check results**: Transcripts will be saved in the output directory.

## Troubleshooting

### Task Not Found Error

If you see `__main__.transcribe_payload_task not found in TaskRegistry`:
1. Clear the queue database: `rm -f transcribe_queue_default.db`
2. Restart the worker
3. Queue a new task

### Queue Database

The queue database is stored as `transcribe_queue_{env}.db` in the current directory. You can:
- Clear it: `rm -f transcribe_queue_*.db`
- Inspect it: SQLite database (use `sqlite3` to view)

