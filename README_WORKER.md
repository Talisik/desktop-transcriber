# Huey Worker - Documentation

## Overview

The huey worker is a background process that consumes and executes transcription tasks from a SQLite queue. It runs continuously, processing tasks as they are queued.

## Prerequisites

- Built executable: `dist/huey_worker`
- Or Python 3.11+ with dependencies installed (if using script)
- Queue database: `transcribe_queue_{env}.db` (created automatically)

## Starting the Worker

### Using the Executable

```bash
./dist/huey_worker
```

### Using the Script

```bash
python huey_worker.py
```

Or with `uv`:

```bash
uv run huey_worker.py
```

## Worker Behavior

When started, the worker will:

1. **Initialize**: Load the Huey consumer and register tasks
2. **Connect to Queue**: Open the SQLite queue database
3. **Listen**: Continuously poll for new tasks
4. **Process**: Execute tasks as they arrive
5. **Log**: Print progress and results to stdout

### Startup Output

```
starting huey worker
   press Ctrl+C to stop

[2026-02-09 16:26:42,498] INFO:huey.consumer:MainThread:Huey consumer started with 1 thread, PID 80805
[2026-02-09 16:26:42,498] INFO:huey.consumer:MainThread:Scheduler runs every 1 second(s).
[2026-02-09 16:26:42,498] INFO:huey.consumer:MainThread:Periodic tasks are enabled.
[2026-02-09 16:26:42,498] INFO:huey.consumer:MainThread:The following commands are available:
+ transcriber_huey.transcribe_payload_task
```

## Stopping the Worker

Press `Ctrl+C` to gracefully stop the worker. The worker will:
- Finish processing the current task (if any)
- Close database connections
- Exit cleanly

## Monitoring Tasks

### Task Execution Logs

When a task is executed, you'll see:

```
[2026-02-09 16:27:03,788] INFO:huey:Worker-1:Executing transcriber_huey.transcribe_payload_task: 77c8eb6c-60bd-4671-a919-5c7b39e998ad 2 retries
starting transcription task
   process_id: test_20260209_162702
   chunks: 1

processing chunk 1/1
   audio_file: 5min.mp4
   language: en
   time range: 0.0s - 60.0s
```

### Successful Completion

```
✓ chunk 1 completed: 45 segments

merging results:
   total segments: 45
transcript saved: output/test_20260209_162702_transcript.json
[2026-02-09 16:27:05,818] INFO:huey:Worker-1:transcriber_huey.transcribe_payload_task: 77c8eb6c-60bd-4671-a919-5c7b39e998ad 2 retries executed in 2.029s
```

### Task Failure

```
chunk 1 failed: [error message]

merging results:
   total segments: 0
transcript saved: output/test_20260209_162702_transcript.json
```

Failed chunks are marked in the output transcript but don't stop the overall process.

## Log Output Explanation

### Info Logs

- `Huey consumer started`: Worker initialization
- `The following commands are available`: Registered tasks
- `Executing transcriber_huey.transcribe_payload_task`: Task started
- `executed in X.XXXs`: Task completion time

### Task Progress

- `starting transcription task`: Task begins
- `processing chunk X/Y`: Processing audio chunk
- `✓ chunk X completed`: Chunk successfully processed
- `chunk X failed`: Chunk processing failed
- `merging results`: Combining all chunks
- `transcript saved`: Final transcript written

## Environment Variables

- `ENV` or `ENVIRONMENT`: Environment name for queue database (default: `default`)
  - Queue database: `transcribe_queue_{env}.db`
  - Allows multiple environments (dev, staging, prod)
- `HF_TOKEN`: HuggingFace token for speaker diarization
- `MODELS_DIR`: Directory for Whisper models (default: `resources/models`)

### Example: Multiple Environments

**Development environment:**
```bash
ENV=dev ./dist/huey_worker
# Uses: transcribe_queue_dev.db
```

**Production environment:**
```bash
ENV=prod ./dist/huey_worker
# Uses: transcribe_queue_prod.db
```

## Performance Considerations

### Sequential Processing

Currently, tasks are processed **sequentially** (one at a time). This ensures:
- Predictable memory usage
- No GPU memory conflicts
- Easier debugging

### Memory Management

The worker:
- Cleans up models after each chunk
- Runs garbage collection
- Clears CUDA cache (if using GPU)

### Resource Usage

- **CPU**: Moderate (transcription is CPU/GPU intensive)
- **Memory**: Varies by model size (large-v3 uses more than base)
- **GPU**: If available, used for faster processing
- **Disk**: Stores transcripts in output directory

## Troubleshooting

### Windows [WinError 2] (FFmpeg not found)

On Windows, if you see `[WinError 2] The system cannot find the file specified` during extraction or transcription, it usually means FFmpeg is not in your system `PATH`.

**Solution:**
Use the `--ffmpeg-path` argument when starting the worker:
```bash
python huey_worker.py --ffmpeg-path "C:\path\to\ffmpeg\bin\ffmpeg.exe"
```
The worker will automatically add this directory to the system `PATH` for the duration of its execution, allowing all libraries to find FFmpeg.

### Worker Won't Start

**Error: `cannot import name 'consumer_main'`**
- Rebuild the executable: `uv run pyinstaller huey_worker.spec --clean`

**Error: `No module named 'transcriber_huey'`**
- Ensure all dependencies are included in the spec file
- Rebuild the executable

### Tasks Not Processing

**Worker shows "The following commands are available" but no tasks execute:**
- Check if tasks are actually queued: `sqlite3 transcribe_queue_default.db "SELECT COUNT(*) FROM task;"`
- Verify the queue database name matches (check ENV variable)
- Restart the worker

**Error: `__main__.transcribe_payload_task not found in TaskRegistry`**
- Clear the queue database: `rm -f transcribe_queue_default.db`
- Restart the worker
- Queue new tasks

### Missing Module Errors

**Error: `No module named 'pyannote.audio.models'`**
- Rebuild the executable with updated spec file
- Ensure `collect_all('pyannote')` is used in the spec

**Error: `No such file or directory: '/tmp/_MEI.../speechbrain/utils'`**
- Rebuild the executable with updated spec file
- Ensure `collect_all('speechbrain')` is used in the spec

### Permission Errors

**Error: `Permission denied: 'resources/models/...'`**
- Check filesystem permissions on `resources/models` directory
- Ensure the worker has write access

### GPU Issues

**Warning: `Can't initialize NVML`**
- This is usually harmless (NVML is for monitoring)
- Transcription will still work if CUDA is available

**CUDA out of memory:**
- Use a smaller model (e.g., `base` instead of `large-v3`)
- Reduce batch size: `--batch-size 8`
- Process fewer chunks simultaneously

## Running Multiple Workers

You can run multiple workers for the same queue to process tasks in parallel:

```bash
# Terminal 1
./dist/huey_worker

# Terminal 2
./dist/huey_worker

# Terminal 3
./dist/huey_worker
```

**Note**: SQLite handles locking automatically, but for better performance with multiple workers, consider using a different backend (Redis, PostgreSQL) in production.

## Production Deployment

### Systemd Service

Create `/etc/systemd/system/huey-worker.service`:

```ini
[Unit]
Description=Huey Transcription Worker
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/transcriber
Environment="ENV=prod"
ExecStart=/path/to/transcriber/dist/huey_worker
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable huey-worker
sudo systemctl start huey-worker
```

### Logging

Redirect logs to a file:
```bash
./dist/huey_worker >> worker.log 2>&1
```

Or use systemd journal:
```bash
journalctl -u huey-worker -f
```

## Monitoring

### Check Worker Status

```bash
ps aux | grep huey_worker
```

### Check Queue Size

```bash
sqlite3 transcribe_queue_default.db "SELECT COUNT(*) FROM task;"
```

### Check Recent Tasks

```bash
sqlite3 transcribe_queue_default.db "SELECT id, name FROM task ORDER BY id DESC LIMIT 10;"
```

## Best Practices

1. **Run one worker per queue database** to avoid conflicts
2. **Monitor disk space** - transcripts and models can be large
3. **Use environment variables** to separate dev/staging/prod
4. **Keep worker running** - restart automatically on failure
5. **Monitor logs** - watch for errors and performance issues
6. **Clear old queue databases** periodically if needed











