# Executables Command Reference

## GPU Support

The Windows executables include CUDA-enabled PyTorch, enabling GPU acceleration for:
- **Transcription**: Uses CTranslate2 (works independently of PyTorch CUDA status)
- **Alignment**: Uses PyTorch (requires CUDA-enabled PyTorch - included in build)
- **Diarization**: Uses PyTorch/pyannote (requires CUDA-enabled PyTorch - included in build)

### Executable Size
- Executables are **~200-300MB larger** due to CUDA-enabled PyTorch libraries
- Total size: ~1GB+ (includes all ML dependencies)

### GPU Requirements
- **NVIDIA GPU** with CUDA drivers installed
- Use `ResourceTracker.exe` to detect GPU availability
- Pass `--device cuda` to `queue_simulator.exe` when GPU is available

### Device Detection Workflow
```
1. Run ResourceTracker.exe to detect GPU
2. If GPU detected, use: queue_simulator.exe --device cuda ...
3. If no GPU, use: queue_simulator.exe --device cpu ... (or omit --device, defaults to cpu)
```

## huey_worker.exe

# Basic usage (auto-detect ffmpeg)
huey_worker.exe

# With custom ffmpeg path
huey_worker.exe --ffmpeg-path "C:\path\to\ffmpeg.exe"## queue_simulator.exe

# Queue single task
queue_simulator.exe --audio-file "path\to\audio.mp4"

# Queue with custom model
queue_simulator.exe --audio-file "path\to\audio.mp4" --model "large-v3"

# Queue multiple tasks
queue_simulator.exe --audio-file "path\to\audio.mp4" --count 5

# Queue with GPU
queue_simulator.exe --audio-file "path\to\audio.mp4" --device cuda

# Queue with custom settings
queue_simulator.exe --audio-file "path\to\audio.mp4" --model "turbo" --batch-size 32 --output-dir "output"

# Queue from payload file
queue_simulator.exe --payload "path\to\payload.json"

# Queue with process ID from database
queue_simulator.exe --process-id "process_id" --audio-file "path\to\audio.mp4" --db-path "salina_vad.db"## Frontend Integration Commands

### Database Workflow (Fetch from DB → Build Payload → Queue)
h
# Basic: Queue task from database with process ID
queue_simulator.exe --process-id "process_123" --audio-file "C:\audio\video.mp4" --db-path "C:\path\to\salina_vad.db"

# With custom model and settings
queue_simulator.exe --process-id "process_123" --audio-file "C:\audio\video.mp4" --db-path "C:\path\to\salina_vad.db" --model "large-v3" --device cuda

# With custom output directory
queue_simulator.exe --process-id "process_123" --audio-file "C:\audio\video.mp4" --db-path "C:\path\to\salina_vad.db" --output-dir "C:\output\transcripts"

# Queue multiple tasks from same process
queue_simulator.exe --process-id "process_123" --audio-file "C:\audio\video.mp4" --db-path "C:\path\to\salina_vad.db" --count 3### Frontend Command Examples

# Example 1: Queue transcription from database
queue_simulator.exe --process-id "20260211_120000_abc123" --audio-file "C:\uploads\audio.mp4" --db-path "C:\data\salina_vad.db" --model "turbo"

# Example 2: High-quality transcription from database
queue_simulator.exe --process-id "20260211_120000_abc123" --audio-file "C:\uploads\audio.mp4" --db-path "C:\data\salina_vad.db" --model "large-v3" --device cuda --batch-size 16

# Example 3: Queue with all options
queue_simulator.exe --process-id "20260211_120000_abc123" --audio-file "C:\uploads\audio.mp4" --db-path "C:\data\salina_vad.db" --model "turbo" --device cuda --compute-type float16 --batch-size 32 --output-dir "C:\output" --models-dir "C:\models"
### Required Parameters for Database Workflow

- `--process-id` - Process ID from `salina_languini_results` table (required)
- `--audio-file` - Full path to audio file (required)
- `--db-path` - Path to database file (default: `salina_vad.db`)

### Optional Parameters

- `--model` - Whisper model (default: `turbo`)
- `--device` - `cuda` or `cpu` (auto-detect if not specified)
- `--compute-type` - `float16`, `float32`, or `int8` (auto-detect if not specified)
- `--batch-size` - Batch size (default: `16`)
- `--output-dir` - Output directory (default: `output`)
- `--models-dir` - Models directory (default: `resources/models`)
- `--count` - Number of tasks to queue (default: `1`)

## Common Workflow

# Terminal 1: Start worker
huey_worker.exe --ffmpeg-path "C:\ffmpeg\bin\ffmpeg.exe"

# Terminal 2: Queue task
queue_simulator.exe --audio-file "C:\audio\video.mp4" --model turbo

# Or queue from database (Frontend use case)
queue_simulator.exe --process-id "process_123" --audio-file "C:\audio\video.mp4" --db-path "salina_vad.db"## Available Models

`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `turbo` (default)

## Arguments

### huey_worker.exe
- `--ffmpeg-path` - Path to ffmpeg executable

### queue_simulator.exe
- `--audio-file` - Audio file path (required for database workflow)
- `--payload` - Path to JSON payload file
- `--payload-json` - JSON payload as string
- `--process-id` - Process ID to fetch from database (requires `--audio-file`)
- `--db-path` - Database file path (default: `salina_vad.db`)
- `--count` - Number of tasks to queue (default: `1`)
- `--model` - Whisper model (default: `turbo`)
- `--device` - Device: `cuda` or `cpu` (default: `cpu` if not specified, use ResourceTracker.exe to detect GPU)
- `--compute-type` - `float16`, `float32`, or `int8` (auto-detect if not specified)
- `--batch-size` - Batch size (default: `16`)
- `--output-dir` - Output directory (default: `output`)
- `--models-dir` - Models directory (default: `resources/models`)
- `--delay` - Delay between queuing tasks in seconds (default: `0.0`)