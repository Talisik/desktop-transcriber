# Executables Command Reference

## GPU Support

The Windows executables include CUDA-enabled PyTorch, enabling GPU acceleration for:
- **Transcription**: Uses CTranslate2 (works independently of PyTorch CUDA status)
- **Alignment**: Uses PyTorch (requires CUDA-enabled PyTorch - included in build)
- **Diarization**: Uses PyTorch/pyannote (requires CUDA-enabled PyTorch - included in build)

### Executable Size
- Executables are **~200-300MB larger** due to CUDA-enabled PyTorch libraries
- Total size: ~1GB+ (includes all ML dependencies)
- Uses **onedir mode** (directory with exe + dependencies) to avoid 4GB single-file limit

### GPU Requirements

#### CUDA Version Matching
**CRITICAL**: PyTorch CUDA version must match your installed CUDA toolkit version.

1. **Check installed CUDA version**:
   ```powershell
   nvcc --version
   ```
   Or check NVIDIA Control Panel → System Information

2. **Verify PyTorch CUDA version** (included in build):
   - Build uses PyTorch with CUDA 12.1 support
   - If your system has CUDA 11.x, you may need to rebuild with matching PyTorch version

3. **CUDA DLLs in PATH**:
   - CUDA DLLs are **NOT bundled** in the executable (loaded from system)
   - Ensure CUDA bin directory is in system PATH:
     - Default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`
     - Add to PATH if not already present
   - Verify DLLs are accessible:
     ```powershell
     Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cublas64_12.dll"
     ```

#### GPU Detection
- **NVIDIA GPU** with CUDA drivers installed
- Use `ResourceTracker.exe` to detect GPU availability
- Pass `--device cuda` to `queue_simulator.exe` when GPU is available

#### Common CUDA Issues

**Error: "CUDA not available" or "No CUDA runtime is found"**
- Verify CUDA toolkit is installed: `nvcc --version`
- Check CUDA bin directory is in PATH
- Verify GPU is detected: `nvidia-smi`
- Ensure PyTorch CUDA version matches system CUDA version

**Error: "DLL load failed" or missing CUDA DLLs**
- Add CUDA bin directory to system PATH
- Restart terminal/application after PATH changes
- Verify DLLs exist in CUDA installation directory

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
huey_worker.exe --ffmpeg-path "C:\path\to\ffmpeg.exe"

### Model Download Paths

**IMPORTANT**: Models are downloaded to a writable location outside the executable bundle.

- **Default location**: `resources/models/` (relative to executable directory)
- **Custom location**: Use `--models-dir` in queue_simulator to specify custom path
- **Writable requirement**: The model directory must be writable (not in Program Files without admin rights)

**Recommended locations**:
- `C:\Users\<username>\AppData\Local\transcriber\models\` (user-specific, always writable)
- `C:\transcriber\models\` (requires admin rights to create)
- Relative to executable: `.\resources\models\` (if executable is in writable location)

**Model cache structure**:
```
models/
├── models--Systran--faster-whisper-tiny/
├── models--Systran--faster-whisper-base/
└── ... (other models)
```

### PyInstaller Temp File Handling

The executable uses **onedir mode** (directory structure) to avoid issues with:
- **4GB file size limit**: Single-file executables can't exceed 4GB on Windows
- **Antivirus scanning**: Antivirus software may flag/scan large single-file executables
- **Startup performance**: Directory mode avoids extraction overhead on each run

**Directory structure**:
```
dist/
└── huey_worker/
    ├── huey_worker.exe
    ├── _internal/
    │   ├── (Python libraries)
    │   ├── (ML dependencies)
    │   └── onnx_model/  (munchkin_chunker model)
    └── (other dependencies)
```

**Antivirus considerations**:
- Some antivirus software may scan the `_internal` directory on first run
- This is normal and may cause a slight delay on first execution
- Add the executable directory to antivirus exclusions if needed for performance

## queue_simulator.exe

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
- `--models-dir` - Models directory (default: `resources/models`) - **Must be writable**
- `--delay` - Delay between queuing tasks in seconds (default: `0.0`)

## Windows-Specific Testing Checklist

When building and testing the Windows executable, verify:

### 1. ONNX Model Access
- [ ] Chunker works without errors (ONNX model should be in `_internal/onnx_model/`)
- [ ] No "onnx_model not found" errors during transcription

### 2. CUDA Functionality
- [ ] GPU detected correctly: `ResourceTracker.exe` shows CUDA availability
- [ ] Transcription with `--device cuda` works without DLL errors
- [ ] CUDA version matches: PyTorch CUDA version matches system CUDA toolkit
- [ ] No "CUDA not available" errors when GPU is present

### 3. Model Download Paths
- [ ] Models download to writable location (not in read-only Program Files)
- [ ] Custom `--models-dir` path works correctly
- [ ] No permission errors when downloading models
- [ ] Models persist between runs (not deleted)

### 4. FFmpeg Detection
- [ ] Auto-detection works if FFmpeg is in PATH
- [ ] `--ffmpeg-path` argument works correctly
- [ ] No "ffmpeg not found" errors during audio extraction

### 5. Temp File Handling
- [ ] Executable starts without antivirus blocking
- [ ] No excessive startup delay (first run may be slower due to antivirus scan)
- [ ] Directory structure is correct (`dist/huey_worker/` with `_internal/` subdirectory)

### 6. Build Verification
- [ ] Executable size is reasonable (< 4GB total in directory)
- [ ] All dependencies are included (no missing DLL errors)
- [ ] Worker processes tasks successfully end-to-end

### 7. TorchCodec / pyannote audio decoding (Windows)
- [ ] `dist/transcriber_huey/_internal/torchcodec/libtorchcodec_core*.dll` exists after build
- [ ] If you bundle FFmpeg DLLs at build time, confirm these exist next to the exe (or at the bundle root):
  - [ ] `avcodec-*.dll`
  - [ ] `avformat-*.dll`
  - [ ] `avutil-*.dll`
  - [ ] `swresample-*.dll`
  - [ ] `swscale-*.dll`
- [ ] Run `transcriber_huey.exe` on a clean Windows machine and confirm you **don’t** see:
  - `torchcodec is not installed correctly so built-in audio decoding will fail`
- [ ] If it still fails on only one machine: install/repair **Microsoft Visual C++ Redistributable** (common cause of “DLL present but won’t load”).
