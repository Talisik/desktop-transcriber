"""
Huey-based async transcription system.
Processes payloads with language-detected audio chunks and transcribes them using WhisperX.
"""
import torch

# CRITICAL: monkey patch torch.load to disable weights_only before any other imports
# pytorch 2.6 changed default to weights_only=True which breaks pyannote/whisperx models
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    """disable weights_only for model loading - we trust whisperx/pyannote"""
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

torch.load = _patched_torch_load

# also patch lightning_fabric if available
try:
    import lightning_fabric.utilities.cloud_io as cloud_io
    cloud_io._load = _patched_torch_load
except ImportError:
    pass

# now safe to import everything else
import json
import os
import sys
import gc
import subprocess
import tempfile
import multiprocessing
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# CRITICAL: Register module with correct name BEFORE any decorators run
# This ensures tasks are registered as 'transcriber_huey.task_name' even when run as script
if __name__ == "__main__":
    # Store original __name__ and register module with correct name
    _original_name = __name__
    sys.modules['transcriber_huey'] = sys.modules[__name__]
    # Temporarily set __name__ so decorators use correct module name
    # Note: We can't actually change __name__ (it's read-only), but registering in sys.modules helps

from pydantic import BaseModel
from huey import SqliteHuey

from transcriber.concretions.whisperx_implementation.whisperx_transcriber import WhisperXTranscriber
from transcriber.concretions.whisperx_implementation.chemas.payload.transcriber_argument_schema import WhisperXTranscriberArgumentSchema
from transcriber.concretions.whisperx_implementation.chemas.custom_types.parameter_types import Device, ComputeType, WhisperModel

# Huey configuration
env = os.getenv("ENV") or os.getenv("ENVIRONMENT") or "default"
db_filename = f"transcribe_queue_{env}.db"
huey = SqliteHuey(filename=db_filename)

# Module-level ffmpeg path (set by worker)
_ffmpeg_path: str | None = None

def set_ffmpeg_path(path: str | None):
    """Set ffmpeg path for the worker."""
    global _ffmpeg_path
    if path:
        # Validate and normalize path (includes existence check and --version test)
        _ffmpeg_path = validate_ffmpeg_path(path)
    else:
        _ffmpeg_path = None


def get_ffmpeg_path() -> str | None:
    """Get the current ffmpeg path."""
    return _ffmpeg_path


# Payload schemas
class MergedMappingSchema(BaseModel):
    failed: bool
    audio_file: str
    language_code: str
    text: str
    confidence: float | None = None
    language_confidence: float | None = None
    transcription_backend: str | None = None
    segments: list[dict] = []
    words: list[dict] = []
    start: float
    end: float
    duration: float


class TranscriptionPayloadSchema(BaseModel):
    language_stats: dict
    language_code: str
    process_id: str
    language_classification: str
    merged_mappings: list[MergedMappingSchema]


def _find_ffmpeg() -> str:
    """
    Find ffmpeg executable in PATH or common locations.
    
    Returns:
        Path to ffmpeg executable
    
    Raises:
        RuntimeError: If ffmpeg is not found
    """
    # Try to find ffmpeg in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # On Windows, try ffmpeg.exe
    if sys.platform == 'win32':
        ffmpeg_path = shutil.which('ffmpeg.exe')
        if ffmpeg_path:
            return ffmpeg_path
        
        # Try common Windows installation locations
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    
    raise RuntimeError(
        "ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.\n"
        "Windows: Download from https://ffmpeg.org/download.html and add to PATH\n"
        "Linux: sudo apt-get install ffmpeg\n"
        "macOS: brew install ffmpeg"
    )


def validate_ffmpeg_path(path: str) -> str:
    """
    Validate and normalize ffmpeg path.
    
    Args:
        path: Path to ffmpeg executable or directory containing it
        
    Returns:
        Normalized path to ffmpeg executable
        
    Raises:
        RuntimeError: If ffmpeg is not found or not executable
    """
    # Normalize path
    normalized_path = os.path.normpath(os.path.expanduser(path))
    
    # Check if path exists
    if not os.path.exists(normalized_path):
        raise RuntimeError(f"ffmpeg not found at specified path: {normalized_path}")
    
    # Handle directory paths - look for ffmpeg executable inside
    if os.path.isdir(normalized_path):
        if sys.platform == 'win32':
            # Look for ffmpeg.exe in directory
            exe_path = os.path.join(normalized_path, 'ffmpeg.exe')
            if os.path.exists(exe_path) and os.path.isfile(exe_path):
                normalized_path = exe_path
            else:
                raise RuntimeError(f"ffmpeg path is a directory, but ffmpeg.exe not found inside: {normalized_path}")
        else:
            # Look for ffmpeg in directory (Linux/Mac)
            exe_path = os.path.join(normalized_path, 'ffmpeg')
            if os.path.exists(exe_path) and os.path.isfile(exe_path):
                normalized_path = exe_path
            else:
                raise RuntimeError(f"ffmpeg path is a directory, but ffmpeg not found inside: {normalized_path}")
    
    # Validate it's a file (not a directory after resolution)
    if not os.path.isfile(normalized_path):
        raise RuntimeError(f"ffmpeg path is not a file: {normalized_path}")
    
    # On Windows, ensure .exe extension if not present
    if sys.platform == 'win32':
        if not normalized_path.lower().endswith('.exe'):
            # Try adding .exe
            exe_path = normalized_path + '.exe'
            if os.path.exists(exe_path) and os.path.isfile(exe_path):
                normalized_path = exe_path
            else:
                raise RuntimeError(f"ffmpeg executable not found: {normalized_path} (tried {exe_path})")
    
    # Test that it's actually ffmpeg by running --version
    try:
        result = subprocess.run(
            [normalized_path, '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg at {normalized_path} failed to run (exit code: {result.returncode})")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg at {normalized_path} timed out when testing")
    except FileNotFoundError:
        raise RuntimeError(f"ffmpeg executable not found: {normalized_path}")
    except Exception as e:
        raise RuntimeError(f"Error validating ffmpeg at {normalized_path}: {e}")
    
    return normalized_path


def extract_audio_segment(
    audio_file: str,
    start: float,
    end: float,
    output_file: str | None = None
) -> str:
    """
    Extract audio segment using ffmpeg.
    
    Args:
        audio_file: Path to source audio file
        start: Start time in seconds
        end: End time in seconds
        output_file: Output file path (creates temp file if None)
    
    Returns:
        Path to extracted segment file
    """
    if output_file is None:
        # Create temp file
        temp_fd, output_file = tempfile.mkstemp(suffix='.wav', prefix='chunk_')
        os.close(temp_fd)
    
    duration = end - start
    
    # Get ffmpeg path: command line arg > environment variable > auto-detection
    if _ffmpeg_path:
        # Normalize path again in case it was set before normalization was added
        ffmpeg_exe = os.path.normpath(_ffmpeg_path)
        if not os.path.exists(ffmpeg_exe):
            raise RuntimeError(f"ffmpeg not found at specified path: {ffmpeg_exe}")
    else:
        # Check environment variable
        ffmpeg_path = os.getenv("FFMPEG_PATH")
        if ffmpeg_path:
            # Normalize environment variable path
            ffmpeg_exe = os.path.normpath(ffmpeg_path)
            if not os.path.exists(ffmpeg_exe):
                raise RuntimeError(f"ffmpeg not found at FFMPEG_PATH: {ffmpeg_exe}")
        else:
            ffmpeg_exe = _find_ffmpeg()  # Fallback to auto-detection
    
    cmd = [
        ffmpeg_exe,
        '-i', audio_file,
        '-ss', str(start),
        '-t', str(duration),
        '-acodec', 'pcm_s16le',  # wav format
        '-ar', '16000',  # sample rate for whisper
        '-ac', '1',  # mono
        '-y',  # overwrite
        output_file
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to extract audio segment: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")


def _process_single_chunk(
    chunk_data: tuple,
    whisper_model: str,
    download_root: str,
    device: str,
    compute_type: str,
    batch_size: int
) -> tuple:
    """
    Process a single chunk in a worker thread.
    
    Args:
        chunk_data: Tuple of (chunk_idx, chunk, total_chunks, audio_file_path)
        whisper_model: Whisper model name
        download_root: Model download directory
        device: Device to use
        compute_type: Compute type
        batch_size: Batch size
    
    Returns:
        Tuple of (chunk_idx, chunk, segments, success, error_message)
    """
    chunk_idx, chunk, total_chunks, audio_file_path = chunk_data
    
    print(f"\n📦 processing chunk {chunk_idx}/{total_chunks} (thread)")
    print(f"   audio_file: {Path(chunk.audio_file).name}")
    print(f"   language: {chunk.language_code}")
    print(f"   time range: {chunk.start}s - {chunk.end}s")
    
    temp_segment_file = None
    try:
        # Use audio file path as-is (no forced absolute path conversion)
        audio_file = Path(chunk.audio_file)
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Extract audio segment
        print(f"   extracting segment: {chunk.start}s - {chunk.end}s")
        temp_segment_file = extract_audio_segment(
            audio_file=str(audio_file),  # Use path as-is
            start=chunk.start,
            end=chunk.end
        )
        
        # Create transcriber instance for this thread
        # Each thread gets its own instance to avoid conflicts
        transcriber = WhisperXTranscriber(device=device)
        
        # Create transcription payload
        transcribe_payload = WhisperXTranscriberArgumentSchema(
            audio_file=temp_segment_file,
            whisper_model=whisper_model,
            download_root=download_root,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=chunk.language_code
        )
        
        # Transcribe
        result, model = transcriber.transcribe(transcribe_payload)
        
        # Extract segments and adjust timestamps
        chunk_segments = result.get("segments", [])
        for segment in chunk_segments:
            segment["start"] = segment.get("start", 0) + chunk.start
            segment["end"] = segment.get("end", 0) + chunk.start
            
            if "words" in segment:
                for word in segment["words"]:
                    word["start"] = word.get("start", 0) + chunk.start
                    word["end"] = word.get("end", 0) + chunk.start
        
        # Update chunk
        chunk.text = " ".join(seg.get("text", "") for seg in chunk_segments)
        chunk.segments = chunk_segments
        chunk.words = [word for seg in chunk_segments for word in seg.get("words", [])]
        chunk.transcription_backend = "whisperx"
        chunk.failed = False
        
        print(f"✓ chunk {chunk_idx} completed: {len(chunk_segments)} segments")
        
        # Cleanup
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return (chunk_idx, chunk, chunk_segments, True, None)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ chunk {chunk_idx} failed: {error_msg}")
        chunk.failed = True
        chunk.text = ""
        chunk.segments = []
        chunk.words = []
        return (chunk_idx, chunk, [], False, error_msg)
        
    finally:
        # Cleanup temp file
        if temp_segment_file and os.path.exists(temp_segment_file):
            try:
                os.unlink(temp_segment_file)
            except Exception as e:
                print(f"⚠️  warning: failed to delete temp file {temp_segment_file}: {e}")


def _transcribe_payload_task_impl(
    payload: dict,
    whisper_model: str,
    download_root: str,
    device: str,
    compute_type: str,
    batch_size: int,
    use_diarization: bool,
    hf_token: str | None,
    output_dir: str
) -> str:
    """
    Transcribe payload with audio chunks (parallel processing).
    
    Returns:
        Path to saved transcript JSON file
    """
    print(f"🎬 starting transcription task")
    
    # validate payload
    payload_schema = TranscriptionPayloadSchema(**payload)
    process_id = payload_schema.process_id
    print(f"   process_id: {process_id}")
    print(f"   chunks: {len(payload_schema.merged_mappings)}")
    
    # collect all segments from all chunks
    all_segments = []
    
    # Determine number of workers (auto-detect or use config)
    # For CPU: use all cores, for CUDA: limit to avoid memory issues
    if device == "cuda":
        # Limit to 2-4 workers for GPU to avoid memory conflicts
        max_workers = min(4, len(payload_schema.merged_mappings), multiprocessing.cpu_count())
    else:
        # Use all cores for CPU
        max_workers = min(multiprocessing.cpu_count(), len(payload_schema.merged_mappings))
    
    print(f"   using {max_workers} worker thread(s) for parallel processing")
    
    # Prepare chunk data for workers
    chunk_data_list = [
        (idx, chunk, len(payload_schema.merged_mappings), chunk.audio_file)
        for idx, chunk in enumerate(payload_schema.merged_mappings, 1)
    ]
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(
                _process_single_chunk,
                chunk_data,
                whisper_model,
                download_root,
                device,
                compute_type,
                batch_size
            ): chunk_data[0]  # chunk_idx
            for chunk_data in chunk_data_list
        }
        
        # Collect results as they complete
        results = {}
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_idx, chunk, chunk_segments, success, error_msg = future.result()
                results[chunk_idx] = (chunk, chunk_segments, success, error_msg)
                
                # Add segments to all_segments
                all_segments.extend(chunk_segments)
                
            except Exception as e:
                print(f"❌ chunk {chunk_idx} exception: {str(e)}")
                # Get chunk from original list
                chunk = payload_schema.merged_mappings[chunk_idx - 1]
                chunk.failed = True
                chunk.text = ""
                chunk.segments = []
                chunk.words = []
                results[chunk_idx] = (chunk, [], False, str(e))
    
    # sort segments chronologically
    all_segments.sort(key=lambda x: x.get("start", 0))
    
    print(f"\n📊 merging results:")
    print(f"   total segments: {len(all_segments)}")
    
    # build transcript output
    machine_name = os.getenv("MACHINE_NAME", "unknown")
    video_file = f"process_{process_id}"  # placeholder, can be extracted from process_id if needed
    
    transcript_data = {
        "process_id": process_id,
        "machine_name": machine_name,
        "video_file": video_file,
        "model_name": whisper_model,
        "language": payload_schema.language_code,
        "segments": all_segments
    }
    
    # save transcript
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{process_id}_transcript.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(transcript_data, f, indent=2)
    
    print(f"✅ transcript saved: {filepath}")
    
    return str(filepath)


# Create task with explicit module name
# Ensure function has correct __module__ attribute for registration
if __name__ == "__main__":
    _transcribe_payload_task_impl.__module__ = "transcriber_huey"

# Register task - this ensures consistent naming
transcribe_payload_task = huey.task(name='transcribe_payload_task', retries=2, retry_delay=60)(_transcribe_payload_task_impl)


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="queue transcription tasks using huey")
    parser.add_argument(
        "--payload",
        type=str,
        help="path to JSON payload file (or '-' for stdin)"
    )
    parser.add_argument(
        "--payload-json",
        type=str,
        help="JSON payload as string"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="whisper model to use (default: base)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="device to use (auto-detects if not specified)"
    )
    parser.add_argument(
        "--compute-type",
        choices=["float16", "float32", "int8"],
        default=None,
        help="compute type (auto-detects if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size for transcription (default: 16)"
    )
    parser.add_argument(
        "--use-diarization",
        action="store_true",
        help="enable speaker diarization (requires HF_TOKEN env var)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="huggingface token for diarization models (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="output directory for transcript files (default: output)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.getenv("MODELS_DIR", "resources/models"),
        help="directory for model files (default: resources/models)"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="wait for task to complete (blocking)"
    )
    
    args = parser.parse_args()
    
    # get payload
    if args.payload_json:
        payload = json.loads(args.payload_json)
    elif args.payload:
        if args.payload == "-":
            payload = json.load(sys.stdin)
        else:
            with open(args.payload, "r") as f:
                payload = json.load(f)
    else:
        parser.error("must provide --payload or --payload-json")
    
    # auto-detect device if not provided
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # auto-detect compute_type if not provided
    if args.compute_type is None:
        args.compute_type = "float32" if args.device == "cpu" else "float16"
    
    # get hf token
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if args.use_diarization and not hf_token:
        parser.error("HF_TOKEN required for diarization (use --hf-token or set HF_TOKEN env var)")
    
    # queue task
    print(f"📤 queuing transcription task")
    print(f"   process_id: {payload.get('process_id', 'unknown')}")
    print(f"   model: {args.model}")
    print(f"   device: {args.device}")
    print(f"   diarization: {'enabled' if args.use_diarization else 'disabled'}")
    
    # when run as script, get task from registered module to ensure correct name
    if __name__ == "__main__":
        import transcriber_huey
        task_func = transcriber_huey.transcribe_payload_task
    else:
        task_func = transcribe_payload_task
    
    task = task_func(
        payload=payload,
        whisper_model=args.model,
        download_root=args.models_dir,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        use_diarization=args.use_diarization,
        hf_token=hf_token,
        output_dir=args.output_dir
    )
    
    if args.wait:
        print(f"⏳ waiting for task to complete...")
        result = task.get(blocking=True, timeout=3600)  # 1 hour timeout
        print(f"✅ task completed")
        print(f"📄 result: {result}")
    else:
        print(f"✅ task queued (task_id: {task.id})")
        print(f"   run worker: python huey_worker.py")


if __name__ == "__main__":
    main()

