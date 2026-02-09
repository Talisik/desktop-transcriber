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
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

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
    Transcribe payload with audio chunks (sequential processing).
    
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
    
    # process each chunk sequentially
    for idx, chunk in enumerate(payload_schema.merged_mappings, 1):
        print(f"\n📦 processing chunk {idx}/{len(payload_schema.merged_mappings)}")
        print(f"   audio_file: {Path(chunk.audio_file).name}")
        print(f"   language: {chunk.language_code}")
        print(f"   time range: {chunk.start}s - {chunk.end}s")
        
        try:
            # create transcriber instance
            transcriber = WhisperXTranscriber(device=device)
            
            # create transcription payload
            transcribe_payload = WhisperXTranscriberArgumentSchema(
                audio_file=chunk.audio_file,
                whisper_model=whisper_model,
                download_root=download_root,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size
            )
            
            # transcribe
            result, model = transcriber.transcribe(transcribe_payload)
            
            # extract segments and adjust timestamps
            chunk_segments = result.get("segments", [])
            for segment in chunk_segments:
                # adjust timestamps to absolute (add chunk start time)
                segment["start"] = segment.get("start", 0) + chunk.start
                segment["end"] = segment.get("end", 0) + chunk.start
                
                # adjust word timestamps too
                if "words" in segment:
                    for word in segment["words"]:
                        word["start"] = word.get("start", 0) + chunk.start
                        word["end"] = word.get("end", 0) + chunk.start
                
                all_segments.append(segment)
            
            # update chunk with results
            chunk.text = " ".join(seg.get("text", "") for seg in chunk_segments)
            chunk.segments = chunk_segments
            chunk.words = [word for seg in chunk_segments for word in seg.get("words", [])]
            chunk.transcription_backend = "whisperx"
            chunk.failed = False
            
            print(f"✓ chunk {idx} completed: {len(chunk_segments)} segments")
            
            # cleanup
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ chunk {idx} failed: {str(e)}")
            chunk.failed = True
            chunk.text = ""
            chunk.segments = []
            chunk.words = []
            continue
    
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

