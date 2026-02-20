#!/usr/bin/env python3
"""
Queue simulation script for Huey transcription tasks.
Use this to queue multiple tasks to SQLite for testing.
"""
import json
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

# Import the transcriber_huey module to get the task
import transcriber_huey

# Import payload generator functions
from payload_generator import (
    calculate_language_stats,
    map_chunks_to_merged_mappings
)


def create_test_payload(
    process_id: str,
    audio_file: str,
    language_code: str = "en",
    start: float = 0.0,
    end: float = 60.0,
    diarized: bool = False,
    transcriber: str = "whisperx"
) -> dict:
    """Create a test payload with a single chunk."""
    # Convert relative paths to absolute to ensure worker can find files
    audio_path = Path(audio_file).resolve()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    return {
        "transcriber": transcriber,
        "process_id": process_id,
        "file": str(audio_path),
        "file_id": None,
        "chat_room_id": None,
        "is_request_reprocess": False,
        "diarized": diarized,
        "language_code": language_code,
        "mappings": [],
        "merged_mappings": [
            {
                "failed": False,
                "audio_file": str(audio_path),  # Use absolute path
                "language_code": language_code,
                "text": "",
                "confidence": 0.9,
                "language_confidence": 0.92,
                "transcription_backend": "sieve",
                "segments": [],
                "words": [],
                "start": start,
                "end": end,
                "duration": end - start
            }
        ],
        "language_classification": "single_language",
        "language_stats": {
            "major_language_percentage": 100,
            "language_breakdown": {
                "counts": {language_code: 1},
                "durations": {language_code: end - start},
                "percentages": {language_code: 100}
            }
        }
    }


def get_db_row_by_process_id(
    process_id: str,
    db_path: str = "salina_vad.db"
) -> Optional[tuple]:
    """
    Query database for row by process_id.
    
    Args:
        process_id: Process ID to search for
        db_path: Path to database file
    
    Returns:
        Database row tuple or None if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM salina_languini_results WHERE process_id = ?",
            (process_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return row
    except (sqlite3.Error, FileNotFoundError) as e:
        print(f"Error querying database: {e}")
        return None


def build_payload_from_db_row(
    db_row: tuple,
    audio_file_path: str,
    diarized: bool = False,
    transcriber: str = "whisperx"
) -> Dict[str, Any]:
    """
    Build transcription payload from database row.
    
    Args:
        db_row: Database row tuple (id, process_id, vad_result_id, language_results_json, 
                metadata, status, created_at, updated_at, error_message)
        audio_file_path: Path to audio file
        diarized: Enable speaker diarization
        transcriber: Transcriber identifier
    
    Returns:
        Dict matching TranscriptionPayloadSchema format
    
    Raises:
        ValueError: If status is not 'completed'
        FileNotFoundError: If audio file doesn't exist
    """
    # Extract fields from row
    # Row structure: (id, process_id, vad_result_id, language_results_json, metadata, 
    #                 status, created_at, updated_at, error_message)
    process_id = db_row[1]
    language_results_json_str = db_row[3]
    metadata_str = db_row[4]
    status = db_row[5]
    
    # Check status
    if status != "completed":
        raise ValueError(f"Process ID '{process_id}' status is '{status}', expected 'completed'")
    
    # Convert relative paths to absolute to ensure worker can find files
    audio_path = Path(audio_file_path).resolve()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Parse JSON strings
    try:
        language_chunks = json.loads(language_results_json_str) if language_results_json_str else []
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid language_results_json for process_id '{process_id}': {e}")
    
    try:
        metadata = json.loads(metadata_str) if metadata_str else {}
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid metadata for process_id '{process_id}': {e}")
    
    # Calculate language stats
    language_stats = calculate_language_stats(language_chunks, metadata)
    
    # Determine major language
    language_breakdown = language_stats.get("language_breakdown", {})
    durations = language_breakdown.get("durations", {})
    major_language = max(durations.items(), key=lambda x: x[1])[0] if durations else "unknown"
    
    # Determine language classification
    multilingual_mode = metadata.get("multilingual_mode", False)
    language_classification = "multilingual" if multilingual_mode else "single_language"
    
    # Map chunks to merged_mappings (use absolute path)
    merged_mappings = map_chunks_to_merged_mappings(language_chunks, str(audio_path))
    
    # Build payload
    payload = {
        "transcriber": transcriber,
        "process_id": process_id,
        "file": str(audio_path),
        "file_id": None,
        "chat_room_id": None,
        "is_request_reprocess": False,
        "diarized": diarized,
        "language_code": major_language,
        "mappings": [],
        "merged_mappings": merged_mappings,
        "language_classification": language_classification,
        "language_stats": language_stats
    }
    
    return payload


def queue_single_task(
    payload: dict,
    model: str = "turbo",
    device: str = None,
    compute_type: str = None,
    batch_size: int = 16,
    use_diarization: bool = False,
    hf_token: str = None,
    output_dir: str = "output",
    models_dir: str = None,
    model_path: str = None
) -> str:
    """Queue a single transcription task."""
    import torch
    
    # auto-detect device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # User explicitly set device - respect it even if PyTorch can't detect CUDA
        # Let WhisperX handle the error if CUDA isn't actually available
        if device == "cuda" and not torch.cuda.is_available():
            print(f"WARNING: CUDA requested but PyTorch reports CUDA unavailable.")
            print(f"   PyTorch version: {torch.__version__}")
            print(f"   CUDA compiled: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
            print(f"   Attempting to use CUDA anyway - WhisperX will handle errors if GPU unavailable.")
            # Don't fall back - let the user's explicit choice stand
    
    # Warn if CUDA requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: CUDA requested but not available. PyTorch reports CUDA unavailable.")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA compiled: {torch.version.cuda}")
        print(f"   Falling back to CPU. Check nvidia drivers and GPU availability.")
        device = "cpu"
    
    # auto-detect compute_type if not provided
    if compute_type is None:
        compute_type = "float32" if device == "cpu" else "float16"
    
    # get models_dir
    if models_dir is None:
        models_dir = os.getenv("MODELS_DIR", "resources/models")
    
    # get hf token
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
    # Inject diarization and transcriber fields into payload
    payload["diarized"] = use_diarization
    payload["transcriber"] = f"whisper_{model}"
    
    # Ensure 'file' field is set
    if "file" not in payload and "merged_mappings" in payload:
        if payload["merged_mappings"]:
            payload["file"] = payload["merged_mappings"][0]["audio_file"]
    
    # queue task
    task = transcriber_huey.transcribe_payload_task(
        payload=payload,
        whisper_model=model,
        download_root=models_dir,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        use_diarization=use_diarization,
        hf_token=hf_token,
        output_dir=output_dir,
        model_path=model_path
    )
    
    return task.id


def main():
    parser = argparse.ArgumentParser(
        description="Queue transcription tasks to SQLite for testing"
    )
    parser.add_argument(
        "--payload",
        type=str,
        help="Path to JSON payload file"
    )
    parser.add_argument(
        "--payload-json",
        type=str,
        help="JSON payload as string"
    )
    parser.add_argument(
        "--process-id",
        type=str,
        help="Process ID to look up in database (or auto-generated for test payload)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="salina_vad.db",
        help="Path to database file (default: salina_vad.db)"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Audio file path (required if creating test payload or using --process-id)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of tasks to queue (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="turbo",
        help="Whisper model to use (default: turbo)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detects if not specified)"
    )
    parser.add_argument(
        "--compute-type",
        choices=["float16", "float32", "int8"],
        default=None,
        help="Compute type (auto-detects if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transcription (default: 16)"
    )
    parser.add_argument(
        "--use-diarization",
        action="store_true",
        help="Enable speaker diarization (requires HF_TOKEN env var or --hf-token)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for diarization models (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for transcript files (default: output)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory for model files (default: resources/models)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to directory containing existing models (fallback when model not found in models-dir)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between queuing tasks in seconds (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # get payload
    if args.payload_json:
        base_payload = json.loads(args.payload_json)
        # Convert relative audio_file paths to absolute to ensure worker can find files
        if "merged_mappings" in base_payload:
            for mapping in base_payload["merged_mappings"]:
                if "audio_file" in mapping:
                    # Always resolve to absolute path (let worker handle file not found errors)
                    audio_path = Path(mapping["audio_file"]).resolve()
                    mapping["audio_file"] = str(audio_path)
    elif args.payload:
        with open(args.payload, "r") as f:
            base_payload = json.load(f)
        
        # Convert relative audio_file paths to absolute to ensure worker can find files
        if "merged_mappings" in base_payload:
            for mapping in base_payload["merged_mappings"]:
                if "audio_file" in mapping:
                    # Always resolve to absolute path (let worker handle file not found errors)
                    audio_path = Path(mapping["audio_file"]).resolve()
                    mapping["audio_file"] = str(audio_path)
    elif args.process_id and args.audio_file:
        # Build payload from database
        print(f"loading process_id from database: {args.process_id}")
        row = get_db_row_by_process_id(args.process_id, args.db_path)
        if row is None:
            parser.error(f"Process ID '{args.process_id}' not found in database '{args.db_path}'")
        
        try:
            base_payload = build_payload_from_db_row(row, args.audio_file)
            print(f"payload built from database")
            print(f"   chunks: {len(base_payload['merged_mappings'])}")
            print(f"   language: {base_payload['language_code']}")
            print(f"   classification: {base_payload['language_classification']}")
        except ValueError as e:
            parser.error(str(e))
        except FileNotFoundError as e:
            parser.error(str(e))
    elif args.audio_file:
        # create test payload
        process_id = args.process_id or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            base_payload = create_test_payload(
                process_id=process_id,
                audio_file=args.audio_file,
                language_code="en"
            )
        except FileNotFoundError as e:
            parser.error(str(e))
    else:
        parser.error("Must provide --payload, --payload-json, --audio-file, or --process-id with --audio-file")
    
    # queue tasks
    print(f"queuing {args.count} task(s) to SQLite")
    print(f"   model: {args.model}")
    print(f"   output_dir: {args.output_dir}")
    
    # Log audio file info from base payload
    if "merged_mappings" in base_payload:
        audio_files = [m.get("audio_file") for m in base_payload["merged_mappings"] if m.get("audio_file")]
        if audio_files:
            unique_files = list(set(audio_files))
            if len(unique_files) == 1:
                print(f"   audio_file: {unique_files[0]}")
            else:
                print(f"   audio_files: {len(unique_files)} unique file(s) across {len(base_payload['merged_mappings'])} chunk(s)")
    
    print()
    
    task_ids = []
    for i in range(args.count):
        # create unique process_id for each task
        if args.count > 1:
            payload = base_payload.copy()
            original_process_id = payload.get("process_id", "test")
            payload["process_id"] = f"{original_process_id}_{i+1:03d}"
        else:
            payload = base_payload
        
        task_id = queue_single_task(
            payload=payload,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            use_diarization=args.use_diarization,
            hf_token=args.hf_token,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            model_path=args.model_path
        )
        
        task_ids.append(task_id)
        process_id = payload.get("process_id", "unknown")
        
        # Log audio file path(s) from payload
        audio_files = []
        if "merged_mappings" in payload:
            for mapping in payload["merged_mappings"]:
                if "audio_file" in mapping:
                    audio_files.append(mapping["audio_file"])
        
        if audio_files:
            # Show unique audio files (in case multiple chunks use same file)
            unique_files = list(set(audio_files))
            if len(unique_files) == 1:
                print(f"[{i+1}/{args.count}] queued: {process_id} (task_id: {task_id})")
                print(f"   audio_file: {unique_files[0]}")
            else:
                print(f"[{i+1}/{args.count}] queued: {process_id} (task_id: {task_id})")
                print(f"   audio_files: {len(unique_files)} unique file(s)")
                for audio_file in unique_files:
                    print(f"      - {audio_file}")
        else:
            print(f"[{i+1}/{args.count}] queued: {process_id} (task_id: {task_id})")
            print(f"   warning: no audio_file found in payload")
        
        if args.delay > 0 and i < args.count - 1:
            import time
            time.sleep(args.delay)
    
    print()
    print(f"queued {len(task_ids)} task(s) successfully")
    print(f"   run worker: python huey_worker.py")
    
    # get queue db filename from env
    env = os.getenv("ENV") or os.getenv("ENVIRONMENT") or "default"
    db_filename = f"transcribe_queue_{env}.db"
    print(f"   queue db: {db_filename}")


if __name__ == "__main__":
    main()

