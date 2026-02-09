#!/usr/bin/env python3
"""
Queue simulation script for Huey transcription tasks.
Use this to queue multiple tasks to SQLite for testing.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Import the transcriber_huey module to get the task
import transcriber_huey


def create_test_payload(
    process_id: str,
    audio_file: str,
    language_code: str = "en",
    start: float = 0.0,
    end: float = 60.0
) -> dict:
    """Create a test payload with a single chunk."""
    return {
        "language_stats": {
            "major_language_percentage": 100,
            "language_breakdown": {
                "counts": {language_code: 1},
                "durations": {language_code: end - start},
                "percentages": {language_code: 100}
            }
        },
        "language_code": language_code,
        "process_id": process_id,
        "language_classification": "single_language",
        "merged_mappings": [
            {
                "failed": False,
                "audio_file": audio_file,
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
        ]
    }


def queue_single_task(
    payload: dict,
    model: str = "turbo",
    device: str = None,
    compute_type: str = None,
    batch_size: int = 16,
    use_diarization: bool = False,
    hf_token: str = None,
    output_dir: str = "output",
    models_dir: str = None
) -> str:
    """Queue a single transcription task."""
    import torch
    
    # auto-detect device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # auto-detect compute_type if not provided
    if compute_type is None:
        compute_type = "float32" if device == "cpu" else "float16"
    
    # get models_dir
    if models_dir is None:
        models_dir = os.getenv("MODELS_DIR", "resources/models")
    
    # get hf token
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
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
        output_dir=output_dir
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
        help="Process ID for test payload (auto-generated if not provided)"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Audio file path (required if creating test payload)"
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
        "--delay",
        type=float,
        default=0.0,
        help="Delay between queuing tasks in seconds (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # get payload
    if args.payload_json:
        base_payload = json.loads(args.payload_json)
    elif args.payload:
        with open(args.payload, "r") as f:
            base_payload = json.load(f)
    elif args.audio_file:
        # create test payload
        process_id = args.process_id or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_payload = create_test_payload(
            process_id=process_id,
            audio_file=args.audio_file,
            language_code="en"
        )
    else:
        parser.error("Must provide --payload, --payload-json, or --audio-file")
    
    # queue tasks
    print(f"📤 queuing {args.count} task(s) to SQLite")
    print(f"   model: {args.model}")
    print(f"   output_dir: {args.output_dir}")
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
            output_dir=args.output_dir,
            models_dir=args.models_dir
        )
        
        task_ids.append(task_id)
        process_id = payload.get("process_id", "unknown")
        print(f"✅ [{i+1}/{args.count}] queued: {process_id} (task_id: {task_id})")
        
        if args.delay > 0 and i < args.count - 1:
            import time
            time.sleep(args.delay)
    
    print()
    print(f"🎉 queued {len(task_ids)} task(s) successfully")
    print(f"   run worker: python huey_worker.py")
    
    # get queue db filename from env
    env = os.getenv("ENV") or os.getenv("ENVIRONMENT") or "default"
    db_filename = f"transcribe_queue_{env}.db"
    print(f"   queue db: {db_filename}")


if __name__ == "__main__":
    main()

