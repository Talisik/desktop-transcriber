#!/usr/bin/env python3
"""
Lightweight queue module for transcriber_huey tasks.
This module only contains the huey instance and task registration - no ML imports.
Used by queue_simulator to enqueue tasks without pulling in torch/whisperx/etc.

The worker will import the real implementation from transcriber_huey.py,
which will override this stub task registration.
"""
import os
from huey import SqliteHuey
from typing import Dict, Any

# Huey configuration (same as transcriber_huey.py)
env = os.getenv("ENV") or os.getenv("ENVIRONMENT") or "default"
db_filename = f"transcribe_queue_{env}.db"
huey = SqliteHuey(filename=db_filename)


# Register a stub task with the same name as the real task
# This allows queue_simulator to enqueue tasks without importing ML libraries
# The worker will import the real implementation which will override this stub
def _transcribe_payload_task_stub(
    payload: Dict[str, Any],
    whisper_model: str,
    download_root: str,
    device: str,
    compute_type: str,
    batch_size: int,
    use_diarization: bool,
    hf_token: str | None,
    output_dir: str,
    model_path: str | None = None
) -> str:
    """
    Stub task for queuing transcription tasks.
    
    This is a lightweight wrapper that allows queue_simulator to enqueue tasks
    without importing torch/whisperx/etc. The actual implementation is in
    transcriber_huey.py and will be executed by the worker.
    
    At runtime, this delegates to the real implementation from transcriber_huey.
    """
    # Dynamically import the real implementation at runtime
    # This avoids pulling in ML libraries during queue_simulator execution
    import transcriber_huey
    return transcriber_huey.transcribe_payload_task(
        payload=payload,
        whisper_model=whisper_model,
        download_root=download_root,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        use_diarization=use_diarization,
        hf_token=hf_token,
        output_dir=output_dir,
        model_path=model_path
    )

# Set __module__ to transcriber_huey so tasks are registered with the same name
_transcribe_payload_task_stub.__module__ = "transcriber_huey"

# Register the task - this will use transcriber_huey as the module name
transcribe_payload_task = huey.task(
    name='transcribe_payload_task', retries=2, retry_delay=60
)(_transcribe_payload_task_stub)

