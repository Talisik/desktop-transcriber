import json
import os
import time
from datetime import datetime
from pathlib import Path
from transcriber.whisperx_main import (
    load_audio,
    transcribe,
    align,
    diarize,
    download_diarization_model,
    download_alignment_model,
)
from transcriber.metrics import ResourceMonitor, get_file_duration
import gc
import torch

from transcriber.concretions.whisperx_implementation.whisperx_transcriber import WhisperXTranscriber
from transcriber.concretions.whisperx_implementation.chemas.payload.transcriber_argument_schema import WhisperXTranscriberArgumentSchema
from transcriber.concretions.whisperx_implementation.chemas.custom_types.parameter_types import Device, ComputeType, WhisperModel

from resource_tracker.concretions.desktop_resource_tracker import DesktopResourceTracker
from resource_tracker.model_profiles.model_profile_reader import ModelProfileReader

def main():

    resource_tracker = DesktopResourceTracker()
    print("Total RAM: ", resource_tracker.get_total_ram())
    print("Total CPU: ", resource_tracker.get_total_cpu())
    print("Has GPU: ", resource_tracker.has_gpu())
    print("GPU VRAM: ", resource_tracker.get_gpu_vram())

    model_profile_reader = ModelProfileReader()
    result = model_profile_reader.get_compatible_models_balanced(resource_tracker=resource_tracker)
    print("Model compatibility: ", result)


if __name__ == "__main__":
    main()
