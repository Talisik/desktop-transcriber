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


def process_file(
    file_path: str,
    device: str = "cpu",
    model: str = "base",
    hf_token: str | None = None,
    download_root: str = "resources/models",
) -> dict:
    """
    process a single file and return metrics.
    
    args:
        file_path: path to audio/video file
        device: device to use (cpu/cuda)
        model: whisper model size
        hf_token: huggingface token for diarization
        download_root: directory for model storage
    
    returns:
        dict with file, process_time, file_duration, and resource metrics
    """
    # initialize resource monitor
    monitor = ResourceMonitor(device=device)
    
    # get file duration
    file_duration = get_file_duration(file_path)
    
    # start monitoring
    monitor.start()
    start_time = time.perf_counter()
    
    # initialize timing variables
    transcription_time = 0.0
    diarization_time = 0.0
    
    try:
        # load models (transcribe() will load the transcription model)
        diarization_model = download_diarization_model(hf_token, device)
        
        # load audio
        audio = load_audio(file_path)
        
        # transcribe
        transcription_start = time.perf_counter()
        transcription_result, transcription_model = transcribe(
            audio, model, download_root, device=device
        )
        
        # align
        alignment_model, metadata = download_alignment_model(
            language_code=transcription_result["language"], device=device
        )
        alignment_result = align(
            transcription_result, (alignment_model, metadata), audio, device=device
        )

        transcription_time = time.perf_counter() - transcription_start
        
        # diarize
        diarization_start = time.perf_counter()
        final_result = diarize(alignment_result, diarization_model, audio)
        diarization_time = time.perf_counter() - diarization_start
        
        # cleanup
        del transcription_model, alignment_model, metadata, diarization_model, audio
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        process_time = time.perf_counter() - start_time
        monitor.stop()
        
        # get metrics
        metrics = monitor.get_metrics()
        
        # get resource limits from environment
        cpu_limit = os.getenv("DOCKER_CPU_LIMIT", "unknown")
        memory_limit = os.getenv("DOCKER_MEMORY_LIMIT", "unknown")
        
        return {
            "file": file_path,
            "device": device,
            "model": model,
            "process_time": round(process_time, 2),
            "transcription_time": round(transcription_time, 2),
            "diarization_time": round(diarization_time, 2),
            "file_duration": round(file_duration, 2),
            "resource_limits": {
                "cpus": cpu_limit,
                "memory": memory_limit
            },
            **metrics,
        }
    
    except Exception as e:
        monitor.stop()
        process_time = time.perf_counter() - start_time
        metrics = monitor.get_metrics()
        
        # get resource limits from environment
        cpu_limit = os.getenv("DOCKER_CPU_LIMIT", "unknown")
        memory_limit = os.getenv("DOCKER_MEMORY_LIMIT", "unknown")
        
        # timing variables already initialized, use current values (0.0 if not completed)
        
        return {
            "file": file_path,
            "device": device,
            "model": model,
            "process_time": round(process_time, 2),
            "transcription_time": transcription_time,
            "diarization_time": diarization_time,
            "file_duration": round(file_duration, 2),
            "resource_limits": {
                "cpus": cpu_limit,
                "memory": memory_limit
            },
            "error": str(e),
            **metrics,
        }


def process_batch(payload: list[dict], hf_token: str | None = None) -> dict:
    """
    process multiple files from json payload.
    
    args:
        payload: list of dicts with "file" and "device" keys
        hf_token: huggingface token for diarization
    
    returns:
        dict with summary and individual results
    """
    results = []
    
    for item in payload:
        file_path = item.get("file")
        device = item.get("device", "cpu")
        model = item.get("model", "tiny")
        
        if not file_path:
            continue
        
        print(f"processing {file_path} on {device}...")
        result = process_file(
            file_path=file_path,
            device=device,
            model=model,
            hf_token=hf_token,
        )
        
        # save individual result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("benchmark_results") / f"benchmark_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        results.append(result)
        print(f"✓ saved results to {output_file}")
    
    return {
        "total_files": len(results),
        "results": results,
    }


def main(payload: list[dict] | None = None):
    """
    main entry point. supports both single file and batch processing.
    
    args:
        payload: optional json payload for batch processing.
                 format: [{"file": "path/to/file.mp4", "device": "cpu"}, ...]
                 if None, processes single file with default settings
    """
    hf_token = "YOUR_HF_TOKEN_HERE"
    
    # batch processing mode
    if payload:
        print(f"processing batch of {len(payload)} files...")
        if torch.cuda.is_available():
            print(f"🚀 cuda available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  cuda not available, using cpu")
        summary = process_batch(payload, hf_token=hf_token)
        print(f"✓ processed {summary['total_files']} files")
        return summary
    
    # single file processing (existing behaviour)
    model = "turbo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"using model: {model} on {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 using cuda device: {torch.cuda.get_device_name(0)}")
        print(f"   gpu memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("⚠️  cuda not available, using cpu")
    
    result = process_file(
        file_path="resources/videos/sample_01.mp4",
        device=device,
        model=model,
        hf_token=hf_token,
    )
    
    # save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("benchmark_results") / f"benchmark_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print("finished transcribing using model:", model)
    print(f"results saved to {output_file}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
