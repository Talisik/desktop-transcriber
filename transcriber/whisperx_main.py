import torch

# monkey patch lightning_fabric's torch.load to disable weights_only
import lightning_fabric.utilities.cloud_io as cloud_io

_original_load = cloud_io._load

def _patched_load(path_or_url, map_location=None, weights_only=None):
    """disable weights_only for model loading - we trust whisperx/pyannote"""
    return torch.load(path_or_url, map_location=map_location, weights_only=False)

cloud_io._load = _patched_load

# now import the rest
import whisperx
import gc
from whisperx.diarize import DiarizationPipeline


def download_transcription_model(
    whisper_model: str = "base",
    device: str = "cuda",
    compute_type: str | None = None,
    hf_token: str | None = None,
    download_root: str | None = None,
    language: str = "en"
) -> None:
    """
    Pre-download all required models for whisperx pipeline.
    
    Args:
        whisper_model: Whisper model size. Options: tiny, base, small, medium, large, large-v2, large-v3
        device: Device to use (cuda/cpu)
        compute_type: Compute type for inference (float16, int8, float32). Auto-detects based on device if None.
        hf_token: HuggingFace token for pyannote models (required for diarization)
        download_root: Custom directory to save models (optional)
        language: Language code for alignment model (e.g., 'en', 'es', 'fr')
    """
    # auto-detect compute_type if not provided
    if compute_type is None:
        compute_type = "float32" if device == "cpu" else "float16"
    
    print(f"downloading whisper model: {whisper_model}")
    model = whisperx.load_model(
        whisper_model, 
        device, 
        compute_type=compute_type,
        download_root=download_root
    )
    print(f"whisper model downloaded")
    del model
    

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print("all models downloaded and cached")


def download_diarization_model(
    hf_token: str,
    device: str = "cuda",
    download_root: str | None = None
):
    print(f"downloading diarization model")
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token, 
        device=device,
    )
    print(f"diarization model downloaded")
    # Do not delete or free memory before returning the model instance
    return diarize_model


def download_alignment_model(
    language_code: str,
    device: str = "cuda"
):
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code, 
        device=device
    )
    return model_a, metadata


def load_audio(audio_file: str):
    return whisperx.load_audio(audio_file)


def transcribe(
    audio: torch.Tensor, 
    whisper_model: str, 
    download_root: str,
    device: str = "cuda",
    compute_type: str | None = None,
    batch_size: int = 16
):
    # auto-detect compute_type if not provided
    if compute_type is None:
        compute_type = "float32" if device == "cpu" else "float16"
    
    model = whisperx.load_model(
        whisper_model, 
        device, 
        compute_type=compute_type,
        download_root=download_root
    )
    result = model.transcribe(audio, batch_size=batch_size)
    return result, model

def align(result: dict, alignment_model, audio: torch.Tensor, device: str = "cuda"):
    model_a, metadata = alignment_model
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device=device)
    return aligned_result

def diarize(result: dict, diarization_model: DiarizationPipeline, audio: torch.Tensor):
    diarization_segments = diarization_model(audio)
    result_with_speakers = whisperx.assign_word_speakers(diarization_segments, result)
    return result_with_speakers
