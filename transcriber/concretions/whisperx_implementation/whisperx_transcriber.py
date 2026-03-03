from transcriber.base.transcriber_base import TranscriberBase
from transcriber.concretions.whisperx_implementation.chemas.custom_types.parameter_types import ComputeType, Device, WhisperModel
from transcriber.concretions.whisperx_implementation.chemas.payload.transcriber_argument_schema import WhisperXTranscriberArgumentSchema
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import gc
import pandas as pd

# Valid Whisper language codes
VALID_WHISPER_LANGUAGES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy",
    "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn",
    "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si",
    "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"
}


def validate_language_code(language: str | None) -> str | None:
    """
    Validate language code and map unsupported codes to 'en'.
    
    Args:
        language: Language code to validate
    
    Returns:
        Valid language code, or 'en' if unsupported, or None if language is None
    """
    if language is None:
        return None
    
    language_lower = language.lower()
    if language_lower in VALID_WHISPER_LANGUAGES:
        return language_lower
    
    # Map unsupported language codes to English
    print(f"WARNING: language code '{language}' not supported, using 'en' instead")
    return "en"


class WhisperXTranscriber(TranscriberBase):

    def __init__(self, device: str | None = None):
        self.hf_token = None
        self.alignment_model = None
        self.diarization_model = None
        
        # Default to CPU if device not provided (device should come from payload)
        if device is None:
            device = "cpu"
        
        # Validate device parameter
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")
        
        self.device = device
        print(f"   using device: {device}")

    def __lazy_download_diarization_model(
        self,
        hf_token: str = None,
        device: str | None = None,
        download_root: str | None = None
    ):
        if device is None:
            device = getattr(self, 'device', "cpu")
        
        # Check if model is already loaded
        if self.diarization_model is not None:
            return
        
        print(f"loading diarization model")
        # Handle both newer (token) and older (use_auth_token) whisperx versions
        try:
            # Try token first (newer versions)
            self.diarization_model = DiarizationPipeline(
                token=hf_token, 
                device=device,
            )
        except TypeError:
            # Fall back to use_auth_token for older versions
            self.diarization_model = DiarizationPipeline(
                use_auth_token=hf_token, 
                device=device,
            )
        print(f"diarization model loaded")

    def __lazy_download_alignment_model(
        self,
        language_code: str,
        device: str | None = None
    ):
        if device is None:
            device = getattr(self, 'device', "cpu")
        
        # Try to load alignment model for requested language
        # Fall back to English if model not available for any reason
        alignment_language = language_code
        
        try:
            self.alignment_model = whisperx.load_align_model(
                language_code=alignment_language, 
                device=device
            )
            print(f"alignment model downloaded (language: {alignment_language})")
        except Exception as e:
            # If model not available for this language, fall back to English
            if alignment_language != "en":
                print(f"WARNING: Alignment model for language '{language_code}' not available, using English alignment model")
                alignment_language = "en"
                try:
                    self.alignment_model = whisperx.load_align_model(
                        language_code=alignment_language, 
                        device=device
                    )
                    print(f"alignment model downloaded (language: {alignment_language})")
                except Exception as fallback_error:
                    # If even English fails, that's a real problem
                    raise RuntimeError(f"Failed to load English alignment model (fallback): {fallback_error}") from fallback_error
            else:
                # Already trying English, so this is a real error
                raise RuntimeError(f"Failed to load English alignment model: {e}") from e

    @staticmethod
    def __load_audio(audio_file: str):
        """
        Load audio file and validate format.
        Ensures audio is in correct shape for WhisperX processing.
        """
        import numpy as np
        import os
        
        # Check if file exists and has content
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {audio_file}")
        
        # Load audio using whisperx
        audio = whisperx.load_audio(audio_file)
        
        # Validate and reshape audio
        if audio is None:
            raise ValueError(f"Failed to load audio from: {audio_file}")
        
        # Ensure audio is numpy array
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Audio must be numpy array, got {type(audio)}")
        
        # Ensure audio is 1D (whisperx expects mono)
        if len(audio.shape) > 1:
            print(f"WARNING: Audio has multiple channels, converting to mono")
            audio = audio.mean(axis=0) 
        
        # Validate audio has samples
        if len(audio) == 0:
            raise ValueError(f"Audio has no samples: {audio_file}")
        
        return audio

    def __lazy_download_transcription_model(
        self,
        whisper_model: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
        hf_token: str | None = None,
        download_root: str | None = None,
        language: str = "en"
    ):
        """
        Pre-download all required models for whisperx pipeline.
        
        Args:
            whisper_model: Whisper model size. Options: tiny, base, small, medium, large, large-v2, large-v3
            device: Device to use (cuda/cpu). Auto-detects if None.
            compute_type: Compute type for inference (float16, int8, float32). Auto-detects based on device if None.
            hf_token: HuggingFace token for pyannote models (required for diarization)
            download_root: Custom directory to save models (optional)
            language: Language code for alignment model (e.g., 'en', 'es', 'fr')
        """
        
        if device is None:
            device = getattr(self, 'device', "cpu")

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


    def __whisperx_transcribe(
        self,
        audio: torch.Tensor, 
        whisper_model: WhisperModel, 
        download_root: str,
        device: Device = Device.cuda,
        compute_type: ComputeType = ComputeType.float16,
        batch_size: int = 16,
        language: str | None = None,
        model_path: str | None = None
    ):
        # Determine which download_root to use
        # model_path is a parent directory containing all models (like download_root)
        # Try model_path first if provided, then fall back to download_root
        effective_download_root = download_root
        
        if model_path:
            from pathlib import Path
            
            model_path_obj = Path(model_path)
            if model_path_obj.exists() and model_path_obj.is_dir():
                # Use model_path as the download root (parent directory containing all models)
                effective_download_root = str(model_path_obj)
            elif model_path_obj.exists() and model_path_obj.is_file():
                # If a file is provided, use its parent directory
                effective_download_root = str(model_path_obj.parent)
        
        # Try to load model with effective_download_root
        # If it fails, whisperx will download it automatically
        model = whisperx.load_model(
            whisper_model, 
            device, 
            compute_type=compute_type,
            download_root=effective_download_root
        )
        # Pass language to skip detection
        # Note: vad_filter is not a valid parameter for model.transcribe()
        # VAD is handled at WhisperX wrapper level, but since we're using
        # pre-segmented audio, VAD overhead should be minimal
        transcribe_kwargs = {"batch_size": batch_size}
        if language:
            # Validate and map unsupported language codes to 'en'
            validated_language = validate_language_code(language)
            if validated_language:
                transcribe_kwargs["language"] = validated_language
        result = model.transcribe(audio, **transcribe_kwargs)
        return result, model

    def __whisperx_align(
        self,
        result: dict,
        alignment_model: tuple,
        audio: torch.Tensor,
        device: Device = Device.cuda
    ):
        model_a, metadata = alignment_model
        aligned_result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata,
            audio, 
            device=device
        )
        return aligned_result

    def __whisperx_diarize(
        self,
        result: dict,
        diarization_model: tuple,
        audio: torch.Tensor
    ):
        diarization_segments = diarization_model(audio)
        result_with_speakers = whisperx.assign_word_speakers(diarization_segments, result)
        return result_with_speakers

    def transcribe(self, payload: WhisperXTranscriberArgumentSchema):
        audio = self.__load_audio(payload.audio_file)

        result, model = self.__whisperx_transcribe(
            audio=audio,
            whisper_model=WhisperModel(payload.whisper_model),
            download_root=payload.download_root,
            device=Device(payload.device),
            compute_type=ComputeType(payload.compute_type),
            batch_size=payload.batch_size,
            language=payload.language,  # Pass language to skip detection
            model_path=payload.model_path  # Pass model_path as fallback
        )

        if self.alignment_model is None:
            # Get detected language from result, or use payload language, or default to "en"
            language_code = payload.language or result.get("language", "en")
            device_str = str(payload.device)
            # Let the method handle fallback to English for unsupported languages
            self.__lazy_download_alignment_model(
                language_code=language_code,
                device=device_str
            )

        aligned_result = self.__whisperx_align(
            result=result,
            alignment_model=self.alignment_model,
            audio=audio,
            device=Device(payload.device)
        )

        return aligned_result, model

    def diarize_audio(
        self,
        audio_file: str,
        hf_token: str | None = None,
        device: str | None = None
    ) -> list:
        """
        Run diarization pipeline on full audio file.
        Returns speaker segments with timestamps (no transcription text).
        
        Args:
            audio_file: Path to audio file
            hf_token: HuggingFace token for pyannote models
            device: Device to use (cuda/cpu)
        
        Returns:
            List of speaker segments: [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.2},
                {"speaker": "SPEAKER_01", "start": 5.2, "end": 12.8},
                ...
            ]
        """
        import os
        
        if device is None:
            device = getattr(self, 'device', "cpu")
        
        # Load audio
        print(f"   loading audio for diarization: {audio_file}")
        audio = whisperx.load_audio(audio_file)
        
        # Load diarization model if not already loaded in this instance
        # Note: Model files are cached by HuggingFace, so this just loads from cache
        if self.diarization_model is None:
            print(f"   loading diarization model on {device}")
            self.__lazy_download_diarization_model(
                hf_token=hf_token,
                device=device
            )
        
        # Run diarization
        print(f"   running diarization pipeline...")
        diarization_result = self.diarization_model(audio)
        
        # Extract speaker segments
        speaker_segments = []
        
        # Check if result is a DataFrame (whisperx returns DataFrame)
        if isinstance(diarization_result, pd.DataFrame):
            # DataFrame format: iterate through rows
            for _, row in diarization_result.iterrows():
                speaker_segments.append({
                    "speaker": row.get("speaker", row.get("label", "SPEAKER_00")),
                    "start": float(row.get("start", 0.0)),
                    "end": float(row.get("end", 0.0))
                })
        else:
            # Fallback: try pyannote Annotation format (itertracks)
            try:
                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    speaker_segments.append({
                        "speaker": speaker,
                        "start": turn.start,
                        "end": turn.end
                    })
            except AttributeError:
                # If neither format works, raise informative error
                raise ValueError(
                    f"Unexpected diarization result type: {type(diarization_result)}. "
                    f"Expected pandas DataFrame or pyannote Annotation."
                )
        
        print(f"   detected {len(set(s['speaker'] for s in speaker_segments))} unique speakers")
        print(f"   extracted {len(speaker_segments)} speaker segments")
        
        return speaker_segments