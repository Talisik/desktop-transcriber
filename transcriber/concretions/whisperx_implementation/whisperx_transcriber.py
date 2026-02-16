from transcriber.base.transcriber_base import TranscriberBase
from transcriber.concretions.whisperx_implementation.chemas.custom_types.parameter_types import ComputeType, Device, WhisperModel
from transcriber.concretions.whisperx_implementation.chemas.payload.transcriber_argument_schema import WhisperXTranscriberArgumentSchema
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import gc

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
        
        # auto-detect device if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device

    def __lazy_download_diarization_model(
        self,
        hf_token: str = None,
        device: str | None = None,
        download_root: str | None = None
    ):
        if device is None:
            device = getattr(self, 'device', "cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"downloading diarization model")
        self.diarization_model = DiarizationPipeline(
            use_auth_token=hf_token, 
            device=device,
        )
        print(f"✓ diarization model downloaded")

    def __lazy_download_alignment_model(
        self,
        language_code: str,
        device: str | None = None
    ):
        if device is None:
            device = getattr(self, 'device', "cuda" if torch.cuda.is_available() else "cpu")
        
        # Use "en" as default if language is "tl" or unsupported
        # Alignment model doesn't support Tagalog
        alignment_language = "en" if language_code == "tl" else language_code
        
        self.alignment_model = whisperx.load_align_model(
            language_code=alignment_language, 
            device=device
        )
        print(f"✓ alignment model downloaded (language: {alignment_language})")

    @staticmethod
    def __load_audio(audio_file: str):
        return whisperx.load_audio(audio_file)

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
            device = getattr(self, 'device', "cuda" if torch.cuda.is_available() else "cpu")

        if compute_type is None:
            compute_type = "float32" if device == "cpu" else "float16"
        
        print(f"downloading whisper model: {whisper_model}")
        model = whisperx.load_model(
            whisper_model, 
            device, 
            compute_type=compute_type,
            download_root=download_root
        )
        print(f"✓ whisper model downloaded")
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
        language: str | None = None
    ):
        model = whisperx.load_model(
            whisper_model, 
            device, 
            compute_type=compute_type,
            download_root=download_root
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
            language=payload.language  # Pass language to skip detection
        )

        if self.alignment_model is None:
            # Get detected language from result, or use payload language, or default to "en"
            language_code = payload.language or result.get("language", "en")
            # Map "tl" to "en" for alignment
            alignment_language = "en" if language_code == "tl" else language_code
            device_str = str(payload.device)
            self.__lazy_download_alignment_model(
                language_code=alignment_language,
                device=device_str
            )

        aligned_result = self.__whisperx_align(
            result=result,
            alignment_model=self.alignment_model,
            audio=audio,
            device=Device(payload.device)
        )

        return aligned_result, model