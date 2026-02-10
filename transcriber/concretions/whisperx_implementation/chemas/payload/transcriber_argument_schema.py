from pydantic import BaseModel

class WhisperXTranscriberArgumentSchema(BaseModel):
    audio_file: str
    whisper_model: str
    download_root: str
    device: str
    compute_type: str
    batch_size: int
    language: str | None = None  # language code to skip detection