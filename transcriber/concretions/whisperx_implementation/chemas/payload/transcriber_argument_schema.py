from pydantic import BaseModel

class WhisperXTranscriberArgumentSchema(BaseModel):
    audio_file: str
    whisper_model: str
    download_root: str
    device: str
    compute_type: str
    batch_size: int