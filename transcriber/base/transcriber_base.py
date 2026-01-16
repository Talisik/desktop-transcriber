from abc import ABC, abstractmethod

class TranscriberBase(ABC):

    @abstractmethod
    def transcribe(self, audio_file: str):
        pass
