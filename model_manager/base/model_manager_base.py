from abc import ABC, abstractmethod


class ModelManagerBase(ABC):
    def __init__(self):
        pass

    def download_model(self, model_name: str):
        pass

    def load_model(self, model_name: str):
        pass