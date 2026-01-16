from abc import ABC, abstractmethod
from typing import Optional


class ResourceTrackerBase(ABC):
    """abstract base class for tracking system resources: ram, cpu, and gpu/vram"""
    
    @abstractmethod
    def get_total_ram(self) -> float:
        pass
    
    @abstractmethod
    def get_total_cpu(self) -> int:
        pass
    
    @abstractmethod
    def has_gpu(self) -> bool:
        pass
    
    @abstractmethod
    def get_gpu_vram(self) -> Optional[float]:

        pass
    
    @abstractmethod
    def track_resource(self, resource: str):
        pass

