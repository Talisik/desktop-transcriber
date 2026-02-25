from resource_tracker.base.resource_tracker_base import ResourceTrackerBase
from typing import Optional
import psutil
import subprocess
import re


class DesktopResourceTracker(ResourceTrackerBase):
    """concrete implementation of resource tracker for desktop systems"""
    
    def __init__(self):
        self._gpu_available: Optional[bool] = None
        self._gpu_vram: Optional[float] = None
        self._gpu_vram_used: Optional[float] = None
    
    def get_total_ram(self) -> float:
        """get total system ram in GB"""
        return psutil.virtual_memory().total / (1024 ** 3)
    
    def get_total_ram_mb(self) -> float:
        """get total system ram in MB"""
        return psutil.virtual_memory().total / (1024 ** 2)
    
    def get_used_ram_mb(self) -> float:
        """get currently used ram in MB"""
        return psutil.virtual_memory().used / (1024 ** 2)
    
    def get_available_ram_mb(self) -> float:
        """get currently available ram in MB"""
        return psutil.virtual_memory().available / (1024 ** 2)
    
    def get_ram_usage_percent(self) -> float:
        """get ram usage percentage"""
        return psutil.virtual_memory().percent

    def get_total_cpu(self) -> int:
        """get total number of cpu cores"""
        return psutil.cpu_count(logical=True)
    
    def get_cpu_usage_percent(self) -> float:
        """get current cpu usage percentage"""
        return psutil.cpu_percent(interval=0.1)

    def has_gpu(self) -> bool:
        """check if system has a gpu available (with fallback to nvidia-smi)"""
        if self._gpu_available is not None:
            return self._gpu_available
        
        # try torch first
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
                return True
        except ImportError:
            pass
        
        # fallback to nvidia-smi if torch fails
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                self._gpu_available = True
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        self._gpu_available = False
        return False
    
    def get_gpu_vram(self) -> Optional[float]:
        """get total gpu vram in GB (with fallback to nvidia-smi)"""
        if self._gpu_vram is not None:
            return self._gpu_vram
        
        if not self.has_gpu():
            return None
        
        # try torch first
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    vram_bytes = torch.cuda.get_device_properties(0).total_memory
                    self._gpu_vram = vram_bytes / (1024 ** 3)
                    return self._gpu_vram
                except Exception:
                    pass
        except ImportError:
            pass
        
        # fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'(\d+)', result.stdout.strip())
                if match:
                    vram_mb = int(match.group(1))
                    self._gpu_vram = vram_mb / 1024  # convert MB to GB
                    return self._gpu_vram
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return None
    
    def get_gpu_vram_mb(self) -> Optional[float]:
        """get total gpu vram in MB"""
        vram_gb = self.get_gpu_vram()
        return vram_gb * 1024 if vram_gb else None
    
    def get_used_gpu_vram_mb(self) -> Optional[float]:
        """get currently used gpu vram in MB"""
        if not self.has_gpu():
            return None
        
        # try torch first
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    vram_used_bytes = torch.cuda.memory_allocated(0)
                    return vram_used_bytes / (1024 ** 2)  # convert to MB
                except Exception:
                    pass
        except ImportError:
            pass
        
        # fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'(\d+)', result.stdout.strip())
                if match:
                    return float(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return None
    
    def track_resource(self, resource: str):
        """track a specific resource (placeholder implementation)"""
        # this could be extended to log or monitor specific resources
        pass  