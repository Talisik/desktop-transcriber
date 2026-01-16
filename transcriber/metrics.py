import threading
import time
import psutil
import torch
from typing import Dict, Optional
from pathlib import Path
import subprocess
import json


class ResourceMonitor:
    """monitor cpu, ram, and gpu memory usage during processing"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # metrics storage
        self.cpu_samples: list[float] = []
        self.ram_samples: list[float] = []  # in GB
        self.gpu_samples: list[float] = []  # in GB
        
        # process reference for tracking
        self.process = psutil.Process()
        self.initial_ram = self.process.memory_info().rss / (1024 ** 3)  # GB
        
        # gpu availability
        self.gpu_available = torch.cuda.is_available() and device == "cuda"
    
    def _monitor_loop(self):
        """background thread monitoring loop"""
        while self.monitoring:
            # cpu usage (percentage)
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
            
            # ram usage (GB)
            ram_gb = self.process.memory_info().rss / (1024 ** 3)
            self.ram_samples.append(ram_gb)
            
            # gpu memory (GB) if available
            if self.gpu_available:
                gpu_bytes = torch.cuda.memory_allocated()
                gpu_gb = gpu_bytes / (1024 ** 3)
                self.gpu_samples.append(gpu_gb)
            else:
                self.gpu_samples.append(0.0)
            
            time.sleep(0.5)  # sample every 500ms
    
    def start(self):
        """start monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self.gpu_samples.clear()
        
        # reset gpu peak memory tracking
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """get peak, average, and final metrics"""
        if not self.cpu_samples:
            # return zeros if no samples
            return {
                "memory": {"peak": 0.0, "average": 0.0, "final": 0.0},
                "cpu": {"peak": 0.0, "average": 0.0, "final": 0.0},
                "gpu": {"peak": 0.0, "average": 0.0, "final": 0.0}
            }
        
        # cpu metrics
        cpu_peak = max(self.cpu_samples) if self.cpu_samples else 0.0
        cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        cpu_final = self.cpu_samples[-1] if self.cpu_samples else 0.0
        
        # ram metrics
        ram_peak = max(self.ram_samples) if self.ram_samples else 0.0
        ram_avg = sum(self.ram_samples) / len(self.ram_samples) if self.ram_samples else 0.0
        ram_final = self.ram_samples[-1] if self.ram_samples else 0.0
        
        # gpu metrics
        if self.gpu_available:
            gpu_peak_bytes = torch.cuda.max_memory_allocated()
            gpu_peak = gpu_peak_bytes / (1024 ** 3)
            gpu_avg = sum(self.gpu_samples) / len(self.gpu_samples) if self.gpu_samples else 0.0
            gpu_final = self.gpu_samples[-1] if self.gpu_samples else 0.0
        else:
            gpu_peak = 0.0
            gpu_avg = 0.0
            gpu_final = 0.0
        
        return {
            "memory": {
                "peak": round(ram_peak, 2),
                "average": round(ram_avg, 2),
                "final": round(ram_final, 2)
            },
            "cpu": {
                "peak": round(cpu_peak, 2),
                "average": round(cpu_avg, 2),
                "final": round(cpu_final, 2)
            },
            "gpu": {
                "peak": round(gpu_peak, 2),
                "average": round(gpu_avg, 2),
                "final": round(gpu_final, 2)
            }
        }


def get_file_duration(file_path: str) -> float:
    """get duration of audio/video file in seconds using ffprobe"""
    try:
        # try ffprobe first (most reliable)
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        return duration
    except (subprocess.CalledProcessError, KeyError, ValueError, FileNotFoundError):
        # fallback: try using mutagen for audio files
        try:
            from mutagen import File as MutagenFile
            audio_file = MutagenFile(file_path)
            if audio_file is not None and hasattr(audio_file, "info"):
                return audio_file.info.length
        except (ImportError, AttributeError):
            pass
        
        # if all else fails, return 0.0
        return 0.0

