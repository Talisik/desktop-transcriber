"""WhisperX model manager implementation"""
import gc
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import whisperx

# patch torch.load for PyTorch 2.6+ weights_only issue
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)
torch.load = _patched_torch_load

from model_manager.base.model_manager_base import ModelManagerBase


class WhisperXModelManager(ModelManagerBase):
    """manages WhisperX model downloads and caching"""
    
    def __init__(self, default_device: str = "cpu", default_compute_type: Optional[str] = None):
        """
        initialise WhisperX model manager
        
        Args:
            default_device: default device to use (cuda/cpu)
            default_compute_type: default compute type (float16, int8, float32). auto-detects if None
        """
        self.default_device = default_device
        self.default_compute_type = default_compute_type
    
    def download_model(
        self,
        model_name: str,
        download_root: str,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        download a WhisperX model to specified directory
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)
            download_root: directory to save the model
            device: device to use (cuda/cpu). uses default if None
            compute_type: compute type (float16, int8, float32). auto-detects if None
            
        Returns:
            dict with download status and model info
        """
        if device is None:
            device = self.default_device
        
        if compute_type is None:
            if self.default_compute_type:
                compute_type = self.default_compute_type
            else:
                # auto-detect based on device
                compute_type = "float32" if device == "cpu" else "float16"
        
        download_path = Path(download_root)
        download_path.mkdir(parents=True, exist_ok=True)
        
        result = {
            "model_name": model_name,
            "download_root": str(download_path),
            "device": device,
            "compute_type": compute_type,
            "status": "downloading",
            "error": None
        }
        
        try:
            # download the model
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(download_path)
            )
            
            # get model info if available
            model_size_mb = None
            if hasattr(model, "model"):
                # faster_whisper model
                if hasattr(model.model, "num_parameters"):
                    # rough estimate: 4 bytes per parameter (float32)
                    num_params = model.model.num_parameters()
                    model_size_mb = (num_params * 4) / (1024 ** 2)
            
            # cleanup
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            
            result.update({
                "status": "success",
                "model_size_mb": round(model_size_mb, 2) if model_size_mb else None
            })
            
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
        
        return result
    
    def check_model_exists(
        self,
        model_name: str,
        download_root: str
    ) -> bool:
        """
        check if a model already exists in the download directory
        
        Args:
            model_name: Whisper model name
            download_root: directory to check
            
        Returns:
            True if model exists, False otherwise
        """
        download_path = Path(download_root)
        if not download_path.exists():
            return False
        
        # faster_whisper models are typically stored in a subdirectory
        # check for common model cache locations
        possible_paths = [
            download_path / model_name,
            download_path / f"{model_name}.pt",
            download_path / "models" / model_name,
        ]
        
        return any(p.exists() for p in possible_paths)
    
    def load_model(self, model_name: str):
        """placeholder - not implemented for download-only manager"""
        raise NotImplementedError("load_model not implemented - use download_model instead")

