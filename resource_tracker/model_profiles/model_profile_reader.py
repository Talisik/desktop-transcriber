"""reader for WhisperX model profiles configuration"""
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from resource_tracker.base.resource_tracker_base import ResourceTrackerBase


class ModelProfileReader:
    """reads and provides access to WhisperX model profiles from JSON config"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        initialise the model profile reader
        
        Args:
            config_path: path to model_profiles.json file. if None, uses default location
        """
        if config_path is None:
            # default to model_profiles.json in same directory
            config_path = Path(__file__).parent / "model_profiles.json"
        
        self.config_path = Path(config_path)
        self._profiles: Dict[str, Any] = {}
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """load profiles from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"model profiles config not found at {self.config_path}"
            )
        
        with open(self.config_path, 'r') as f:
            self._profiles = json.load(f)
    
    def get_profile(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        get profile for a specific model
        
        Args:
            model_name: name of the model (e.g., 'tiny', 'base', 'small', etc.)
        
        Returns:
            model profile dict or None if not found
        """
        return self._profiles.get(model_name)
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        get all model profiles
        
        Returns:
            dict mapping model names to their profiles
        """
        return self._profiles.copy()
    
    def get_available_models(self) -> list[str]:
        """
        get list of all available model names
        
        Returns:
            list of model names
        """
        return list(self._profiles.keys())
    
    def get_compute_type_options(
        self, 
        model_name: str
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        get compute type options for a specific model
        
        Args:
            model_name: name of the model
        
        Returns:
            dict of compute type options or None if model not found
        """
        profile = self.get_profile(model_name)
        if profile is None:
            return None
        return profile.get("compute_type_options")
    
    def get_batch_size_ranges(
        self, 
        model_name: str
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        get batch size ranges for a specific model
        
        Args:
            model_name: name of the model
        
        Returns:
            dict of batch size ranges per compute type or None if model not found
        """
        profile = self.get_profile(model_name)
        if profile is None:
            return None
        return profile.get("batch_size_ranges")
    
    def get_resource_requirements(
        self, 
        model_name: str
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        get resource requirements for a specific model
        
        Args:
            model_name: name of the model
        
        Returns:
            dict with min_ram_gb, recommended_ram_gb, min_vram_gb, 
            recommended_vram_gb, min_cpu_cores or None if model not found
        """
        profile = self.get_profile(model_name)
        if profile is None:
            return None
        
        return {
            "min_ram_gb": profile.get("min_ram_gb"),
            "recommended_ram_gb": profile.get("recommended_ram_gb"),
            "min_vram_gb": profile.get("min_vram_gb"),
            "recommended_vram_gb": profile.get("recommended_vram_gb"),
            "min_cpu_cores": profile.get("min_cpu_cores")
        }
    
    def get_performance_characteristics(
        self, 
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        get performance characteristics for a specific model
        
        Args:
            model_name: name of the model
        
        Returns:
            dict with performance_tier, speed_tier, model_size_mb or None if model not found
        """
        profile = self.get_profile(model_name)
        if profile is None:
            return None
        
        return {
            "performance_tier": profile.get("performance_tier"),
            "speed_tier": profile.get("speed_tier"),
            "model_size_mb": profile.get("model_size_mb")
        }
    
    def reload(self) -> None:
        """reload profiles from config file"""
        self._load_profiles()
    
    def _model_fits_specs(
        self,
        model_name: str,
        ram_gb: float,
        cpu_cores: int,
        has_gpu: bool,
        vram_gb: Optional[float] = None
    ) -> Tuple[bool, str]:
        # print("Model name: ", model_name)
        profile = self.get_profile(model_name)

        # print("Profile: ", profile)
        # print("RAM: ", ram_gb)
        # print("CPU cores: ", cpu_cores)
        # print("Has GPU: ", has_gpu)
        # print("VRAM: ", vram_gb)

        if profile is None:
            return False, f"model '{model_name}' not found"
        
        # check RAM
        min_ram = profile.get("min_ram_gb", 0)
        if ram_gb < min_ram:
            return False, f"insufficient RAM: {ram_gb:.1f}GB < {min_ram:.1f}GB required"
        
        # check CPU cores
        min_cpu = profile.get("min_cpu_cores", 0)
        if cpu_cores < min_cpu:
            return False, f"insufficient CPU cores: {cpu_cores} < {min_cpu} required"
        
        # check GPU/VRAM requirements
        min_vram = profile.get("min_vram_gb")
        if min_vram is not None:
            if not has_gpu:
                return False, f"model requires GPU but none available"
            if vram_gb is None:
                return False, f"model requires GPU but VRAM info unavailable"
            if vram_gb < min_vram:
                return False, f"insufficient VRAM: {vram_gb:.1f}GB < {min_vram:.1f}GB required"
        
        return True, "model fits system specs"
    
    def get_compatible_models(
        self,
        resource_tracker: Optional[ResourceTrackerBase] = None,
        ram_gb: Optional[float] = None,
        cpu_cores: Optional[int] = None,
        has_gpu: Optional[bool] = None,
        vram_gb: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        # get specs from resource tracker if provided
        if resource_tracker is not None:
            ram_gb = resource_tracker.get_total_ram()
            cpu_cores = resource_tracker.get_total_cpu()
            has_gpu = resource_tracker.has_gpu()
            vram_gb = resource_tracker.get_gpu_vram() if has_gpu else None
        else:
            # validate required params
            if ram_gb is None or cpu_cores is None or has_gpu is None:
                raise ValueError(
                    "must provide resource_tracker or all of ram_gb, cpu_cores, has_gpu"
                )
        
        compatible = []
        for model_name in self.get_available_models():
            fits, reason = self._model_fits_specs(
                model_name, ram_gb, cpu_cores, has_gpu, vram_gb
            )
            if fits:
                profile = self.get_profile(model_name)
                compatible.append({
                    "model_name": model_name,
                    "profile": profile,
                    "fit_reason": reason
                })
        
        return compatible
    
    def get_compatible_models_ranked(
        self,
        resource_tracker: Optional[ResourceTrackerBase] = None,
        ram_gb: Optional[float] = None,
        cpu_cores: Optional[int] = None,
        has_gpu: Optional[bool] = None,
        vram_gb: Optional[float] = None,
        sort_by: str = "performance_tier"
    ) -> List[Dict[str, Any]]:
        """
        get compatible models ranked by performance or speed
        
        Args:
            resource_tracker: ResourceTrackerBase instance (if provided, uses its specs)
            ram_gb: total system RAM in GB (required if resource_tracker not provided)
            cpu_cores: number of CPU cores (required if resource_tracker not provided)
            has_gpu: whether GPU is available (required if resource_tracker not provided)
            vram_gb: total GPU VRAM in GB (optional)
            sort_by: sort by 'performance_tier' (highest first) or 'speed_tier' (highest first)
        
        Returns:
            list of compatible models sorted by specified tier
        """
        compatible = self.get_compatible_models(
            resource_tracker, ram_gb, cpu_cores, has_gpu, vram_gb
        )
        
        if sort_by == "performance_tier":
            compatible.sort(
                key=lambda x: x["profile"].get("performance_tier", 0),
                reverse=True
            )
        elif sort_by == "speed_tier":
            compatible.sort(
                key=lambda x: x["profile"].get("speed_tier", 0),
                reverse=True
            )
        else:
            raise ValueError(f"sort_by must be 'performance_tier' or 'speed_tier', got '{sort_by}'")
        
        return compatible
    
    def get_compatible_models_quality_first(
        self,
        resource_tracker: ResourceTrackerBase
    ) -> List[Dict[str, Any]]:
        """
        get compatible models ranked by quality/performance (highest first)
        
        Args:
            resource_tracker: ResourceTrackerBase instance
        
        Returns:
            list of compatible models sorted by performance tier (highest first)
        """
        return self.get_compatible_models_ranked(
            resource_tracker=resource_tracker,
            sort_by="performance_tier"
        )
    
    def get_compatible_models_speed_first(
        self,
        resource_tracker: ResourceTrackerBase
    ) -> List[Dict[str, Any]]:
        """
        get compatible models ranked by speed (highest first)
        
        Args:
            resource_tracker: ResourceTrackerBase instance
        
        Returns:
            list of compatible models sorted by speed tier (highest first)
        """
        return self.get_compatible_models_ranked(
            resource_tracker=resource_tracker,
            sort_by="speed_tier"
        )
    
    def get_compatible_models_balanced(
        self,
        resource_tracker: ResourceTrackerBase
    ) -> List[Dict[str, Any]]:
        """
        get compatible models ranked by balanced score (average of quality and speed)
        
        Args:
            resource_tracker: ResourceTrackerBase instance
        
        Returns:
            list of compatible models sorted by balanced score (highest first)
        """
        compatible = self.get_compatible_models(resource_tracker=resource_tracker)
        
        compatible.sort(
            key=lambda x: (
                x["profile"].get("performance_tier", 0) + 
                x["profile"].get("speed_tier", 0)
            ) / 2,
            reverse=True
        )
        
        return compatible
    
    def check_model_compatibility(
        self,
        model_name: str,
        resource_tracker: Optional[ResourceTrackerBase] = None,
        ram_gb: Optional[float] = None,
        cpu_cores: Optional[int] = None,
        has_gpu: Optional[bool] = None,
        vram_gb: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        check if a specific model is compatible with system specs
        
        Args:
            model_name: name of the model to check
            resource_tracker: ResourceTrackerBase instance (if provided, uses its specs)
            ram_gb: total system RAM in GB (required if resource_tracker not provided)
            cpu_cores: number of CPU cores (required if resource_tracker not provided)
            has_gpu: whether GPU is available (required if resource_tracker not provided)
            vram_gb: total GPU VRAM in GB (optional)
        
        Returns:
            tuple of (is_compatible: bool, reason: str)
        """
        # get specs from resource tracker if provided
        if resource_tracker is not None:
            ram_gb = resource_tracker.get_total_ram()
            cpu_cores = resource_tracker.get_total_cpu()
            has_gpu = resource_tracker.has_gpu()
            vram_gb = resource_tracker.get_gpu_vram() if has_gpu else None
        else:
            # validate required params
            if ram_gb is None or cpu_cores is None or has_gpu is None:
                raise ValueError(
                    "must provide resource_tracker or all of ram_gb, cpu_cores, has_gpu"
                )
        
        return self._model_fits_specs(model_name, ram_gb, cpu_cores, has_gpu, vram_gb)

