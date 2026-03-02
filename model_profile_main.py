#!/usr/bin/env python3
"""
Model Profile - Standalone executable for determining compatible WhisperX models.
Accepts JSON input from resource_tracker.exe or detects system specs directly.

Usage:
    # Pipeline mode (from resource_tracker output):
    resource_tracker.exe | model_profile.exe
    # OR
    resource_tracker.exe > resources.json && model_profile.exe --input resources.json
    
    # Standalone mode (detects resources directly):
    model_profile.exe
"""
import argparse
import json
import sys

from pathlib import Path
from typing import Optional, Dict, Any
from resource_tracker.concretions.desktop_resource_tracker import DesktopResourceTracker
from resource_tracker.model_profiles.model_profile_reader import ModelProfileReader


def parse_resource_tracker_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    parse resource tracker JSON output into format needed for model profile reader
    
    Args:
        json_data: JSON dict from resource_tracker.exe output
        
    Returns:
        dict with ram_gb, ram_mb, cpu_cores, has_gpu, vram_gb, vram_mb, etc.
    """
    ram_data = json_data.get("ram", {})
    cpu_data = json_data.get("cpu", {})
    gpu_data = json_data.get("gpu", {})
    
    # extract RAM values - use total instead of available
    ram_total_mb = ram_data.get("total_mb", 0)
    ram_available_mb = ram_data.get("available_mb", ram_total_mb)
    ram_used_mb = ram_data.get("used_mb", 0)
    ram_total_gb = ram_total_mb / 1024
    ram_available_gb = ram_available_mb / 1024
    
    # extract CPU
    cpu_cores = cpu_data.get("cores", 0)
    
    # extract GPU/VRAM - use total instead of available
    has_gpu = gpu_data.get("available", False)
    vram_total_mb = gpu_data.get("vram_total_mb")
    vram_used_mb = gpu_data.get("vram_used_mb", 0) if has_gpu else None
    
    # use total VRAM (not available)
    vram_total_gb = None
    if has_gpu and vram_total_mb is not None:
        vram_total_gb = vram_total_mb / 1024
    
    return {
        "ram_available_mb": ram_available_mb,
        "ram_total_mb": ram_total_mb,
        "ram_used_mb": ram_used_mb,
        "ram_available_gb": ram_available_gb,
        "ram_total_gb": ram_total_gb,
        "ram_usage_percent": ram_data.get("usage_percent", 0),
        "cpu_cores": cpu_cores,
        "cpu_usage_percent": cpu_data.get("usage_percent", 0),
        "has_gpu": has_gpu,
        "vram_total_mb": vram_total_mb,
        "vram_used_mb": vram_used_mb,
        "vram_available_mb": None,  # not used anymore
        "vram_available_gb": None,  # not used anymore
        "vram_total_gb": vram_total_gb
    }


def main():
    """main entry point for model profile checker"""
    parser = argparse.ArgumentParser(
        description="determine compatible WhisperX models based on system resources"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="path to JSON file from resource_tracker.exe (or use stdin)"
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.2,
        help="safety margin multiplier for resource requirements (default: 1.2 = 20%% buffer)"
    )
    
    args = parser.parse_args()
    
    try:
        reader = ModelProfileReader()
        resource_data = None
        tracker = None
        
        # try to read from input file or stdin
        if args.input:
            # resolve path - handle both relative and absolute paths
            input_path = Path(args.input)
            if not input_path.is_absolute():
                # if relative, resolve from current working directory
                input_path = Path.cwd() / input_path
            
            if not input_path.exists():
                error_result = {
                    "error": f"input file not found: {args.input}",
                    "resolved_path": str(input_path),
                    "type": "FileNotFoundError"
                }
                print(json.dumps(error_result, indent=2), file=sys.stderr)
                return 1
            
            try:
                with open(input_path, 'r') as f:
                    json_data = json.load(f)
                    resource_data = parse_resource_tracker_json(json_data)
            except json.JSONDecodeError as e:
                error_result = {
                    "error": f"invalid JSON in input file: {str(e)}",
                    "file": str(input_path),
                    "type": "JSONDecodeError"
                }
                print(json.dumps(error_result, indent=2), file=sys.stderr)
                return 1
        elif not sys.stdin.isatty():
            # stdin might have data (piped input) - try to read it
            try:
                json_data = json.load(sys.stdin)
                resource_data = parse_resource_tracker_json(json_data)
            except (json.JSONDecodeError, ValueError):
                # stdin is empty or invalid, use standalone mode
                tracker = DesktopResourceTracker()
                resource_data = {
                    "ram_available_mb": tracker.get_available_ram_mb(),
                    "ram_total_mb": tracker.get_total_ram_mb(),
                    "ram_used_mb": tracker.get_used_ram_mb(),
                    "ram_available_gb": tracker.get_available_ram_mb() / 1024,
                    "ram_total_gb": tracker.get_total_ram_mb() / 1024,
                    "ram_usage_percent": tracker.get_ram_usage_percent(),
                    "cpu_cores": tracker.get_total_cpu(),
                    "cpu_usage_percent": tracker.get_cpu_usage_percent(),
                    "has_gpu": tracker.has_gpu(),
                    "vram_total_mb": tracker.get_gpu_vram_mb() if tracker.has_gpu() else None,
                    "vram_used_mb": tracker.get_used_gpu_vram_mb() if tracker.has_gpu() else None,
                    "vram_available_mb": None,
                    "vram_available_gb": None,
                    "vram_total_gb": tracker.get_gpu_vram() if tracker.has_gpu() else None
                }
        else:
            # no input provided, use standalone mode
            tracker = DesktopResourceTracker()
            resource_data = {
                "ram_available_mb": tracker.get_available_ram_mb(),
                "ram_total_mb": tracker.get_total_ram_mb(),
                "ram_used_mb": tracker.get_used_ram_mb(),
                "ram_available_gb": tracker.get_available_ram_mb() / 1024,
                "ram_total_gb": tracker.get_total_ram_mb() / 1024,
                "ram_usage_percent": tracker.get_ram_usage_percent(),
                "cpu_cores": tracker.get_total_cpu(),
                "cpu_usage_percent": tracker.get_cpu_usage_percent(),
                "has_gpu": tracker.has_gpu(),
                "vram_total_mb": tracker.get_gpu_vram_mb() if tracker.has_gpu() else None,
                "vram_used_mb": tracker.get_used_gpu_vram_mb() if tracker.has_gpu() else None,
                "vram_available_mb": None,
                "vram_available_gb": None,
                "vram_total_gb": tracker.get_gpu_vram() if tracker.has_gpu() else None
            }
        
        # get compatible models using total resources (not available)
        if tracker:
            # use tracker object (standalone mode)
            compatible_quality = reader.get_compatible_models_quality_first(
                tracker, use_available_resources=False, safety_margin=args.safety_margin
            )
            compatible_speed = reader.get_compatible_models_speed_first(
                tracker, use_available_resources=False, safety_margin=args.safety_margin
            )
            compatible_balanced = reader.get_compatible_models_balanced(
                tracker, use_available_resources=False, safety_margin=args.safety_margin
            )
            all_compatible = reader.get_compatible_models(
                tracker, use_available_resources=False, safety_margin=args.safety_margin
            )
        else:
            # use direct parameters (pipeline mode) - use total resources
            all_compatible = reader.get_compatible_models(
                ram_gb=resource_data["ram_total_gb"],
                cpu_cores=resource_data["cpu_cores"],
                has_gpu=resource_data["has_gpu"],
                vram_gb=resource_data["vram_total_gb"],
                use_available_resources=False,
                safety_margin=args.safety_margin
            )
            
            # sort for quality, speed, and balanced
            compatible_quality = sorted(
                all_compatible,
                key=lambda x: x["profile"].get("performance_tier", 0),
                reverse=True
            )
            compatible_speed = sorted(
                all_compatible,
                key=lambda x: x["profile"].get("speed_tier", 0),
                reverse=True
            )
            compatible_balanced = sorted(
                all_compatible,
                key=lambda x: (
                    x["profile"].get("performance_tier", 0) + 
                    x["profile"].get("speed_tier", 0)
                ) / 2,
                reverse=True
            )
        
        # build results
        results = {
            "system_resources": {
                "ram": {
                    "total_mb": round(resource_data["ram_total_mb"], 2),
                    "available_mb": round(resource_data["ram_available_mb"], 2),
                    "used_mb": round(resource_data["ram_used_mb"], 2),
                    "usage_percent": round(resource_data["ram_usage_percent"], 2)
                },
                "cpu": {
                    "cores": resource_data["cpu_cores"],
                    "usage_percent": round(resource_data["cpu_usage_percent"], 2)
                },
                "gpu": {
                    "available": resource_data["has_gpu"],
                    "vram_total_mb": round(resource_data["vram_total_mb"], 2) if resource_data["vram_total_mb"] else None,
                    "vram_available_mb": round(resource_data["vram_available_mb"], 2) if resource_data["vram_available_mb"] else None,
                    "vram_used_mb": round(resource_data["vram_used_mb"], 2) if resource_data["vram_used_mb"] else None
                }
            },
            "compatible_models": {
                "quality_first": [
                    {
                        "model_name": m["model_name"],
                        "performance_tier": m["profile"].get("performance_tier"),
                        "speed_tier": m["profile"].get("speed_tier"),
                        "model_size_mb": m["profile"].get("model_size_mb")
                    }
                    for m in compatible_quality
                ],
                "speed_first": [
                    {
                        "model_name": m["model_name"],
                        "performance_tier": m["profile"].get("performance_tier"),
                        "speed_tier": m["profile"].get("speed_tier"),
                        "model_size_mb": m["profile"].get("model_size_mb")
                    }
                    for m in compatible_speed
                ],
                "balanced": [
                    {
                        "model_name": m["model_name"],
                        "performance_tier": m["profile"].get("performance_tier"),
                        "speed_tier": m["profile"].get("speed_tier"),
                        "model_size_mb": m["profile"].get("model_size_mb"),
                        "balanced_score": round(
                            (m["profile"].get("performance_tier", 0) + m["profile"].get("speed_tier", 0)) / 2,
                            2
                        )
                    }
                    for m in compatible_balanced
                ],
                "all": [
                    {
                        "model_name": m["model_name"],
                        "performance_tier": m["profile"].get("performance_tier"),
                        "speed_tier": m["profile"].get("speed_tier"),
                        "model_size_mb": m["profile"].get("model_size_mb")
                    }
                    for m in all_compatible
                ]
            },
            "recommendations": {
                "best_quality": compatible_quality[0]["model_name"] if compatible_quality else None,
                "fastest": compatible_speed[0]["model_name"] if compatible_speed else None,
                "best_balanced": compatible_balanced[0]["model_name"] if compatible_balanced else None
            }
        }
        
        # output as JSON
        print(json.dumps(results, indent=2))
        return 0
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "type": type(e).__name__
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

