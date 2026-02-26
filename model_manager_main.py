#!/usr/bin/env python3
"""
Model Manager - Standalone executable for downloading WhisperX models.
Accepts JSON input from model_profile.exe to determine compatible models.

Usage:
    # Pipeline mode (from model_profile output):
    model_profile.exe | model_manager.exe --model tiny --output-dir ./models
    
    # File input mode:
    model_profile.exe > profile.json
    model_manager.exe --input profile.json --model small --output-dir ./models
    
    # Direct model download (no profile input):
    model_manager.exe --model base --output-dir ./models
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from model_manager.concretions.whisperx_model_manager import WhisperXModelManager


def parse_model_profile_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    parse model profile JSON output
    
    Args:
        json_data: JSON dict from model_profile.exe output
        
    Returns:
        dict with compatible models and recommendations
    """
    return {
        "compatible_models": json_data.get("compatible_models", {}),
        "recommendations": json_data.get("recommendations", {}),
        "system_resources": json_data.get("system_resources", {})
    }


def validate_model_name(model_name: str, profile_data: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
    """
    validate model name against profile recommendations if available
    
    Args:
        model_name: model name to validate
        profile_data: parsed profile data (optional)
        
    Returns:
        tuple of (is_valid: bool, warning_message: Optional[str])
    """
    if profile_data is None:
        return True, None
    
    all_compatible = profile_data.get("compatible_models", {}).get("all", [])
    compatible_names = [m.get("model_name") for m in all_compatible]
    
    if compatible_names and model_name not in compatible_names:
        recommendations = profile_data.get("recommendations", {})
        best_quality = recommendations.get("best_quality")
        fastest = recommendations.get("fastest")
        
        suggestions = []
        if best_quality:
            suggestions.append(f"best_quality: {best_quality}")
        if fastest:
            suggestions.append(f"fastest: {fastest}")
        
        warning = f"model '{model_name}' not in compatible models list"
        if suggestions:
            warning += f". suggestions: {', '.join(suggestions)}"
        
        return False, warning
    
    return True, None


def main():
    """main entry point for model manager"""
    parser = argparse.ArgumentParser(
        description="download WhisperX models to specified directory"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="WhisperX model name (tiny, base, small, medium, large, large-v2, large-v3)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="directory to download the model to"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="path to JSON file from model_profile.exe (optional, for validation)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="device to use for model download (default: cpu)"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["float16", "int8", "float32"],
        help="compute type for model (auto-detects based on device if not specified)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip validation against profile recommendations"
    )
    
    args = parser.parse_args()
    
    try:
        profile_data = None
        
        # try to read profile input if provided
        if args.input:
            with open(args.input, 'r') as f:
                json_data = json.load(f)
                profile_data = parse_model_profile_json(json_data)
        elif not sys.stdin.isatty():
            # stdin has data (piped input)
            json_data = json.load(sys.stdin)
            profile_data = parse_model_profile_json(json_data)
        
        # validate model name if profile data available
        if profile_data and not args.skip_validation:
            is_valid, warning = validate_model_name(args.model, profile_data)
            if not is_valid and warning:
                result = {
                    "status": "validation_failed",
                    "error": warning,
                    "model_name": args.model
                }
                print(json.dumps(result, indent=2))
                return 1
        
        # initialise model manager
        manager = WhisperXModelManager(
            default_device=args.device,
            default_compute_type=args.compute_type
        )
        
        # check if model already exists
        model_exists = manager.check_model_exists(args.model, args.output_dir)
        
        if model_exists:
            result = {
                "status": "already_exists",
                "model_name": args.model,
                "download_root": str(Path(args.output_dir).absolute()),
                "message": "model already exists in download directory"
            }
        else:
            # download the model
            result = manager.download_model(
                model_name=args.model,
                download_root=args.output_dir,
                device=args.device,
                compute_type=args.compute_type
            )
        
        # add profile context if available
        if profile_data:
            result["profile_context"] = {
                "recommended_quality": profile_data.get("recommendations", {}).get("best_quality"),
                "recommended_speed": profile_data.get("recommendations", {}).get("fastest"),
                "recommended_balanced": profile_data.get("recommendations", {}).get("best_balanced")
            }
        
        # output result
        print(json.dumps(result, indent=2))
        
        # return error code if download failed
        if result.get("status") == "error":
            return 1
        
        return 0
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())




