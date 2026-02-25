#!/usr/bin/env python3
"""
Resource Tracker - Standalone executable for detecting system resources.
Outputs JSON with RAM, CPU, and GPU information.
"""
import json
import sys
from resource_tracker.concretions.desktop_resource_tracker import DesktopResourceTracker


def main():
    """main entry point for resource tracker"""
    try:
        tracker = DesktopResourceTracker()
        
        # collect all resource data
        results = {
            "ram": {
                "total_mb": round(tracker.get_total_ram_mb(), 2),
                "used_mb": round(tracker.get_used_ram_mb(), 2),
                "available_mb": round(tracker.get_available_ram_mb(), 2),
                "usage_percent": round(tracker.get_ram_usage_percent(), 2),
                "total_gb": round(tracker.get_total_ram(), 2)
            },
            "cpu": {
                "cores": tracker.get_total_cpu(),
                "usage_percent": round(tracker.get_cpu_usage_percent(), 2)
            },
            "gpu": {
                "available": tracker.has_gpu(),
                "vram_total_mb": round(tracker.get_gpu_vram_mb(), 2) if tracker.get_gpu_vram_mb() else None,
                "vram_used_mb": round(tracker.get_used_gpu_vram_mb(), 2) if tracker.get_used_gpu_vram_mb() else None,
                "vram_total_gb": round(tracker.get_gpu_vram(), 2) if tracker.get_gpu_vram() else None
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

