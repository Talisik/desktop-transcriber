"""
Huey worker process runner.
Run this to consume and execute queued transcription tasks.

Usage:
    python huey_worker.py
    
Or use the huey command directly (if installed):
    huey consumer.py consumer.huey
"""
import sys
import subprocess
from pathlib import Path

# Import consumer module to register tasks
# This ensures tasks are registered before starting the consumer
import consumer

# Import huey instance
from transcriber_huey import huey


def run_consumer_subprocess():
    """Run consumer via subprocess (for script mode)"""
    script_dir = Path(__file__).parent
    subprocess.run([
        sys.executable,
        "-m", "huey.bin.huey_consumer",
        "consumer.huey"
    ], cwd=str(script_dir))


def run_consumer_programmatic():
    """Run consumer programmatically (for executable mode)"""
    from huey.bin.huey_consumer import consumer_main
    import sys
    
    # Set up sys.argv as if called from command line
    # Format: ['script_name', 'consumer.huey']
    original_argv = sys.argv
    sys.argv = ['huey_worker', 'consumer.huey']
    
    try:
        consumer_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    print("🚀 starting huey worker")
    print("   press Ctrl+C to stop\n")
    
    try:
        # Check if running as PyInstaller executable
        if getattr(sys, 'frozen', False):
            # Running as executable - use programmatic approach
            run_consumer_programmatic()
        else:
            # Running as script - use subprocess
            run_consumer_subprocess()
    except KeyboardInterrupt:
        print("\n👋 worker stopped")
    except Exception as e:
        print(f"❌ error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
