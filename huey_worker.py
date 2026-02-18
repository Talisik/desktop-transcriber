import os
import sys
import argparse
from pathlib import Path

# 1. PRE-PARSE ffmpeg-path before any heavy imports
# This prevents the "torchcodec" and "pyannote" warnings from machine learning libs
# that are imported via transcriber_huey/consumer.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huey transcription worker", add_help=False)
    parser.add_argument("--ffmpeg-path", type=str, default=None)
    args, unknown = parser.parse_known_args()

    if args.ffmpeg_path:
        # Validate and normalize path early
        ffmpeg_exe = os.path.normpath(os.path.expanduser(args.ffmpeg_path))
        if os.path.isdir(ffmpeg_exe):
            if sys.platform == 'win32':
                ffmpeg_exe = os.path.join(ffmpeg_exe, 'ffmpeg.exe')
            else:
                ffmpeg_exe = os.path.join(ffmpeg_exe, 'ffmpeg')

        if os.path.exists(ffmpeg_exe):
            os.environ['FFMPEG_PATH'] = ffmpeg_exe
            # Inject to PATH immediately so libraries find it during import
            ffmpeg_dir = os.path.dirname(os.path.abspath(ffmpeg_exe))
            path_env = os.environ.get("PATH", "")
            if ffmpeg_dir not in path_env.split(os.pathsep):
                print(f"[PRE-START] FFmpeg injected into PATH: {ffmpeg_dir}")
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + path_env
        else:
            print(f"WARNING [PRE-START] FFmpeg not found at: {ffmpeg_exe}")

# 2. NOW it is safe to import everything else (heavy modules)
import subprocess

# Import consumer module to register tasks
import consumer

# Import huey instance and configuration tools
from transcriber_huey import huey, set_ffmpeg_path, get_ffmpeg_path


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
    original_argv = sys.argv
    sys.argv = ['huey_worker', 'consumer.huey']

    try:
        consumer_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    # Standard parser for the rest of the arguments
    parser = argparse.ArgumentParser(description="Huey transcription worker")
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        default=None,
        help="Path to ffmpeg executable (default: auto-detect)"
    )
    args = parser.parse_args()

    # Ensure transcriber_huey's internal state matches our early injection
    if args.ffmpeg_path:
        try:
            set_ffmpeg_path(args.ffmpeg_path)
            validated_path = get_ffmpeg_path()
            print(f"starting huey worker with ffmpeg: {validated_path}")
        except RuntimeError as e:
            print(f"ERROR: ffmpeg validation failed: {e}")
            sys.exit(1)
    else:
        print("starting huey worker (auto-detecting ffmpeg)")

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
        print("\nworker stopped")
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
