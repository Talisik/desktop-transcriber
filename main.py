import json
import sys
import argparse
from pathlib import Path
import torch

from transcriber.concretions.whisperx_implementation.whisperx_transcriber import WhisperXTranscriber
from transcriber.concretions.whisperx_implementation.chemas.payload.transcriber_argument_schema import WhisperXTranscriberArgumentSchema
from transcriber.concretions.whisperx_implementation.chemas.custom_types.parameter_types import Device, ComputeType, WhisperModel


def main():
    parser = argparse.ArgumentParser(description="transcribe audio files using whisperx")
    parser.add_argument(
        "audio_file",
        type=str,
        help="path to audio/video file to transcribe"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="whisper model name (e.g., tiny, base, small, medium, large, large-v2, large-v3, turbo)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="resources/models",
        help="path to model directory (default: resources/models)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="device to use (cuda/cpu). defaults to cpu"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["float16", "float32", "int8"],
        default=None,
        help="compute type. auto-detects based on device if not provided"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size for transcription (default: 16)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output json file path (default: prints to stdout)"
    )
    
    args = parser.parse_args()
    
    # validate audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # use provided device (defaults to cpu)
    device = args.device
    
    # auto-detect compute type if not provided
    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float32" if device == "cpu" else "float16"
    
    print(f"🎙️  transcribing: {args.audio_file}")
    print(f"   model: {args.model}")
    print(f"   model path: {args.model_path}")
    print(f"   device: {device}")
    print(f"   compute type: {compute_type}")
    print(f"   batch size: {args.batch_size}")
    
    # initialize transcriber
    transcriber = WhisperXTranscriber(device=device)
    
    # create payload
    payload = WhisperXTranscriberArgumentSchema(
        audio_file=str(audio_path.absolute()),
        whisper_model=args.model,
        device=device,
        compute_type=compute_type,
        batch_size=args.batch_size,
        download_root=args.model_path
    )
    
    # transcribe
    print("\n⏳ transcribing...")
    result, model = transcriber.transcribe(payload)
    
    # output result
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\ntranscription saved to: {args.output}")
    else:
        print("\ntranscription result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
