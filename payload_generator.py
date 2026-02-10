#!/usr/bin/env python3
"""
Payload generator from salina_languini_results database.
Converts database rows into TranscriptionPayloadSchema format for transcriber.
"""
import json
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path


def calculate_language_stats(
    language_chunks: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate language statistics from chunks and metadata.
    
    Args:
        language_chunks: List of language chunk dicts with 'language', 'duration', 'confidence'
        metadata: Metadata dict with language_distribution info
    
    Returns:
        Dict with language_stats structure
    """
    # Calculate durations per language
    language_durations = {}
    language_counts = {}
    language_confidences = {}
    
    for chunk in language_chunks:
        lang = chunk.get("language", "unknown")
        duration = chunk.get("duration", 0.0)
        confidence = chunk.get("confidence", 0.0)
        
        if lang not in language_durations:
            language_durations[lang] = 0.0
            language_counts[lang] = 0
            language_confidences[lang] = []
        
        language_durations[lang] += duration
        language_counts[lang] += 1
        language_confidences[lang].append(confidence)
    
    # Calculate total duration
    total_duration = sum(language_durations.values())
    
    # Calculate percentages
    language_percentages = {}
    for lang, duration in language_durations.items():
        if total_duration > 0:
            language_percentages[lang] = (duration / total_duration) * 100
        else:
            language_percentages[lang] = 0.0
    
    # Determine major language (highest duration)
    major_language = max(language_durations.items(), key=lambda x: x[1])[0] if language_durations else "unknown"
    major_language_percentage = language_percentages.get(major_language, 0.0)
    
    return {
        "major_language_percentage": major_language_percentage,
        "language_breakdown": {
            "counts": language_counts,
            "durations": language_durations,
            "percentages": language_percentages
        }
    }


def map_chunks_to_merged_mappings(
    language_chunks: List[Dict[str, Any]],
    audio_file_path: str
) -> List[Dict[str, Any]]:
    """
    Map language chunks to merged_mappings format.
    
    Args:
        language_chunks: List of language chunk dicts from database
        audio_file_path: Path to audio file
    
    Returns:
        List of merged_mapping dicts
    """
    merged_mappings = []
    
    for chunk in language_chunks:
        merged_mapping = {
            "failed": False,
            "audio_file": str(audio_file_path),
            "language_code": chunk.get("language", "unknown"),
            "text": "",
            "confidence": chunk.get("confidence", 0.0),
            "language_confidence": chunk.get("confidence", 0.0),
            "transcription_backend": "sieve",
            "segments": [],
            "words": [],
            "start": chunk.get("start", 0.0),
            "end": chunk.get("end", 0.0),
            "duration": chunk.get("duration", 0.0)
        }
        merged_mappings.append(merged_mapping)
    
    return merged_mappings


def create_payload_from_db_row(
    db_row: tuple,
    audio_file_path: str
) -> Dict[str, Any]:
    """
    Create transcription payload from database row.
    
    Args:
        db_row: Database row tuple (id, process_id, vad_result_id, language_results_json, 
                metadata, status, created_at, updated_at, error_message)
        audio_file_path: Path to audio file
    
    Returns:
        Dict matching TranscriptionPayloadSchema format
    """
    # Extract fields from row
    # Row structure: (id, process_id, vad_result_id, language_results_json, metadata, 
    #                 status, created_at, updated_at, error_message)
    process_id = db_row[1]
    language_results_json_str = db_row[3]
    metadata_str = db_row[4]
    
    # Parse JSON strings
    try:
        language_chunks = json.loads(language_results_json_str) if language_results_json_str else []
    except (json.JSONDecodeError, TypeError):
        language_chunks = []
    
    try:
        metadata = json.loads(metadata_str) if metadata_str else {}
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    
    # Calculate language stats
    language_stats = calculate_language_stats(language_chunks, metadata)
    
    # Determine major language
    language_breakdown = language_stats.get("language_breakdown", {})
    durations = language_breakdown.get("durations", {})
    major_language = max(durations.items(), key=lambda x: x[1])[0] if durations else "unknown"
    
    # Determine language classification
    multilingual_mode = metadata.get("multilingual_mode", False)
    language_classification = "multilingual" if multilingual_mode else "single_language"
    
    # Map chunks to merged_mappings
    merged_mappings = map_chunks_to_merged_mappings(language_chunks, audio_file_path)
    
    # Build payload
    payload = {
        "language_stats": language_stats,
        "language_code": major_language,
        "process_id": process_id,
        "language_classification": language_classification,
        "merged_mappings": merged_mappings
    }
    
    return payload


def get_db_row_by_process_id(
    process_id: str,
    db_path: str = "salina_vad.db"
) -> Optional[tuple]:
    """
    Query database for row by process_id.
    
    Args:
        process_id: Process ID to search for
        db_path: Path to database file
    
    Returns:
        Database row tuple or None if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM salina_languini_results WHERE process_id = ?",
            (process_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return row
    except (sqlite3.Error, FileNotFoundError) as e:
        print(f"Error querying database: {e}")
        return None


def create_payload_from_db(
    process_id: str,
    audio_file_path: str,
    db_path: str = "salina_vad.db"
) -> Optional[Dict[str, Any]]:
    """
    Create transcription payload from database by process_id.
    
    Args:
        process_id: Process ID to look up in database
        audio_file_path: Path to audio file
        db_path: Path to database file (default: "salina_vad.db")
    
    Returns:
        Dict matching TranscriptionPayloadSchema format, or None if not found
    """
    row = get_db_row_by_process_id(process_id, db_path)
    if row is None:
        return None
    
    return create_payload_from_db_row(row, audio_file_path)


def save_payload_to_file(
    payload: Dict[str, Any],
    output_path: str
) -> Path:
    """
    Save payload to JSON file.
    
    Args:
        payload: Payload dict
        output_path: Path to save JSON file
    
    Returns:
        Path to saved file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate transcription payload from salina_languini_results database"
    )
    parser.add_argument(
        "--process-id",
        type=str,
        required=True,
        help="Process ID to look up in database"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="salina_vad.db",
        help="Path to database file (default: salina_vad.db)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (optional, prints to stdout if not provided)"
    )
    
    args = parser.parse_args()
    
    # Generate payload
    payload = create_payload_from_db(
        process_id=args.process_id,
        audio_file_path=args.audio_file,
        db_path=args.db_path
    )
    
    if payload is None:
        print(f"❌ Process ID '{args.process_id}' not found in database")
        exit(1)
    
    # Save or print
    if args.output:
        output_file = save_payload_to_file(payload, args.output)
        print(f"✅ Payload saved to: {output_file}")
    else:
        print(json.dumps(payload, indent=2))

