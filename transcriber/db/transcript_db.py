"""
Database operations for storing transcript granular data.

Handles saving words, sentences, segments, paragraphs, and speakers
to SQLite database using the existing schema.
"""
import sqlite3
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Get database connection with foreign keys enabled.
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_speakers_table(conn: sqlite3.Connection) -> None:
    """
    Create speakers table if not exists.
    
    Args:
        conn: Database connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            id TEXT PRIMARY KEY,
            process_id TEXT NOT NULL,
            label TEXT NOT NULL,
            color TEXT NOT NULL DEFAULT '#808080',
            timestamp_start REAL NOT NULL,
            timestamp_end REAL NOT NULL,
            date_created DATETIME DEFAULT CURRENT_TIMESTAMP,
            date_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (process_id) REFERENCES transcriptions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_speakers_process_id ON speakers(process_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_speakers_timestamp ON speakers(timestamp_start)")


def create_words_table(conn: sqlite3.Connection) -> None:
    """
    Create words table if not exists.
    
    Args:
        conn: Database connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id TEXT PRIMARY KEY,
            process_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            base_version TEXT,
            base_version_deleted INTEGER NOT NULL DEFAULT 0,
            is_conflicted INTEGER NOT NULL DEFAULT 0,
            timestamp_start REAL NOT NULL,
            timestamp_end REAL NOT NULL,
            word TEXT NOT NULL,
            word_segment_idx INTEGER NOT NULL,
            word_idx INTEGER NOT NULL,
            word_type TEXT,
            date_created DATETIME DEFAULT CURRENT_TIMESTAMP,
            date_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            confidence REAL,
            speaker TEXT,
            speaker_id TEXT,
            color TEXT,
            speaker_name TEXT,
            
            FOREIGN KEY (process_id) REFERENCES transcriptions(id) ON DELETE CASCADE,
            FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE SET NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_words_process_id ON words(process_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_words_speaker_id ON words(speaker_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_words_timestamp ON words(timestamp_start)")


def create_segments_table(conn: sqlite3.Connection) -> None:
    """
    Create segments table if not exists.
    
    Args:
        conn: Database connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY,
            process_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            base_version TEXT,
            base_version_deleted INTEGER NOT NULL DEFAULT 0,
            is_conflicted INTEGER NOT NULL DEFAULT 0,
            timestamp_start REAL NOT NULL,
            timestamp_end REAL NOT NULL,
            speaker TEXT,
            speaker_id TEXT,
            date_created DATETIME DEFAULT CURRENT_TIMESTAMP,
            date_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            proofread INTEGER NOT NULL DEFAULT 0,
            words TEXT,
            text TEXT NOT NULL,
            
            FOREIGN KEY (process_id) REFERENCES transcriptions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_process_id ON segments(process_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_timestamp ON segments(timestamp_start)")


def create_sentences_table(conn: sqlite3.Connection) -> None:
    """
    Create sentences table if not exists.
    
    Args:
        conn: Database connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id TEXT PRIMARY KEY,
            process_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            base_version TEXT,
            base_version_deleted INTEGER NOT NULL DEFAULT 0,
            is_conflicted INTEGER NOT NULL DEFAULT 0,
            chapter TEXT,
            timestamp_start REAL,
            timestamp_end REAL,
            chunker_used TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            idioms_available INTEGER NOT NULL DEFAULT 0,
            text TEXT NOT NULL,
            words TEXT,
            
            FOREIGN KEY (process_id) REFERENCES transcriptions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_process_id ON sentences(process_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_timestamp ON sentences(timestamp_start)")


def create_paragraphs_table(conn: sqlite3.Connection) -> None:
    """
    Create paragraphs table if not exists.
    
    Args:
        conn: Database connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paragraphs (
            id TEXT PRIMARY KEY,
            process_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            base_version TEXT,
            base_version_deleted INTEGER NOT NULL DEFAULT 0,
            is_conflicted INTEGER NOT NULL DEFAULT 0,
            chapter TEXT,
            timestamp_start REAL,
            timestamp_end REAL,
            original_paragraph_id TEXT,
            created_by TEXT,
            updated_by TEXT,
            date_created DATETIME DEFAULT CURRENT_TIMESTAMP,
            date_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            words TEXT,
            text TEXT NOT NULL,
            chunker_used TEXT NOT NULL,
            
            FOREIGN KEY (process_id) REFERENCES transcriptions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_process_id ON paragraphs(process_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_timestamp ON paragraphs(timestamp_start)")


def extract_speaker_segments(
    all_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract continuous speaker segments from word-level data.
    Groups consecutive words with same speaker label into time ranges.
    
    Args:
        all_segments: List of whisperx segments with words
    
    Returns:
        List of speaker segments with label, start, end
    """
    speaker_segments = []
    current_speaker = None
    current_start = None
    current_end = None
    
    for segment in all_segments:
        for word in segment.get("words", []):
            word_speaker = word.get("speaker")
            word_start = word.get("start", 0.0)
            word_end = word.get("end", 0.0)
            
            if word_speaker:
                if word_speaker == current_speaker:
                    # Extend current segment
                    current_end = max(current_end, word_end) if current_end else word_end
                else:
                    # Save previous segment if exists
                    if current_speaker and current_start is not None:
                        speaker_segments.append({
                            "label": current_speaker,
                            "start": current_start,
                            "end": current_end
                        })
                    
                    # Start new segment
                    current_speaker = word_speaker
                    current_start = word_start
                    current_end = word_end
    
    # Save final segment
    if current_speaker and current_start is not None:
        speaker_segments.append({
            "label": current_speaker,
            "start": current_start,
            "end": current_end
        })
    
    return speaker_segments


def _parse_timestamp_to_seconds(timestamp_str: str) -> float:
    """
    Parse HH:MM:SS.mmm timestamp string to seconds.
    
    Args:
        timestamp_str: Timestamp in format "HH:MM:SS.mmm"
    
    Returns:
        Time in seconds as float
    """
    parts = timestamp_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    secs_parts = parts[2].split(".")
    seconds = int(secs_parts[0])
    milliseconds = int(secs_parts[1])
    
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds


def create_transcriptions_table(conn: sqlite3.Connection) -> None:
    """
    Create transcriptions table if not exists (minimal schema).
    Full schema should be provided by teammate's migration.
    
    Args:
        conn: Database connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id TEXT PRIMARY KEY,
            transcription_name TEXT,
            video_source TEXT,
            transcribe_language TEXT,
            transcribe_language_iso TEXT,
            process_status TEXT DEFAULT 'completed',
            date_created DATETIME DEFAULT CURRENT_TIMESTAMP,
            date_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


def create_or_update_transcription(
    conn: sqlite3.Connection,
    process_id: str,
    transcription_data: Dict[str, Any]
) -> None:
    """
    Create or update transcription record.
    
    Args:
        conn: Database connection
        process_id: Process ID (maps to transcriptions.id)
        transcription_data: Metadata dict with language, video_source, etc.
    """
    # Ensure transcriptions table exists
    create_transcriptions_table(conn)
    
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM transcriptions WHERE id = ?", (process_id,))
    exists = cursor.fetchone()
    
    if exists:
        cursor.execute("""
            UPDATE transcriptions 
            SET date_updated = CURRENT_TIMESTAMP,
                transcribe_language = ?,
                transcribe_language_iso = ?
            WHERE id = ?
        """, (
            transcription_data.get('language'),
            transcription_data.get('language_code'),
            process_id
        ))
    else:
        cursor.execute("""
            INSERT INTO transcriptions (
                id, transcription_name, video_source,
                transcribe_language, transcribe_language_iso,
                process_status, date_created, date_updated
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (
            process_id,
            transcription_data.get('transcription_name', process_id),
            transcription_data.get('video_source', ''),
            transcription_data.get('language'),
            transcription_data.get('language_code'),
            'completed'
        ))


def insert_speakers_batch(
    conn: sqlite3.Connection,
    process_id: str,
    all_segments: List[Dict[str, Any]],
    default_color: str = '#808080'
) -> Dict[str, str]:
    """
    Insert speakers and return mapping: speaker_label -> speaker_id.
    
    Args:
        conn: Database connection
        process_id: Process ID
        all_segments: List of whisperx segments with words
        default_color: Default hex color for speakers
    
    Returns:
        Dictionary mapping speaker label to speaker id
    """
    create_speakers_table(conn)
    
    speaker_segments = extract_speaker_segments(all_segments)
    speaker_map = {}  # label -> id mapping
    
    # Track unique speakers by label (same label = same speaker_id)
    seen_labels = set()
    speakers_data = []
    
    for seg in speaker_segments:
        label = seg["label"]
        
        if label not in seen_labels:
            seen_labels.add(label)
            speaker_id = str(uuid.uuid4())
            speaker_map[label] = speaker_id
            
            speakers_data.append((
                speaker_id,
                process_id,
                label,
                default_color,
                seg["start"],
                seg["end"],
                datetime.now(),
                datetime.now()
            ))
    
    if speakers_data:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO speakers (
                id, process_id, label, color,
                timestamp_start, timestamp_end,
                date_created, date_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, speakers_data)
    
    return speaker_map


def insert_words_batch(
    conn: sqlite3.Connection,
    process_id: str,
    all_segments: List[Dict[str, Any]],
    speaker_map: Dict[str, str]
) -> None:
    """
    Insert words from all_segments.
    
    Args:
        conn: Database connection
        process_id: Process ID
        all_segments: List of whisperx segments with words
        speaker_map: Dictionary mapping speaker label to speaker id
    """
    words_data = []
    
    for seg_idx, segment in enumerate(all_segments):
        segment_words = segment.get("words", [])
        for word_idx, word in enumerate(segment_words):
            word_id = str(uuid.uuid4())
            speaker_label = word.get("speaker")
            speaker_id = speaker_map.get(speaker_label) if speaker_label else None
            
            words_data.append((
                word_id,
                process_id,
                0,  # version
                None,  # base_version
                0,  # base_version_deleted
                0,  # is_conflicted
                word.get("start", 0.0),  # timestamp_start (REAL)
                word.get("end", 0.0),  # timestamp_end (REAL)
                word.get("word", ""),
                seg_idx,  # word_segment_idx
                word_idx,  # word_idx
                None,  # word_type
                datetime.now(),  # date_created
                datetime.now(),  # date_updated
                0,  # is_deleted
                word.get("score", 1.0),  # confidence
                None,  # speaker (deprecated, use speaker_id)
                speaker_id,  # speaker_id FK to speakers.id
                None,  # color (deprecated, use speakers.color)
                None  # speaker_name
            ))
    
    if words_data:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO words (
                id, process_id, version, base_version, base_version_deleted,
                is_conflicted, timestamp_start, timestamp_end, word,
                word_segment_idx, word_idx, word_type, date_created,
                date_updated, is_deleted, confidence, speaker, speaker_id,
                color, speaker_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, words_data)


def insert_segments_batch(
    conn: sqlite3.Connection,
    process_id: str,
    cc_segments: List[Dict[str, Any]],
    speaker_map: Dict[str, str]
) -> None:
    """
    Insert segments from CC format (40cpl) - convert timestamps to REAL.
    
    Args:
        conn: Database connection
        process_id: Process ID
        cc_segments: List of CC format segments
        speaker_map: Dictionary mapping speaker label to speaker id (not used, segments.speaker_id is NULL)
    """
    segments_data = []
    
    for segment in cc_segments:
        segment_id = str(uuid.uuid4())
        speaker_label = segment.get("speaker")
        # segments.speaker_id is NULL/separate (not FK to speakers)
        speaker_id = None
        
        # Parse timestamps from formatted string to REAL (seconds)
        timestamp_start = _parse_timestamp_to_seconds(segment["start"])
        timestamp_end = _parse_timestamp_to_seconds(segment["end"])
        
        segments_data.append((
            segment_id,
            process_id,
            0,  # version
            None,  # base_version
            0,  # base_version_deleted
            0,  # is_conflicted
            timestamp_start,  # timestamp_start (REAL)
            timestamp_end,  # timestamp_end (REAL)
            speaker_label,  # speaker (TEXT label)
            speaker_id,  # speaker_id (NULL/separate, not FK)
            datetime.now(),  # date_created
            datetime.now(),  # date_updated
            0,  # is_deleted
            0,  # proofread
            json.dumps(segment.get("words", [])),  # words JSON
            segment.get("text", "")  # text
        ))
    
    if segments_data:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO segments (
                id, process_id, version, base_version, base_version_deleted,
                is_conflicted, timestamp_start, timestamp_end, speaker,
                speaker_id, date_created, date_updated, is_deleted,
                proofread, words, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, segments_data)


def insert_sentences_batch(
    conn: sqlite3.Connection,
    process_id: str,
    paragraphs: List[Dict[str, Any]]
) -> None:
    """
    Insert sentences from chunked transcript paragraphs.
    
    Args:
        conn: Database connection
        process_id: Process ID
        paragraphs: List of paragraph dicts with sentences
    """
    sentences_data = []
    
    for paragraph in paragraphs:
        for sentence in paragraph.get("sentences", []):
            sentence_id = str(uuid.uuid4())
            
            sentences_data.append((
                sentence_id,
                process_id,
                0,  # version
                None,  # base_version
                0,  # base_version_deleted
                0,  # is_conflicted
                None,  # chapter
                sentence.get("start", 0.0),  # timestamp_start (REAL)
                sentence.get("end", 0.0),  # timestamp_end (REAL)
                "munchkin",  # chunker_used
                0,  # is_deleted
                0,  # idioms_available
                sentence.get("text", ""),  # text
                json.dumps(sentence.get("words", []))  # words JSON
            ))
    
    if sentences_data:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO sentences (
                id, process_id, version, base_version, base_version_deleted,
                is_conflicted, chapter, timestamp_start, timestamp_end,
                chunker_used, is_deleted, idioms_available, text, words
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sentences_data)


def insert_paragraphs_batch(
    conn: sqlite3.Connection,
    process_id: str,
    paragraphs: List[Dict[str, Any]]
) -> None:
    """
    Insert paragraphs from chunked transcript.
    
    Args:
        conn: Database connection
        process_id: Process ID
        paragraphs: List of paragraph dicts
    """
    paragraphs_data = []
    
    for paragraph in paragraphs:
        paragraph_id = str(uuid.uuid4())
        
        paragraphs_data.append((
            paragraph_id,
            process_id,
            0,  # version
            None,  # base_version
            0,  # base_version_deleted
            0,  # is_conflicted
            None,  # chapter
            paragraph.get("start", 0.0),  # timestamp_start (REAL)
            paragraph.get("end", 0.0),  # timestamp_end (REAL)
            None,  # original_paragraph_id
            None,  # created_by
            None,  # updated_by
            datetime.now(),  # date_created
            datetime.now(),  # date_updated
            0,  # is_deleted
            json.dumps(paragraph.get("paragraph_words", [])),  # words JSON
            paragraph.get("text", ""),  # text
            "munchkin"  # chunker_used
        ))
    
    if paragraphs_data:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO paragraphs (
                id, process_id, version, base_version, base_version_deleted,
                is_conflicted, chapter, timestamp_start, timestamp_end,
                original_paragraph_id, created_by, updated_by, date_created,
                date_updated, is_deleted, words, text, chunker_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, paragraphs_data)


def save_transcript_to_db(
    db_path: str,
    process_id: str,
    all_segments: List[Dict[str, Any]],
    cc_result: Dict[str, Any],
    chunked_transcript_data: Dict[str, Any],
    transcription_metadata: Dict[str, Any]
) -> None:
    """
    Main function to save all transcript data to database.
    
    Args:
        db_path: Path to SQLite database file
        process_id: Process ID (maps to transcriptions.id)
        all_segments: List of whisperx segments with words
        cc_result: CC format result with cc_segments
        chunked_transcript_data: Chunked transcript data with paragraphs
        transcription_metadata: Metadata dict with language, video_source, etc.
    
    Raises:
        Exception: If database operations fail
    """
    conn = get_db_connection(db_path)
    
    try:
        conn.execute("BEGIN TRANSACTION")
        
        # 0. Create all tables
        logger.info(f"Creating database tables for process_id: {process_id}")
        create_transcriptions_table(conn)
        create_speakers_table(conn)
        create_words_table(conn)
        create_segments_table(conn)
        create_sentences_table(conn)
        create_paragraphs_table(conn)
        logger.info("All database tables created/verified")
        
        # 1. Create/update transcription record
        create_or_update_transcription(conn, process_id, transcription_metadata)
        logger.info(f"Transcription record created/updated: {process_id}")
        
        # 2. Insert speakers FIRST (get mapping)
        speaker_map = insert_speakers_batch(conn, process_id, all_segments)
        logger.info(f"Inserted {len(speaker_map)} speakers")
        
        # 3. Insert words (with speaker_id FK)
        word_count = sum(len(seg.get("words", [])) for seg in all_segments)
        insert_words_batch(conn, process_id, all_segments, speaker_map)
        logger.info(f"Inserted {word_count} words")
        
        # 4. Insert segments (from CC format, with NULL speaker_id)
        segment_count = len(cc_result.get("cc_segments", []))
        insert_segments_batch(conn, process_id, cc_result["cc_segments"], speaker_map)
        logger.info(f"Inserted {segment_count} segments")
        
        # 5. Insert sentences and paragraphs
        paragraphs = chunked_transcript_data.get("paragraphs", [])
        sentence_count = sum(len(p.get("sentences", [])) for p in paragraphs)
        insert_sentences_batch(conn, process_id, paragraphs)
        logger.info(f"Inserted {sentence_count} sentences")
        insert_paragraphs_batch(conn, process_id, paragraphs)
        logger.info(f"Inserted {len(paragraphs)} paragraphs")
        
        conn.commit()
        logger.info(f"Successfully saved transcript data to database: {db_path}")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save transcript to database: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

