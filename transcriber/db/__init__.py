"""Database module for storing transcript data."""

from transcriber.db.transcript_db import (
    save_transcript_to_db,
    get_db_connection,
    create_speakers_table,
)

__all__ = [
    "save_transcript_to_db",
    "get_db_connection",
    "create_speakers_table",
]

