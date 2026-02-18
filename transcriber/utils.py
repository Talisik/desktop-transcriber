"""
Utility functions for transcript processing.
"""
from typing import List, Dict, Any
import uuid


def format_timestamp(
    seconds: float,
    always_include_hours: bool = True,
    unit: str = "seconds"
) -> str:
    """
    Convert seconds (or milliseconds) to HH:MM:SS.mmm format.
    
    Args:
        seconds: Time value in seconds or milliseconds
        always_include_hours: Whether to always include hours component
        unit: Unit of input value ("seconds" or "milliseconds")
    
    Returns:
        Formatted timestamp string (e.g., "00:01:23.456")
    """
    # convert milliseconds to seconds if needed
    if unit == "milliseconds":
        seconds = seconds / 1000.0
    
    # handle negative values
    if seconds < 0:
        seconds = 0
    
    # calculate components
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    # round to nearest millisecond
    milliseconds = int(round((secs - int(secs)) * 1000))
    secs_int = int(secs)
    
    # format output
    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs_int:02d}.{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{secs_int:02d}.{milliseconds:03d}"


def convert_to_cc(
    segments: List[Dict[str, Any]],
    cpl: int = 40,
    segment_start_index: int = 0,
    offset: float = 0.0
) -> Dict[str, Any]:
    """
    Convert transcript segments to closed caption format with CPL limit.
    
    Processes word-level transcription data and converts it into CC segments
    where each segment does not exceed the specified characters per line (CPL).
    Words are never split mid-character, only at word boundaries.
    
    Args:
        segments: List of utterance segments with word-level timestamps
                 Each segment should have: text, start, end, words array
                 Each word should have: word, start, end, score
        cpl: Characters per line limit (default: 40)
        segment_start_index: Starting index for segment numbering (default: 0)
        offset: Time offset in milliseconds for chunked audio (default: 0.0)
    
    Returns:
        Dict containing cc_segments array with formatted caption data
    """
    cc_segments = []
    segment_index = segment_start_index
    
    # convert offset from milliseconds to seconds
    offset_seconds = offset / 1000.0 if offset > 0 else 0.0
    
    # iterate through each utterance segment
    for segment in segments:
        words = segment.get("words", [])
        
        if not words:
            # skip empty segments
            continue
        
        # track current CC segment being built
        current_words = []
        current_text = ""
        segment_start_time = None
        segment_end_time = None
        
        for i, word in enumerate(words):
            word_text = word.get("word", "").strip()
            
            if not word_text:
                continue
            
            # calculate what the new text would be
            if current_text:
                # add space before word
                proposed_text = current_text + " " + word_text
            else:
                proposed_text = word_text
            
            # check if adding this word would exceed CPL
            if len(proposed_text) <= cpl:
                # word fits, add it to current segment
                current_text = proposed_text
                current_words.append({
                    "word": word_text,
                    "start": word.get("start", 0) + offset_seconds,
                    "end": word.get("end", 0) + offset_seconds,
                    "score": word.get("score", 0.0)
                })
                
                # update segment timing
                if segment_start_time is None:
                    segment_start_time = word.get("start", 0) + offset_seconds
                segment_end_time = word.get("end", 0) + offset_seconds
            else:
                # word doesn't fit, save current segment and start new one
                if current_text:
                    # save the completed segment
                    cc_segments.append({
                        "id": str(uuid.uuid4()),
                        "index": segment_index,
                        "text": current_text,
                        "start": format_timestamp(segment_start_time),
                        "end": format_timestamp(segment_end_time),
                        "words": current_words
                    })
                    segment_index += 1
                
                # start new segment with current word
                current_text = word_text
                current_words = [{
                    "word": word_text,
                    "start": word.get("start", 0) + offset_seconds,
                    "end": word.get("end", 0) + offset_seconds,
                    "score": word.get("score", 0.0)
                }]
                segment_start_time = word.get("start", 0) + offset_seconds
                segment_end_time = word.get("end", 0) + offset_seconds
        
        # save any remaining segment
        if current_text:
            cc_segments.append({
                "id": str(uuid.uuid4()),
                "index": segment_index,
                "text": current_text,
                "start": format_timestamp(segment_start_time),
                "end": format_timestamp(segment_end_time),
                "words": current_words
            })
            segment_index += 1
    
    return {
        "cc_segments": cc_segments,
        "total_segments": len(cc_segments)
    }


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


def cc_to_srt(cc_segments: List[Dict[str, Any]]) -> str:
    """
    Convert CC segments to SRT (SubRip) subtitle format.
    
    SRT format:
    - Sequence number
    - Start --> End (HH:MM:SS,mmm format with comma)
    - Subtitle text
    - Blank line
    
    Args:
        cc_segments: List of CC segments with id, index, text, start, end
    
    Returns:
        SRT formatted string
    """
    srt_lines = []
    
    for idx, segment in enumerate(cc_segments, 1):
        # sequence number
        srt_lines.append(str(idx))
        
        # timestamps (convert period to comma for SRT)
        start = segment["start"].replace(".", ",")
        end = segment["end"].replace(".", ",")
        srt_lines.append(f"{start} --> {end}")
        
        # text
        srt_lines.append(segment["text"])
        
        # blank line
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def cc_to_vtt(cc_segments: List[Dict[str, Any]]) -> str:
    """
    Convert CC segments to WebVTT (Web Video Text Tracks) subtitle format.
    
    VTT format:
    - WEBVTT header
    - Blank line
    - Start --> End (HH:MM:SS.mmm format with period)
    - Subtitle text
    - Blank line
    
    Args:
        cc_segments: List of CC segments with id, index, text, start, end
    
    Returns:
        VTT formatted string
    """
    vtt_lines = ["WEBVTT", ""]
    
    for segment in cc_segments:
        # timestamps (already in correct format with period)
        start = segment["start"]
        end = segment["end"]
        vtt_lines.append(f"{start} --> {end}")
        
        # text
        vtt_lines.append(segment["text"])
        
        # blank line
        vtt_lines.append("")
    
    return "\n".join(vtt_lines)


def create_paragraphed_transcript(
    cc_segments: List[Dict[str, Any]],
    target_duration: float = 30.0,
    max_duration: float = 35.0
) -> Dict[str, Any]:
    """
    Create paragraphed transcript by grouping CC segments into ~30s chunks.
    
    Uses soft splitting - finds nearest segment boundary around target duration
    to avoid cutting mid-sentence.
    
    Args:
        cc_segments: List of CC segments with text, start, end timestamps
        target_duration: Target duration per paragraph in seconds (default: 30s)
        max_duration: Maximum duration before forcing split (default: 35s)
    
    Returns:
        Dict with paragraphs array containing grouped segments
    """
    paragraphs = []
    current_paragraph = []
    current_texts = []
    words = []
    paragraph_start_time = None
    paragraph_start_seconds = None
    
    for segment in cc_segments:
        # parse timestamps
        segment_start_seconds = _parse_timestamp_to_seconds(segment["start"])
        segment_end_seconds = _parse_timestamp_to_seconds(segment["end"])
        
        # initialize first paragraph
        if paragraph_start_time is None:
            paragraph_start_time = segment["start"]
            paragraph_start_seconds = segment_start_seconds
        
        # calculate current paragraph duration if we add this segment
        potential_duration = segment_end_seconds - paragraph_start_seconds
        
        # check if we should split
        should_split = False
        if potential_duration > max_duration:
            # force split if exceeding max duration
            should_split = True
        elif potential_duration > target_duration and len(current_paragraph) > 0:
            # soft split if we've reached target and have content
            should_split = True
        
        if should_split and current_paragraph:
            # save current paragraph
            paragraph_text = " ".join(current_texts)
            paragraph_end_time = current_paragraph[-1]["end"]
            paragraph_end_seconds = _parse_timestamp_to_seconds(paragraph_end_time)
            duration = paragraph_end_seconds - paragraph_start_seconds
            word_count = len(paragraph_text.split())
            
            paragraphs.append({
                "index": len(paragraphs),
                "text": paragraph_text,
                "start": paragraph_start_time,
                "words": words,
                "end": paragraph_end_time,
                "duration": round(duration, 3),
                "word_count": word_count,
                "char_count": len(paragraph_text)
            })
            
            # start new paragraph
            current_paragraph = [segment]
            current_texts = [segment["text"]]
            paragraph_start_time = segment["start"]
            paragraph_start_seconds = segment_start_seconds
            words = segment.get("words", [])
        else:
            # add to current paragraph
            current_paragraph.append(segment)
            current_texts.append(segment["text"])
            segment_words = segment["words"]
            words.extend(segment_words)
        


    # save final paragraph
    if current_paragraph:
        paragraph_text = " ".join(current_texts)
        paragraph_end_time = current_paragraph[-1]["end"]
        paragraph_end_seconds = _parse_timestamp_to_seconds(paragraph_end_time)
        duration = paragraph_end_seconds - paragraph_start_seconds
        word_count = len(paragraph_text.split())
        
        paragraphs.append({
            "index": len(paragraphs),
            "text": paragraph_text,
            "words": words,
            "start": paragraph_start_time,
            "end": paragraph_end_time,
            "duration": round(duration, 3),
            "word_count": word_count,
            "char_count": len(paragraph_text)
        })
    
    return {
        "paragraphs": paragraphs,
        "total_paragraphs": len(paragraphs),
    }

