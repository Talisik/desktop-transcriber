"""
Utility functions for transcript processing.
"""
from typing import List, Dict, Any, Optional
from collections import Counter
import uuid
import re
import asyncio


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


def _get_segment_speaker(words: List[Dict[str, Any]]) -> Optional[str]:
    """
    Determine speaker for a segment using majority vote from words.
    
    Args:
        words: List of word dicts with optional 'speaker' field
    
    Returns:
        Speaker ID (e.g., "SPEAKER_00") or None if no speaker info
    """
    speakers = [w.get("speaker") for w in words if w.get("speaker")]
    if not speakers:
        return None
    
    # majority vote
    speaker_counts = Counter(speakers)
    return speaker_counts.most_common(1)[0][0]


def convert_to_cc(
    segments: List[Dict[str, Any]],
    cpl: int = 40,
    segment_start_index: int = 0,
    offset: float = 0.0,
    include_speaker: bool = True
) -> Dict[str, Any]:
    """
    Convert transcript segments to closed caption format with CPL limit.
    
    Processes word-level transcription data and converts it into CC segments
    where each segment does not exceed the specified characters per line (CPL).
    Words are never split mid-character, only at word boundaries.
    
    Args:
        segments: List of utterance segments with word-level timestamps
                 Each segment should have: text, start, end, words array
                 Each word should have: word, start, end, score, speaker (optional)
        cpl: Characters per line limit (default: 40)
        segment_start_index: Starting index for segment numbering (default: 0)
        offset: Time offset in milliseconds for chunked audio (default: 0.0)
        include_speaker: Whether to include speaker information (default: True)
    
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
        current_speaker = None
        
        for i, word in enumerate(words):
            word_text = word.get("word", "").strip()
            word_speaker = word.get("speaker")
            
            if not word_text:
                continue
            
            # check for speaker change
            if include_speaker and word_speaker and word_speaker != current_speaker:
                # if we have content and speaker changed, save current segment
                if current_text and current_speaker is not None:
                    cc_segments.append({
                        "id": str(uuid.uuid4()),
                        "index": segment_index,
                        "text": current_text,
                        "start": format_timestamp(segment_start_time),
                        "end": format_timestamp(segment_end_time),
                        "speaker": current_speaker,
                        "words": current_words
                    })
                    segment_index += 1
                    
                    # reset for new speaker
                    current_text = ""
                    current_words = []
                    segment_start_time = None
                
                current_speaker = word_speaker
            
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
                word_data = {
                    "word": word_text,
                    "start": word.get("start", 0) + offset_seconds,
                    "end": word.get("end", 0) + offset_seconds,
                    "score": word.get("score", 0.0)
                }
                if include_speaker and word_speaker:
                    word_data["speaker"] = word_speaker
                current_words.append(word_data)
                
                # update segment timing
                if segment_start_time is None:
                    segment_start_time = word.get("start", 0) + offset_seconds
                    if include_speaker and word_speaker:
                        current_speaker = word_speaker
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
                        "speaker": current_speaker if include_speaker else None,
                        "words": current_words
                    })
                    segment_index += 1
                
                # start new segment with current word
                current_text = word_text
                current_words = [{
                    "word": word_text,
                    "start": word.get("start", 0) + offset_seconds,
                    "end": word.get("end", 0) + offset_seconds,
                    "score": word.get("score", 0.0),
                    **({"speaker": word_speaker} if include_speaker and word_speaker else {})
                }]
                segment_start_time = word.get("start", 0) + offset_seconds
                segment_end_time = word.get("end", 0) + offset_seconds
                if include_speaker and word_speaker:
                    current_speaker = word_speaker
        
        # save any remaining segment
        if current_text:
            cc_segments.append({
                "id": str(uuid.uuid4()),
                "index": segment_index,
                "text": current_text,
                "start": format_timestamp(segment_start_time),
                "end": format_timestamp(segment_end_time),
                "speaker": current_speaker if include_speaker else None,
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


def cc_to_srt(cc_segments: List[Dict[str, Any]], include_speaker: bool = True) -> str:
    """
    Convert CC segments to SRT (SubRip) subtitle format.
    
    SRT format:
    - Sequence number
    - Start --> End (HH:MM:SS,mmm format with comma)
    - Subtitle text (optionally with speaker label)
    - Blank line
    
    Args:
        cc_segments: List of CC segments with id, index, text, start, end, speaker (optional)
        include_speaker: Whether to prefix text with speaker label (default: True)
    
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
        
        # text with optional speaker label
        text = segment["text"]
        if include_speaker and segment.get("speaker"):
            speaker = segment["speaker"]
            srt_lines.append(f"[{speaker}] {text}")
        else:
            srt_lines.append(text)
        
        # blank line
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def cc_to_vtt(cc_segments: List[Dict[str, Any]], include_speaker: bool = True) -> str:
    """
    Convert CC segments to WebVTT (Web Video Text Tracks) subtitle format.
    
    VTT format:
    - WEBVTT header
    - Blank line
    - Start --> End (HH:MM:SS.mmm format with period)
    - Subtitle text (optionally with speaker label)
    - Blank line
    
    Args:
        cc_segments: List of CC segments with id, index, text, start, end, speaker (optional)
        include_speaker: Whether to prefix text with speaker label (default: True)
    
    Returns:
        VTT formatted string
    """
    vtt_lines = ["WEBVTT", ""]
    
    for segment in cc_segments:
        # timestamps (already in correct format with period)
        start = segment["start"]
        end = segment["end"]
        vtt_lines.append(f"{start} --> {end}")
        
        # text with optional speaker label
        text = segment["text"]
        if include_speaker and segment.get("speaker"):
            speaker = segment["speaker"]
            vtt_lines.append(f"<v {speaker}>{text}</v>")
        else:
            vtt_lines.append(text)
        
        # blank line
        vtt_lines.append("")
    
    return "\n".join(vtt_lines)


def create_paragraphed_transcript(
    cc_segments: List[Dict[str, Any]],
    target_duration: float = 30.0,
    max_duration: float = 35.0,
    include_speaker: bool = True
) -> Dict[str, Any]:
    """
    Create paragraphed transcript by grouping CC segments into ~30s chunks.
    
    Uses soft splitting - finds nearest segment boundary around target duration
    to avoid cutting mid-sentence. Tracks speaker changes within paragraphs.
    
    Args:
        cc_segments: List of CC segments with text, start, end timestamps, speaker (optional)
        target_duration: Target duration per paragraph in seconds (default: 30s)
        max_duration: Maximum duration before forcing split (default: 35s)
        include_speaker: Whether to include speaker information (default: True)
    
    Returns:
        Dict with paragraphs array containing grouped segments with speaker info
    """
    paragraphs = []
    current_paragraph = []
    current_texts = []
    words = []
    paragraph_start_time = None
    paragraph_start_seconds = None
    paragraph_speakers = []  # track all speakers in paragraph
    
    for segment in cc_segments:
        # parse timestamps
        segment_start_seconds = _parse_timestamp_to_seconds(segment["start"])
        segment_end_seconds = _parse_timestamp_to_seconds(segment["end"])
        segment_speaker = segment.get("speaker") if include_speaker else None
        
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
            
            # determine paragraph speaker(s)
            paragraph_data = {
                "index": len(paragraphs),
                "text": paragraph_text,
                "start": paragraph_start_time,
                "words": words,
                "end": paragraph_end_time,
                "duration": round(duration, 3),
                "word_count": word_count,
                "char_count": len(paragraph_text)
            }
            
            if include_speaker:
                # get unique speakers in paragraph
                unique_speakers = list(set(paragraph_speakers))
                if len(unique_speakers) == 1:
                    paragraph_data["speaker"] = unique_speakers[0]
                elif len(unique_speakers) > 1:
                    # multiple speakers - use majority vote
                    speaker_counts = Counter(paragraph_speakers)
                    paragraph_data["speaker"] = speaker_counts.most_common(1)[0][0]
                    paragraph_data["speakers"] = unique_speakers  # list all speakers
            
            paragraphs.append(paragraph_data)
            
            # start new paragraph
            current_paragraph = [segment]
            current_texts = [segment["text"]]
            paragraph_start_time = segment["start"]
            paragraph_start_seconds = segment_start_seconds
            words = segment.get("words", [])
            paragraph_speakers = [segment_speaker] if segment_speaker else []
        else:
            # add to current paragraph
            current_paragraph.append(segment)
            current_texts.append(segment["text"])
            segment_words = segment.get("words", [])
            words.extend(segment_words)
            if include_speaker and segment_speaker:
                paragraph_speakers.append(segment_speaker)

    # save final paragraph
    if current_paragraph:
        paragraph_text = " ".join(current_texts)
        paragraph_end_time = current_paragraph[-1]["end"]
        paragraph_end_seconds = _parse_timestamp_to_seconds(paragraph_end_time)
        duration = paragraph_end_seconds - paragraph_start_seconds
        word_count = len(paragraph_text.split())
        
        paragraph_data = {
            "index": len(paragraphs),
            "text": paragraph_text,
            "words": words,
            "start": paragraph_start_time,
            "end": paragraph_end_time,
            "duration": round(duration, 3),
            "word_count": word_count,
            "char_count": len(paragraph_text)
        }
        
        if include_speaker:
            # get unique speakers in paragraph
            unique_speakers = list(set(paragraph_speakers))
            if len(unique_speakers) == 1:
                paragraph_data["speaker"] = unique_speakers[0]
            elif len(unique_speakers) > 1:
                # multiple speakers - use majority vote
                speaker_counts = Counter(paragraph_speakers)
                paragraph_data["speaker"] = speaker_counts.most_common(1)[0][0]
                paragraph_data["speakers"] = unique_speakers  # list all speakers
        
        paragraphs.append(paragraph_data)
    
    return {
        "paragraphs": paragraphs,
        "total_paragraphs": len(paragraphs),
    }


def extract_full_text_from_segments(segments: List[Dict[str, Any]]) -> str:
    """
    Extracts the full text content from a list of whisperx segments.
    
    Args:
        segments: List of whisperx segments with 'text' field
        
    Returns:
        Full text string with all segment texts joined
    """
    return " ".join(segment.get("text", "") for segment in segments)


def normalize_text_for_matching(text: str) -> str:
    """
    Normalizes text by converting to lowercase, removing punctuation,
    and normalizing whitespace for robust matching.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text string
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def map_words_to_sentences(
    all_segments: List[Dict[str, Any]],
    chunker_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Maps whisperx word-level segments to munchkin-chunker's sentences and paragraphs,
    calculating timestamps and building the hierarchical structure.
    
    Args:
        all_segments: List of whisperx segments with words
        chunker_result: Result from munchkin chunker with final_chunks_results
        
    Returns:
        List of paragraph dictionaries with sentences and words mapped
    """
    # Flatten all words from segments, preserving order and metadata
    # word.copy() preserves all fields including 'speaker' when diarization is enabled
    all_words = []
    for segment in all_segments:
        if "words" in segment:
            for word in segment["words"]:
                all_words.append(word.copy())  # Preserves speaker labels and all other fields
    
    if not all_words:
        return []
    
    # Get chunks from chunker result
    chunks = chunker_result.get("final_chunks_results", [])
    if not chunks:
        return []
    
    paragraphs = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_sentences = chunk.get("sentences", [])
        chunk_content = chunk.get("content", "")
        
        # Track word index as we match
        word_idx = 0
        paragraph_words = []
        sentences_data = []
        
        for sentence_text in chunk_sentences:
            # Normalize sentence text for matching
            normalized_sentence = normalize_text_for_matching(sentence_text)
            
            # Extract words from normalized sentence
            sentence_words_text = normalized_sentence.split()
            matched_words = []
            
            # Try to match words sequentially
            for word_text in sentence_words_text:
                if not word_text:  # Skip empty words
                    continue
                    
                # Look for matching word starting from current position
                found = False
                # First try exact match after normalization
                for i in range(word_idx, len(all_words)):
                    word_obj = all_words[i]
                    word_normalized = normalize_text_for_matching(word_obj.get("word", ""))
                    
                    # Exact match after normalization
                    # word_obj is from all_words which contains copied word objects with all fields (including speaker)
                    if word_text == word_normalized:
                        matched_words.append(word_obj)  # Preserves speaker label if present
                        word_idx = i + 1
                        found = True
                        break
                
                # If not found, try substring match
                if not found:
                    for i in range(word_idx, len(all_words)):
                        word_obj = all_words[i]
                        word_normalized = normalize_text_for_matching(word_obj.get("word", ""))
                        
                        # Check if normalized word matches (substring or contains)
                        # word_obj preserves all fields including speaker label
                        if word_text in word_normalized or word_normalized in word_text:
                            matched_words.append(word_obj)  # Preserves speaker label if present
                            word_idx = i + 1
                            found = True
                            break
                
                # If still not found, skip this word (might be punctuation-only or unmatched)
                if not found:
                    # Try to find closest match within next 5 words
                    best_match = None
                    best_score = 0
                    for i in range(word_idx, min(word_idx + 5, len(all_words))):
                        word_obj = all_words[i]
                        word_normalized = normalize_text_for_matching(word_obj.get("word", ""))
                        # Simple similarity: count matching characters
                        if word_text and word_normalized:
                            common_chars = sum(1 for c in word_text if c in word_normalized)
                            score = common_chars / max(len(word_text), len(word_normalized))
                            if score > best_score and score > 0.5:  # At least 50% similarity
                                best_score = score
                                best_match = (i, word_obj)
                    
                    if best_match:
                        matched_words.append(best_match[1])  # Preserves speaker label if present
                        word_idx = best_match[0] + 1
            
            # Calculate sentence timestamps
            if matched_words:
                sentence_start = min(w.get("start", 0.0) for w in matched_words)
                sentence_end = max(w.get("end", 0.0) for w in matched_words)
            else:
                # Fallback: use previous word's end or 0
                sentence_start = all_words[word_idx - 1].get("end", 0.0) if word_idx > 0 else 0.0
                sentence_end = sentence_start
            
            # Build sentence data (keep timestamps as floats)
            sentence_data = {
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": matched_words
            }
            sentences_data.append(sentence_data)
            # Add words to paragraph_words (preserves all fields including speaker labels)
            paragraph_words.extend(matched_words)  # All word fields including speaker are preserved
        
        # Calculate paragraph timestamps
        if paragraph_words:
            paragraph_start = min(w.get("start", 0.0) for w in paragraph_words)
            paragraph_end = max(w.get("end", 0.0) for w in paragraph_words)
        else:
            paragraph_start = 0.0
            paragraph_end = 0.0
        
        # Ensure continuity: next paragraph starts where previous ends
        if paragraphs and chunk_idx > 0:
            previous_paragraph_end = paragraphs[-1].get("end", 0.0)
            # Only adjust if there's a gap (don't create overlaps)
            if paragraph_start > previous_paragraph_end:
                paragraph_start = previous_paragraph_end
        
        # Build paragraph data (keep timestamps as floats)
        paragraph_data = {
            "index": chunk_idx,
            "text": chunk_content,
            "start": paragraph_start,
            "end": paragraph_end,
            "paragraph_words": paragraph_words,
            "sentences": sentences_data
        }
        paragraphs.append(paragraph_data)
    
    return paragraphs


def build_chunked_transcript(
    all_segments: List[Dict[str, Any]],
    chunker_result: Dict[str, Any],
    process_id: str,
    machine_name: str,
    video_file: str,
    model_name: str,
    language: str,
    diarized: bool = False,
    num_speakers: int = 0
) -> Dict[str, Any]:
    """
    Build hierarchical chunked transcript JSON structure.
    
    Args:
        all_segments: List of whisperx segments with words
        chunker_result: Result from munchkin chunker
        process_id: Process ID
        machine_name: Machine name
        video_file: Video file identifier
        model_name: Model name used
        language: Language code
        diarized: Whether diarization was enabled
        num_speakers: Number of speakers (if diarized)
        
    Returns:
        Dictionary with chunked transcript structure
    """
    paragraphs = map_words_to_sentences(all_segments, chunker_result)
    
    result = {
        "process_id": process_id,
        "machine_name": machine_name,
        "video_file": video_file,
        "model_name": model_name,
        "language": language,
        "total_paragraphs": len(paragraphs),
        "paragraphs": paragraphs
    }
    
    if diarized:
        result["diarized"] = True
        result["num_speakers"] = num_speakers
    
    return result

