import math
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = max(0, int(math.floor(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def get_transcript_data(video_id: str, preferred_languages: list = None) -> dict:
    """
    Fetch transcript snippets with timestamps, duration, and formatted time.
    Returns:
    {
        "video_id": str,
        "language": str,
        "language_code": str,
        "is_generated": bool,
        "snippets": [
            {"text": str, "start": float, "duration": float, "time": str, "timestamp_url": str}
        ],
        "full_text": str
    }
    """
    if preferred_languages is None:
        preferred_languages = ["en", "en-US", "en-GB", "es", "hi", "fr", "de"]

    api = YouTubeTranscriptApi()

    try:
        # First try listing transcripts to find the best match
        transcript_list = api.list(video_id)
        
        # Try to find user preferred language
        selected_transcript = None
        try:
            selected_transcript = transcript_list.find_transcript(preferred_languages)
        except Exception:
            # Try to find manually created first, then generated
            for t in transcript_list:
                if not t.is_generated:
                    selected_transcript = t
                    break
            if not selected_transcript:
                selected_transcript = next(iter(transcript_list), None)

        if not selected_transcript:
            raise NoTranscriptFound(video_id, preferred_languages)

        fetched = selected_transcript.fetch()
        snippets = []
        for s in fetched.snippets:
            clean_text = s.text.replace("\n", " ").strip()
            if not clean_text:
                continue
            start_sec = float(s.start)
            snippets.append({
                "text": clean_text,
                "start": start_sec,
                "duration": float(s.duration),
                "time": format_timestamp(start_sec),
                "timestamp_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}s"
            })

        full_text = " ".join(item["text"] for item in snippets)

        return {
            "video_id": video_id,
            "language": getattr(selected_transcript, "language", "Unknown"),
            "language_code": getattr(selected_transcript, "language_code", "en"),
            "is_generated": getattr(selected_transcript, "is_generated", False),
            "snippets": snippets,
            "full_text": full_text
        }

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript) as e:
        # Direct fetch fallback
        try:
            fetched = api.fetch(video_id)
            snippets = []
            for s in fetched.snippets:
                clean_text = s.text.replace("\n", " ").strip()
                if not clean_text:
                    continue
                start_sec = float(s.start)
                snippets.append({
                    "text": clean_text,
                    "start": start_sec,
                    "duration": float(s.duration),
                    "time": format_timestamp(start_sec),
                    "timestamp_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}s"
                })
            full_text = " ".join(item["text"] for item in snippets)
            return {
                "video_id": video_id,
                "language": "English",
                "language_code": "en",
                "is_generated": getattr(fetched, "is_generated", True),
                "snippets": snippets,
                "full_text": full_text
            }
        except Exception:
            raise RuntimeError(f"Could not retrieve transcript for video {video_id}: {str(e)}")

def get_transcript(video_id: str) -> str:
    """
    Backwards-compatible function returning the concatenated transcript text.
    """
    data = get_transcript_data(video_id)
    return data["full_text"]