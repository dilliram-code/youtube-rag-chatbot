import re
import urllib.request
import json
from googleapiclient.discovery import build
from src.config import get_youtube_api_key

def create_youtube_client(api_key: str = None):
    """Create a YouTube Data API client if API key is available."""
    key = api_key or get_youtube_api_key()
    if not key:
        return None
    try:
        return build("youtube", "v3", developerKey=key)
    except Exception:
        return None

def extract_video_id(url_or_id: str) -> str:
    """
    Extract YouTube video ID from various URL formats or raw video ID:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/live/VIDEO_ID
    - Raw 11-char ID
    """
    if not url_or_id:
        raise ValueError("Please provide a YouTube URL or Video ID.")
    
    url = url_or_id.strip()

    # Exact 11 characters ID match
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url

    # Common URL patterns
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/|\/live\/)([a-zA-Z0-9_-]{11})",
        r"[?&]v=([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a valid 11-character YouTube video ID from '{url_or_id}'.")

def get_video_info(video_id: str, api_key: str = None) -> dict:
    """
    Get video metadata using YouTube Data API v3 with automatic oEmbed fallback.
    """
    clean_id = extract_video_id(video_id)
    default_thumbnail = f"https://img.youtube.com/vi/{clean_id}/maxresdefault.jpg"
    fallback_thumbnail = f"https://img.youtube.com/vi/{clean_id}/hqdefault.jpg"

    # 1. Try YouTube Data API
    youtube = create_youtube_client(api_key)
    if youtube:
        try:
            response = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=clean_id
            ).execute()

            if response.get("items"):
                item = response["items"][0]
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                thumbnails = snippet.get("thumbnails", {})
                best_thumb = (
                    thumbnails.get("maxres", {}).get("url") or
                    thumbnails.get("high", {}).get("url") or
                    thumbnails.get("medium", {}).get("url") or
                    default_thumbnail
                )

                return {
                    "video_id": clean_id,
                    "title": snippet.get("title", f"YouTube Video ({clean_id})"),
                    "description": snippet.get("description", ""),
                    "channel": snippet.get("channelTitle", "Unknown Channel"),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail_url": best_thumb,
                    "view_count": stats.get("viewCount", "N/A"),
                    "like_count": stats.get("likeCount", "N/A"),
                    "url": f"https://www.youtube.com/watch?v={clean_id}"
                }
        except Exception:
            pass  # Fall through to oEmbed

    # 2. Resilient oEmbed fallback (zero keys required)
    try:
        req = urllib.request.Request(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={clean_id}&format=json",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return {
                "video_id": clean_id,
                "title": data.get("title", f"YouTube Video ({clean_id})"),
                "description": "",
                "channel": data.get("author_name", "YouTube Creator"),
                "published_at": "",
                "thumbnail_url": data.get("thumbnail_url", default_thumbnail),
                "view_count": "N/A",
                "like_count": "N/A",
                "url": f"https://www.youtube.com/watch?v={clean_id}"
            }
    except Exception:
        # 3. Minimal fallback
        return {
            "video_id": clean_id,
            "title": f"YouTube Video ({clean_id})",
            "description": "",
            "channel": "YouTube",
            "published_at": "",
            "thumbnail_url": fallback_thumbnail,
            "view_count": "N/A",
            "like_count": "N/A",
            "url": f"https://www.youtube.com/watch?v={clean_id}"
        }

def search_youtube(query: str, api_key: str = None, max_results: int = 5) -> list:
    """Search YouTube for videos."""
    youtube = create_youtube_client(api_key)
    if not youtube:
        return []
    
    try:
        response = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results
        ).execute()

        results = []
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            vid = item.get("id", {}).get("videoId")
            if vid:
                results.append({
                    "video_id": vid,
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                    "url": f"https://www.youtube.com/watch?v={vid}"
                })
        return results
    except Exception:
        return []