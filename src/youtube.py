from googleapiclient.discovery import build

from src.config import YOUTUBE_API_KEY

def create_youtube_client():
  """
  Create a YouTube Data API client
  """
  
  youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
  )
  