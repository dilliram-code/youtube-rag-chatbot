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
  
  return youtube

def extract_video_id(url):
  """
  Extract the video ID from a YouTube URL
  """
  
  if "v=" in url:
    return url.split("v=")[1].split("&")[0]
  
  if "youtu.be/" in url:
    return url.split("youtu.be/")[1].split("?")[0]
  
  raise ValueError("Invalid YouTube URL.")

def get_video_info(video_id):
  """
  Get basic information about a YouTube video.
  """
  youtube = create_youtube_client()
  response = youtube.videos().list(
    part="snippet",
    id=video_id
  ).execute()
  
  
  