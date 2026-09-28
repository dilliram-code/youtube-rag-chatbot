from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
  """
  Get the transcript of the YouTube video.
  """
  api = YouTubeTranscriptApi()
  transcript = api.fetch(video_id)
  text = " ".join(
    item.text for item in transcript
  )
  
  return text