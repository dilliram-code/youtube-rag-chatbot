import os 

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY=os.getenv("YOUTUBE_API_KEY")

if not OPENAI_API_KEY:
  raise ValueError("OPENAI_API_KEY is missing.")

if not YOUTUBE_API_KEY:
  raise ValueError("YOUTUBE_API_KEY is missing.")
