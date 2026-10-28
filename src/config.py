import os
from dotenv import load_dotenv

# Load existing environment variables
load_dotenv()

def get_openai_api_key():
    """Retrieve OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY", "").strip()

def get_youtube_api_key():
    """Retrieve YouTube API key from environment."""
    return os.getenv("YOUTUBE_API_KEY", "").strip()

def set_openai_api_key(api_key: str):
    """Set OpenAI API key dynamically in environment."""
    os.environ["OPENAI_API_KEY"] = api_key.strip()

def set_youtube_api_key(api_key: str):
    """Set YouTube API key dynamically in environment."""
    os.environ["YOUTUBE_API_KEY"] = api_key.strip()

# Default constants
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
