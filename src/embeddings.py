import logging
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
import chromadb.utils.embedding_functions as chromadb_ef
from src.config import get_openai_api_key, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class LocalONNXEmbeddings(Embeddings):
    """
    LangChain compatible wrapper for Chroma's default fast ONNX embedding function
    (all-MiniLM-L6-v2). Completely free, runs locally, no API keys needed.
    """
    def __init__(self):
        self._ef = chromadb_ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0]

def get_embedding_model(
    provider: str = "openai",
    api_key: str = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> Embeddings:
    """
    Instantiate and return an Embeddings provider.
    Falls back gracefully to LocalONNXEmbeddings if OpenAI key is missing or invalid.
    """
    if provider == "local":
        return LocalONNXEmbeddings()

    # Try OpenAI
    key = api_key or get_openai_api_key()
    if key:
        try:
            return OpenAIEmbeddings(
                openai_api_key=key,
                model=model_name
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI Embeddings, falling back to local ONNX: {e}")
            return LocalONNXEmbeddings()
    
    # Fallback to local
    return LocalONNXEmbeddings()

def validate_openai_api_key(api_key: str) -> tuple[bool, str]:
    """Test if an OpenAI API key is active and valid."""
    if not api_key or not api_key.strip():
        return False, "API key is empty."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        client.models.list()
        return True, "OpenAI API key is valid!"
    except Exception as e:
        err_msg = str(e)
        if "expired" in err_msg.lower():
            return False, "API key has expired. Please provide a new key."
        if "invalid" in err_msg.lower() or "401" in err_msg:
            return False, "Invalid API key credentials."
        return False, f"OpenAI verification failed: {err_msg[:80]}..."
