import re
import os
import shutil
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from src.config import CHROMA_PERSIST_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K
from src.transcript import format_timestamp

def sanitize_collection_name(video_id: str) -> str:
    """Ensure Chroma-compliant collection name (3-63 chars, alphanumeric with - or _)."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", video_id)
    # Chroma requires collection names between 3 and 63 characters
    name = f"yt_{clean}"
    if len(name) < 3:
        name = name + "_col"
    return name[:63]

def build_timestamped_chunks(
    snippets: List[Dict[str, Any]],
    video_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Group chronological transcript snippets into semantic chunks while
    preserving start/end timestamps and clickable video time references.
    """
    if not snippets:
        return []

    documents = []
    current_texts = []
    current_char_count = 0
    chunk_start_time = snippets[0]["start"]
    chunk_start_idx = 0

    i = 0
    while i < len(snippets):
        snippet = snippets[i]
        text = snippet["text"]
        current_texts.append(text)
        current_char_count += len(text) + 1

        # Check if current chunk exceeds the desired size or reached the end
        if current_char_count >= chunk_size or i == len(snippets) - 1:
            chunk_content = " ".join(current_texts)
            chunk_end_time = snippet["start"] + snippet["duration"]
            start_str = format_timestamp(chunk_start_time)
            end_str = format_timestamp(chunk_end_time)

            doc = Document(
                page_content=chunk_content,
                metadata={
                    "video_id": video_id,
                    "start_seconds": chunk_start_time,
                    "end_seconds": chunk_end_time,
                    "start_time": start_str,
                    "end_time": end_str,
                    "time_range": f"{start_str} - {end_str}",
                    "timestamp_url": f"https://www.youtube.com/watch?v={video_id}&t={int(chunk_start_time)}s"
                }
            )
            documents.append(doc)

            # Calculate overlap by stepping back snippets
            if i == len(snippets) - 1:
                break

            overlap_chars = 0
            step_back = 0
            for j in range(i, chunk_start_idx, -1):
                overlap_chars += len(snippets[j]["text"]) + 1
                if overlap_chars >= chunk_overlap:
                    step_back = (i - j)
                    break
                step_back = (i - j)

            # Move index forward with overlap
            new_idx = max(chunk_start_idx + 1, i - step_back + 1)
            i = new_idx
            current_texts = []
            current_char_count = 0
            if i < len(snippets):
                chunk_start_time = snippets[i]["start"]
                chunk_start_idx = i
        else:
            i += 1

    return documents

def create_vectorstore(
    documents: List[Document],
    embedding_model: Embeddings,
    video_id: str,
    persist_directory: Optional[str] = CHROMA_PERSIST_DIR
) -> Chroma:
    """
    Index documents in Chroma vector database.
    """
    collection_name = sanitize_collection_name(video_id)
    
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        # Using persistent client storage
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    else:
        # In-memory vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name
        )

    return vectorstore

def get_retriever(vectorstore: Chroma, k: int = DEFAULT_TOP_K):
    """Return a retriever with search_kwargs k."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def search_similar_chunks(vectorstore: Chroma, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
    """Retrieve top k relevant chunks for a user query."""
    return vectorstore.similarity_search(query, k=k)

def clear_vectorstore_cache(persist_directory: str = CHROMA_PERSIST_DIR):
    """Clear cached ChromaDB persistent files."""
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            return True
        except Exception:
            return False
    return True
