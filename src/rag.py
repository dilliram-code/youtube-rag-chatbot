import logging
from typing import List, Dict, Any, Generator, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from src.config import get_openai_api_key, DEFAULT_CHAT_MODEL, DEFAULT_TOP_K
from src.vectorstore import search_similar_chunks

logger = logging.getLogger(__name__)

SYSTEM_RAG_PROMPT = """You are TubeIntellect, a premium and helpful AI assistant specialized in analyzing YouTube videos.
Your goal is to answer questions thoroughly, accurately, and engagingly based on the provided video transcript excerpts.

Guidelines:
1. Ground your answer in the provided video context. Do NOT hallucinate facts not present in the video.
2. Whenever citing a specific topic, insight, or quote, reference the timestamp in markdown link format using the provided URL, e.g., `[MM:SS](URL)` or `[HH:MM:SS](URL)`.
3. If the transcript does not contain the answer, politely state: "Based on the video transcript, this topic is not discussed in the video." Then, offer a brief summary of what the video does focus on if relevant.
4. Structure your response with clean markdown headings, bullet points, or bold highlights for maximum readability.
"""

def get_chat_model(
    model_name: str = DEFAULT_CHAT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    streaming: bool = False
) -> Optional[ChatOpenAI]:
    """Instantiate ChatOpenAI client if API key is provided."""
    key = api_key or get_openai_api_key()
    if not key:
        return None
    try:
        return ChatOpenAI(
            model=model_name,
            openai_api_key=key,
            temperature=temperature,
            streaming=streaming
        )
    except Exception as e:
        logger.error(f"Error initializing ChatOpenAI: {e}")
        return None

def format_context_for_prompt(docs: List[Any]) -> str:
    """Format retrieved document chunks with timestamps into a prompt context block."""
    context_lines = []
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        time_range = meta.get("time_range", meta.get("start_time", "N/A"))
        url = meta.get("timestamp_url", "")
        chunk_header = f"--- Excerpt {idx} (Time: {time_range} | Link: {url}) ---"
        context_lines.append(f"{chunk_header}\n{doc.page_content.strip()}")
    return "\n\n".join(context_lines)

def build_messages(
    query: str,
    context_str: str,
    video_title: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None
) -> list:
    """Build LangChain message list with system prompt, context, history, and query."""
    messages = [
        SystemMessage(content=SYSTEM_RAG_PROMPT)
    ]

    # Context Message
    video_intro = f"Current Video Title: '{video_title}'\n\n" if video_title else ""
    context_content = (
        f"{video_intro}VIDEO TRANSCRIPT EXCERPTS:\n{context_str}\n\n"
        f"Answer the user's question using the excerpts above."
    )
    messages.append(SystemMessage(content=context_content))

    # Add previous chat history (up to last 6 messages)
    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # Current query
    messages.append(HumanMessage(content=query))
    return messages

def query_rag_stream(
    query: str,
    vectorstore: Chroma,
    video_title: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = DEFAULT_CHAT_MODEL,
    api_key: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 0.3
) -> tuple[Generator[str, None, None], List[Any]]:
    """
    Search vectorstore for context, build messages, and stream LLM response tokens.
    Returns (generator_of_text_tokens, list_of_retrieved_source_docs).
    """
    # 1. Retrieve top chunks
    retrieved_docs = search_similar_chunks(vectorstore, query, k=top_k)
    context_str = format_context_for_prompt(retrieved_docs)

    # 2. Check LLM availability
    llm = get_chat_model(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        streaming=True
    )

    if not llm:
        def fallback_generator():
            yield (
                "⚠️ **OpenAI API Key Required or Expired**\n\n"
                "To generate natural AI responses, please enter an active **OpenAI API Key** in the sidebar. "
                "However, the vector similarity search successfully located the most relevant sections of the video below!\n\n"
                "**Relevant Video Segments Found:**\n"
            )
            for i, doc in enumerate(retrieved_docs, 1):
                m = doc.metadata or {}
                t = m.get("time_range", "N/A")
                u = m.get("timestamp_url", "#")
                yield f"- **[{t}]({u})**: {doc.page_content[:180]}...\n"

        return fallback_generator(), retrieved_docs

    # 3. Stream from OpenAI
    messages = build_messages(query, context_str, video_title, chat_history)

    def stream_tokens():
        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            err_str = str(e)
            if "expired" in err_str.lower() or "401" in err_str:
                yield (
                    "\n\n❌ **Error**: Your OpenAI API key appears to be invalid or expired. "
                    "Please update it in the sidebar settings."
                )
            else:
                yield f"\n\n❌ **Error generating response**: {err_str}"

    return stream_tokens(), retrieved_docs

def generate_video_summary_stream(
    full_text: str,
    video_title: str,
    model_name: str = DEFAULT_CHAT_MODEL,
    api_key: Optional[str] = None
) -> Generator[str, None, None]:
    """Generate an executive summary with Key Takeaways and Core Insights."""
    llm = get_chat_model(model_name=model_name, api_key=api_key, temperature=0.2, streaming=True)
    if not llm:
        def err_gen():
            yield "Please provide a valid OpenAI API key in the sidebar to generate video summaries."
        return err_gen()

    # Limit text to ~15,000 chars to avoid context overflow
    truncated_text = full_text[:16000] if len(full_text) > 16000 else full_text

    prompt = (
        f"You are an expert video summarizer. Summarize the following YouTube video.\n\n"
        f"Video Title: {video_title}\n\n"
        f"Transcript:\n{truncated_text}\n\n"
        f"Please provide:\n"
        f"1. **Core Summary**: 2-3 sentence high-level overview.\n"
        f"2. **Key Takeaways**: 4-6 bullet points of the most valuable insights.\n"
        f"3. **Target Audience / Who Should Watch**: 1-2 sentences.\n"
    )

    def stream():
        try:
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"Error generating summary: {e}"

    return stream()
