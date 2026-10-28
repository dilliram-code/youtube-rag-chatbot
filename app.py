import os
import streamlit as st
from src.config import (
    get_openai_api_key,
    set_openai_api_key,
    get_youtube_api_key,
    DEFAULT_CHAT_MODEL,
    DEFAULT_TOP_K
)
from src.youtube import extract_video_id, get_video_info
from src.transcript import get_transcript_data
from src.embeddings import get_embedding_model, validate_openai_api_key
from src.vectorstore import build_timestamped_chunks, create_vectorstore
from src.rag import query_rag_stream, generate_video_summary_stream

# Page configuration
st.set_page_config(
    page_title="TubeIntellect — Modern YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "video_data" not in st.session_state:
    st.session_state["video_data"] = None

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "custom_openai_key" not in st.session_state:
    st.session_state["custom_openai_key"] = get_openai_api_key()

if "embedding_provider" not in st.session_state:
    st.session_state["embedding_provider"] = "local"  # Default to local ONNX for 100% reliable zero-key startup

# Example videos for 1-click test
SAMPLE_VIDEOS = {
    "Rick Astley (Music)": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "Steve Jobs 2005 Speech": "https://www.youtube.com/watch?v=UF8uR6Z6KLc",
    "Veritasium (Physics)": "https://www.youtube.com/watch?v=bHIhgxav9LY",
}

# ---------------------------------------------------------
# Dynamic Glassy CSS System (Dark & Light Theme)
# ---------------------------------------------------------
def apply_custom_styles(theme: str):
    if theme == "dark":
        css_vars = """
        --bg-gradient: radial-gradient(circle at 15% 15%, #0f172a 0%, #060913 100%);
        --glass-bg: rgba(22, 30, 49, 0.65);
        --glass-card: rgba(30, 41, 69, 0.55);
        --glass-card-hover: rgba(38, 52, 88, 0.7);
        --glass-border: rgba(255, 255, 255, 0.12);
        --glass-border-glow: rgba(99, 102, 241, 0.45);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-primary: #6366f1;
        --accent-secondary: #a855f7;
        --accent-cyan: #06b6d4;
        --brand-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        --chat-user-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.85) 0%, rgba(139, 92, 246, 0.85) 100%);
        --chat-bot-bg: rgba(23, 31, 51, 0.75);
        --shadow-elevation: 0 16px 36px -10px rgba(0, 0, 0, 0.6), 0 0 1px 1px rgba(255, 255, 255, 0.08) inset;
        --glow-shadow: 0 0 25px rgba(99, 102, 241, 0.25);
        --sidebar-bg: rgba(13, 18, 30, 0.85);
        --tag-bg: rgba(99, 102, 241, 0.18);
        """
    else:  # Light Mode
        css_vars = """
        --bg-gradient: radial-gradient(circle at 10% 10%, #f0f4ff 0%, #f8fafc 100%);
        --glass-bg: rgba(255, 255, 255, 0.75);
        --glass-card: rgba(255, 255, 255, 0.82);
        --glass-card-hover: rgba(255, 255, 255, 0.95);
        --glass-border: rgba(99, 102, 241, 0.18);
        --glass-border-glow: rgba(99, 102, 241, 0.35);
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --accent-primary: #4f46e5;
        --accent-secondary: #9333ea;
        --accent-cyan: #0284c7;
        --brand-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        --chat-user-bg: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        --chat-bot-bg: rgba(255, 255, 255, 0.85);
        --shadow-elevation: 0 16px 36px -10px rgba(99, 102, 241, 0.15), 0 0 1px 1px rgba(99, 102, 241, 0.1) inset;
        --glow-shadow: 0 0 25px rgba(99, 102, 241, 0.18);
        --sidebar-bg: rgba(248, 250, 252, 0.85);
        --tag-bg: rgba(99, 102, 241, 0.12);
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

        :root {{
            {css_vars}
        }}

        /* App Base Styling */
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            background: var(--bg-gradient) !important;
            color: var(--text-primary) !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* Custom Glass Sidebar */
        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg) !important;
            backdrop-filter: blur(28px) saturate(180%) !important;
            border-right: 1px solid var(--glass-border) !important;
            box-shadow: 4px 0 24px rgba(0,0,0,0.1) !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: var(--glass-border) !important;
        }}

        /* Main Glass Cards */
        .glass-panel {{
            background: var(--glass-card);
            backdrop-filter: blur(20px) saturate(170%);
            -webkit-backdrop-filter: blur(20px) saturate(170%);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 24px;
            box-shadow: var(--shadow-elevation);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 24px;
        }}

        .glass-panel:hover {{
            border-color: var(--glass-border-glow);
            box-shadow: var(--shadow-elevation), var(--glow-shadow);
        }}

        /* Header Hero */
        .hero-title {{
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 2.6rem;
            background: var(--brand-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding-bottom: 6px;
            letter-spacing: -0.02em;
        }}

        .hero-subtitle {{
            color: var(--text-secondary);
            font-size: 1.05rem;
            font-weight: 400;
            margin-top: 4px;
            margin-bottom: 20px;
        }}

        /* Floating Pill Badges */
        .glass-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 14px;
            border-radius: 9999px;
            background: var(--tag-bg);
            border: 1px solid var(--glass-border);
            color: var(--accent-primary);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 10px #10b981;
            animation: pulse-glow 2s infinite;
        }}

        @keyframes pulse-glow {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.3); opacity: 0.6; }}
        }}

        /* Video Showcase Card */
        .video-card {{
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            align-items: center;
            background: var(--glass-card);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: var(--shadow-elevation);
            backdrop-filter: blur(20px);
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }}

        .video-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--brand-gradient);
        }}

        .video-thumb {{
            width: 240px;
            max-width: 100%;
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
            border: 1px solid var(--glass-border);
            transition: transform 0.3s ease;
        }}

        .video-thumb:hover {{
            transform: scale(1.02);
        }}

        .video-info {{
            flex: 1;
            min-width: 260px;
        }}

        .video-info h3 {{
            margin: 0 0 8px 0;
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            color: var(--text-primary);
            font-weight: 700;
            line-height: 1.3;
        }}

        .video-meta-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }}

        .meta-pill {{
            background: var(--tag-bg);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 4px 10px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        /* Chat Bubbles */
        .chat-bubble-user {{
            background: var(--chat-user-bg);
            color: #ffffff !important;
            padding: 16px 20px;
            border-radius: 20px 20px 4px 20px;
            margin-left: auto;
            max-width: 85%;
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
            font-size: 0.98rem;
            line-height: 1.5;
            word-wrap: break-word;
        }}

        .chat-bubble-bot {{
            background: var(--chat-bot-bg);
            color: var(--text-primary) !important;
            padding: 20px 24px;
            border-radius: 20px 20px 20px 4px;
            border: 1px solid var(--glass-border);
            margin-right: auto;
            max-width: 90%;
            box-shadow: var(--shadow-elevation);
            font-size: 0.98rem;
            line-height: 1.6;
            backdrop-filter: blur(20px);
            word-wrap: break-word;
        }}

        /* Timestamp Citations Pill */
        .ts-pill {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(99, 102, 241, 0.15);
            color: var(--accent-primary) !important;
            border: 1px solid var(--glass-border);
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            text-decoration: none !important;
            transition: all 0.2s ease;
        }}

        .ts-pill:hover {{
            background: var(--accent-primary);
            color: #ffffff !important;
            transform: translateY(-1px);
        }}

        /* Streamlit Input Styling */
        div[data-baseweb="input"] {{
            background: var(--glass-card) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(12px) !important;
            color: var(--text-primary) !important;
            transition: all 0.2s ease !important;
        }}

        div[data-baseweb="input"]:focus-within {{
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
        }}

        /* Custom Button Styling */
        .stButton button {{
            background: var(--glass-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 12px !important;
            padding: 8px 18px !important;
            font-weight: 600 !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        }}

        .stButton button:hover {{
            background: var(--brand-gradient) !important;
            color: #ffffff !important;
            border-color: transparent !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
        }}

        /* Primary Action Buttons */
        div[data-testid="stButton"] button[kind="primary"] {{
            background: var(--brand-gradient) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
        }}

        /* Responsive utilities */
        @media (max-width: 768px) {{
            .hero-title {{
                font-size: 2rem;
            }}
            .video-card {{
                flex-direction: column;
                align-items: stretch;
            }}
            .video-thumb {{
                width: 100%;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply dynamic CSS styles based on active theme
apply_custom_styles(st.session_state["theme"])

# ---------------------------------------------------------
# Sidebar: Settings, Model Config & API Management
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.caption("Customize models, retrieval, and API credentials.")

    # 1. Theme Switcher in Sidebar
    theme_col1, theme_col2 = st.columns([1, 1])
    with theme_col1:
        if st.button("🌙 Dark Mode", use_container_width=True):
            st.session_state["theme"] = "dark"
            st.rerun()
    with theme_col2:
        if st.button("☀️ Light Mode", use_container_width=True):
            st.session_state["theme"] = "light"
            st.rerun()

    st.markdown("---")

    # 2. OpenAI API Key Manager
    st.markdown("#### 🔑 OpenAI API Key")
    current_key = st.session_state.get("custom_openai_key", "")
    key_input = st.text_input(
        "Enter OpenAI Key",
        value=current_key,
        type="password",
        placeholder="sk-proj-...",
        help="Required for generating natural language answers. If expired, enter your active key here."
    )

    if key_input != current_key:
        st.session_state["custom_openai_key"] = key_input.strip()
        set_openai_api_key(key_input.strip())
        st.rerun()

    # Validation check button
    if st.button("🔍 Verify Key", use_container_width=True):
        if not key_input:
            st.warning("Please paste an OpenAI API key.")
        else:
            with st.spinner("Checking key validity..."):
                is_valid, msg = validate_openai_api_key(key_input)
                if is_valid:
                    st.success("✅ " + msg)
                else:
                    st.error("❌ " + msg)

    st.markdown("---")

    # 3. Model & Embedding Configuration
    st.markdown("#### 🧠 Model Selection")
    selected_chat_model = st.selectbox(
        "Chat LLM",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini is ultra-fast, intelligent, and cost-efficient."
    )

    embedding_choice = st.radio(
        "Embeddings Provider",
        ["Local ONNX (all-MiniLM-L6-v2) ⚡ Free", "OpenAI (text-embedding-3-small)"],
        index=0 if st.session_state["embedding_provider"] == "local" else 1,
        help="Local ONNX runs completely offline inside Chroma with 0 API tokens and no cost!"
    )
    provider_str = "local" if "Local" in embedding_choice else "openai"
    st.session_state["embedding_provider"] = provider_str

    st.markdown("---")

    # 4. RAG Hyperparameters
    st.markdown("#### 🎛️ RAG Retrieval Parameters")
    retrieval_k = st.slider("Top Chunks (k)", min_value=2, max_value=8, value=DEFAULT_TOP_K)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")

    # 5. Session Actions
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if st.button("🔄 Reset Current Video", use_container_width=True):
        st.session_state["video_data"] = None
        st.session_state["vectorstore"] = None
        st.session_state["messages"] = []
        st.rerun()

    st.caption("Powered by LangChain, ChromaDB & Streamlit")

# ---------------------------------------------------------
# Main Page: Header & Video Ingestion
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown('<div class="hero-title">TubeIntellect AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Next-generation RAG assistant for YouTube. Ask questions, explore timestamps, and distill any video in seconds.</div>',
        unsafe_allow_html=True
    )
with header_col2:
    st.write("")
    current_mode_label = "🌙 Dark Mode" if st.session_state["theme"] == "dark" else "☀️ Light Mode"
    new_theme = "light" if st.session_state["theme"] == "dark" else "dark"
    if st.button(f"{'☀️ Light' if st.session_state['theme'] == 'dark' else '🌙 Dark'}", use_container_width=True):
        st.session_state["theme"] = new_theme
        st.rerun()

# ---------------------------------------------------------
# Video Link Ingestion Section
# ---------------------------------------------------------
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

# Sample Pill Buttons for instant demonstration
st.markdown("<small style='color:var(--text-secondary); font-weight:600;'>TRY WITH QUICK DEMO VIDEOS:</small>", unsafe_allow_html=True)
pill_cols = st.columns(len(SAMPLE_VIDEOS))
for idx, (label, sample_url) in enumerate(SAMPLE_VIDEOS.items()):
    if pill_cols[idx].button(f"🎬 {label}", key=f"pill_{idx}", use_container_width=True):
        st.session_state["video_url_input"] = sample_url
        st.rerun()

st.write("")
input_col, btn_col = st.columns([5, 1.3])
default_url = st.session_state.get("video_url_input", "")
with input_col:
    video_url = st.text_input(
        "YouTube Video URL",
        value=default_url,
        placeholder="Paste YouTube link here: https://www.youtube.com/watch?v=... or https://youtu.be/...",
        label_visibility="collapsed"
    )

with btn_col:
    load_button = st.button("🚀 Load Video", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Video Ingestion & Indexing Pipeline
# ---------------------------------------------------------
if load_button and video_url.strip():
    with st.spinner("Extracting video ID and fetching metadata..."):
        try:
            video_id = extract_video_id(video_url)
            info = get_video_info(video_id)
            st.session_state["video_info"] = info

            # Fetch transcript data with timestamps
            with st.spinner("Retrieving video transcript and captions..."):
                transcript_data = get_transcript_data(video_id)
                st.session_state["transcript_data"] = transcript_data

            # Chunking and Vectorstore Indexing
            with st.spinner("Building semantic vector store with Chroma..."):
                emb_model = get_embedding_model(
                    provider=st.session_state["embedding_provider"],
                    api_key=st.session_state.get("custom_openai_key")
                )
                chunks = build_timestamped_chunks(
                    transcript_data["snippets"],
                    video_id=video_id
                )
                vectorstore = create_vectorstore(
                    documents=chunks,
                    embedding_model=emb_model,
                    video_id=video_id,
                    persist_directory=None  # Fast in-memory session store
                )
                st.session_state["vectorstore"] = vectorstore
                st.session_state["chunk_count"] = len(chunks)
                st.session_state["video_data"] = {
                    "video_id": video_id,
                    "info": info,
                    "chunks": chunks,
                    "total_snippets": len(transcript_data["snippets"]),
                    "language": transcript_data["language"],
                    "is_generated": transcript_data["is_generated"]
                }
                # Reset conversation for new video
                st.session_state["messages"] = [
                    {
                        "role": "assistant",
                        "content": f"👋 Hello! I have processed **{info['title']}**. You can ask me any question about this video or use the quick summary buttons below!"
                    }
                ]
            st.success("✅ Video loaded and indexed successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")

# ---------------------------------------------------------
# Video Showcase Card (If a video is active)
# ---------------------------------------------------------
if st.session_state.get("video_data"):
    v_data = st.session_state["video_data"]
    info = v_data["info"]
    video_id = v_data["video_id"]
    chunks_count = st.session_state.get("chunk_count", len(v_data.get("chunks", [])))

    st.markdown(
        f"""
        <div class="video-card">
            <a href="https://www.youtube.com/watch?v={video_id}" target="_blank">
                <img src="{info['thumbnail_url']}" class="video-thumb" alt="{info['title']}" />
            </a>
            <div class="video-info">
                <span class="glass-badge"><span class="status-dot"></span> Ready for Q&A</span>
                <h3>{info['title']}</h3>
                <p style="color:var(--text-secondary); margin:0; font-size:0.95rem;">👤 {info['channel']}</p>
                <div class="video-meta-tags">
                    <span class="meta-pill">📊 {chunks_count} Vector Chunks</span>
                    <span class="meta-pill">🌐 {v_data['language']} {'(Auto-captions)' if v_data['is_generated'] else '(Manual CC)'}</span>
                    <span class="meta-pill">⚡ Embedding: {st.session_state['embedding_provider'].upper()}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick Action Buttons
    action_cols = st.columns([1, 1, 1])
    with action_cols[0]:
        if st.button("⚡ 1-Click Executive Summary", use_container_width=True):
            user_msg = "Please provide an executive summary of this video with key takeaways and target audience."
            st.session_state["messages"].append({"role": "user", "content": user_msg})
            st.session_state["trigger_query"] = user_msg
            st.rerun()

    with action_cols[1]:
        if st.button("📌 5 Core Highlights & Timestamps", use_container_width=True):
            user_msg = "What are the 5 most important highlights or moments in this video? Include timestamps for each."
            st.session_state["messages"].append({"role": "user", "content": user_msg})
            st.session_state["trigger_query"] = user_msg
            st.rerun()

    with action_cols[2]:
        with st.popover("📜 Browse Full Transcript", use_container_width=True):
            st.markdown(f"**Full Transcript for: {info['title']}**")
            snippets = st.session_state.get("transcript_data", {}).get("snippets", [])
            if snippets:
                search_term = st.text_input("Filter transcript by keyword", "")
                filtered_snippets = [s for s in snippets if search_term.lower() in s['text'].lower()] if search_term else snippets
                st.caption(f"Showing {len(filtered_snippets)} segments")
                transcript_container = st.container(height=380)
                with transcript_container:
                    for s in filtered_snippets:
                        st.markdown(
                            f"<small><a href='{s['timestamp_url']}' target='_blank' class='ts-pill'>▶ {s['time']}</a> {s['text']}</small><br>",
                            unsafe_allow_html=True
                        )
            else:
                st.info("Transcript text is available above.")

# ---------------------------------------------------------
# Chat Conversation Area
# ---------------------------------------------------------
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 💬 Video Q&A Assistant")

# Display previous conversation messages
for msg in st.session_state["messages"]:
    role = msg.get("role")
    content = msg.get("content", "")
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(
            f'<div class="chat-bubble-user">{content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-bubble-bot">{content}</div>',
            unsafe_allow_html=True
        )

        # If sources exist, display in expandable citation drawer
        if sources:
            with st.expander("📍 Citations & Video Moments Referenced", expanded=False):
                for idx, src in enumerate(sources, 1):
                    meta = getattr(src, "metadata", {}) or {}
                    time_range = meta.get("time_range", meta.get("start_time", "N/A"))
                    url = meta.get("timestamp_url", "#")
                    text_snippet = getattr(src, "page_content", "")[:240]
                    st.markdown(
                        f"""
                        <div style="margin-bottom:10px; padding:10px; background:var(--glass-bg); border-radius:10px; border:1px solid var(--glass-border);">
                            <a href="{url}" target="_blank" class="ts-pill">▶ Jump to {time_range}</a>
                            <p style="margin:6px 0 0 0; font-size:0.88rem; color:var(--text-secondary);">{text_snippet}...</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

st.write("")

# ---------------------------------------------------------
# Handle Streaming Assistant Turn (if triggered)
# ---------------------------------------------------------
trigger_query = st.session_state.pop("trigger_query", None)

if trigger_query and st.session_state.get("vectorstore"):
    v_info = st.session_state.get("video_data", {}).get("info", {})
    v_title = v_info.get("title", "")

    # Create assistant placeholder
    bot_container = st.empty()
    token_stream, source_docs = query_rag_stream(
        query=trigger_query,
        vectorstore=st.session_state["vectorstore"],
        video_title=v_title,
        chat_history=st.session_state["messages"][:-1],
        model_name=selected_chat_model,
        api_key=st.session_state.get("custom_openai_key"),
        top_k=retrieval_k,
        temperature=temperature
    )

    full_response = ""
    for token in token_stream:
        full_response += token
        bot_container.markdown(
            f'<div class="chat-bubble-bot">{full_response}▌</div>',
            unsafe_allow_html=True
        )

    bot_container.markdown(
        f'<div class="chat-bubble-bot">{full_response}</div>',
        unsafe_allow_html=True
    )

    # Save to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_response,
        "sources": source_docs
    })
    st.rerun()

# ---------------------------------------------------------
# Interactive Prompt Input Box
# ---------------------------------------------------------
prompt = st.chat_input(
    placeholder="Ask anything about the video..." if st.session_state.get("video_data") else "Please load a YouTube video first above to start asking questions!"
)

if prompt:
    if not st.session_state.get("vectorstore"):
        st.warning("⚠️ Please paste a YouTube link and click 'Load Video' above first!")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["trigger_query"] = prompt
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
