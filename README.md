# 🎬 TubeIntellect — YouTube RAG Chatbot

An elegant, modern, and glassy AI-powered chatbot for exploring and asking questions about any YouTube video. Powered by **LangChain**, **ChromaDB**, **OpenAI**, and **Streamlit**.

---

## ✨ Features

- 💎 **Ultra-Modern Glassmorphic UI**: High-fidelity frosted glass cards, dynamic gradients, subtle shadows, and responsive layout.
- 🌓 **Dual Themes (Dark & Light)**: Seamlessly toggle between Cyber Dark (Deep Obsidian & Neon Accents) and Crystal Frost Light modes.
- ⚡ **Timestamp-Aware RAG**: Preserves start and end seconds for every chunk so AI answers include clickable `[MM:SS]` jump links directly to that moment in the video.
- 🧠 **Dual Embedding Options**:
  - **Local ONNX (`all-MiniLM-L6-v2`)**: 100% free offline embedding with zero API quota usage.
  - **OpenAI (`text-embedding-3-small`)**: Industry-standard high-dimensional embeddings.
- 📊 **Video Showcase Card**: Displays video thumbnail, channel name, vector chunk count, language badges, and auto-generated caption detection.
- 🚀 **1-Click Actions**:
  - **⚡ 1-Click Executive Summary**: Get high-level takeaways, core insights, and target audience.
  - **📌 5 Core Highlights**: Pinpoint key moments with timestamps.
  - **📜 Full Transcript Browser**: Interactive keyword search across all video segments with instant jump links.
- 🛡️ **Zero-Crash Resilience**: Automatic oEmbed fallback for metadata even without a YouTube API key, and in-app API key verification.

---

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Configure Environment (`.env`)
Make sure your `.env` contains:
```env
OPENAI_API_KEY=your_openai_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here
```
*(You can also update or verify your OpenAI key directly in the app's sidebar).*

### 3. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Architecture

```
youtube-rag-chatbot/
├── .streamlit/
│   └── config.toml          # Streamlit theme & headless settings
├── src/
│   ├── config.py            # Environment configuration & defaults
│   ├── youtube.py           # Robust URL parsing & metadata extraction (oEmbed + v3 API)
│   ├── transcript.py        # Timestamp-aware transcript extraction (youtube-transcript-api)
│   ├── embeddings.py        # OpenAI + Local ONNX embeddings
│   ├── vectorstore.py       # Semantic timestamped chunking & ChromaDB management
│   └── rag.py               # RAG prompt pipeline & streaming answers
├── app.py                   # Modern Streamlit UI with Glassmorphism & Theme switcher
└── requirements.txt         # Project dependencies
```
