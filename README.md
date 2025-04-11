# PyChat — Conversational AI Desktop Client + Retrieval‑Augmented Generation



> **Branch:** `rag2` — experimental RAG integration (LightRAG‑style vector search + PDF ingestion)

PyChat is a **cross‑platform PyQt5 desktop application** that unifies the power of multiple large‑language‑model providers **(OpenAI, Anthropic Claude, Google Gemini, Ollama/local models)** into a single, elegant chat workspace.\
The `rag2` branch layers in a lightweight Retrieval‑Augmented Generation (RAG) pipeline so you can **query your own documents** with cosine‑similarity vector search and let any LLM ground its answers in your data.

---

## ✨ Key Features

- OpenAI ChatGPT
- Anthropic Claude 3 (Opus/Sonnet/Haiku)
- Google Gemini
- Ollama local models (Llama‑3, DeepSeek, etc.) | | **RAG Toolkit** | \* `simple_rag_manager.py` – TF‑IDF or Sentence‑BERT embeddings (switchable)
- SQLite‑backed vector store
- PDF/TXT/MD ingestion with automatic chunking
- Cosine‑similarity retrieval piped straight into the chat prompt | | **Rich UI** | \* Syntax‑highlighted code blocks
- Real‑time streaming
- Search across threads with jump‑to‑message
- Dark/light themes | | **Productivity** | \* Pre‑prompt templates
- AI‑to‑AI conversations
- Message export & copy‑code buttons | | **Privacy First** | \* All chat history & embeddings stored **locally** (SQLite)
- API keys saved via Qt settings – never transmitted or synced |

---

## 📦 Installation

````bash
# 1 – Clone the repo & checkout rag2
$ git clone https://github.com/Magnetron85/PyChat.git
$ cd PyChat && git checkout rag2

# 2 – Create a virtual environment (recommended)
$ python -m venv .venv && source .venv/Scripts/activate  # Windows PowerShell

# 3 – Install dependencies
$ pip install PyQt5 requests qtconsole fuzzywuzzy google-genai scikit-learn sentence-transformers numpy pandas requests PyMuPDF tqdm python-dotenv google-generativeai rich ruff mypy

---

## 🔑 Provider Setup

Set your API keys as environment variables **or** enter them once in *Settings → Providers*:

```powershell
# PowerShell example
$env:OPENAI_API_KEY = "sk‑…"
$env:ANTHROPIC_API_KEY = "claude‑sk‑…"
$env:GEMINI_API_KEY = "…"
````

Ollama just needs the daemon running locally: `ollama serve`.

---

## 🚀 Running PyChat

```bash
python pychat.py
```

The main window shows **Threads** (left), **Chat** (center), and **Search** (right).  Start a new thread, pick a provider/model, and chat away.

---

## 📚 Using the RAG Workflow

1. **Ingest documents to Knowledge Database** (PDF, TXT, DOCX):

   This chunks the document, embeds the chunks, and stores them in `rag_vectors.db`.

2. **Ask questions** in PyChat:\
   Select *Enable RAG* (checkbox near the prompt) → the top‑k similar chunks are appended to the system context before your query hits the LLM.

3. **Tune retrieval** (optional):

   - Edit `RAG_EMBEDDING_MODEL` in `simple_rag_manager.py` (`sentence‑transformers/all‑MiniLM‑L6‑v2` by default).
   - Adjust `chunk_size`, `overlap`, and `top_k` constants.

---

## 🛠️ Project Structure (rag2)

```
PyChat/
├─ pychat.py                 # Main application entry point
├─ simple_rag_manager.py     # Tiny vector store + ingest CLI
├─ simple_rag_ui.py          # UI hooks for RAG toggle & status
├─ openai_handler.py         # Provider adapters …
├─ anthropic_handler.py
├─ ollama_handler.py
├─ thread_ui_components.py   # All PyQt widgets
└─ requirements.txt
```

---



## 📄 License

[MIT](LICENSE)

---



