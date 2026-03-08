# Telecom Egypt Intelligent Chatbot

A free, local RAG-powered chatbot for Telecom Egypt (te.eg), built with:

| Component | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 1.5 Flash (free tier) |
| Embeddings | sentence-transformers (local, free) |
| Vector Store | ChromaDB (local, free) |
| Web Scraping | BeautifulSoup + requests |
| Document Parsing | PyMuPDF, python-docx, Pillow |

---

## Quick Start

### 1. Clone / navigate to the project folder

```bash
cd telecom_egypt_chatbot
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Gemini API key

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **Create API key**
4. Copy the key

### 5. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and replace `your_gemini_api_key_here` with your actual key:

```
GEMINI_API_KEY=AIzaSy...
```

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## How it works

```
User question
     │
     ▼
sentence-transformers (embed query locally)
     │
     ▼
ChromaDB (find top-5 most relevant chunks)
     │
     ▼
Build prompt: system + context chunks + question
     │
     ▼
Gemini 1.5 Flash (generate answer, streamed)
     │
     ▼
Streamlit UI (display answer + source citations)
```

### First run
On the first run the scraper crawls te.eg (up to 60 pages) and stores the
content in `./chroma_db/`. This takes about **2–5 minutes** depending on your
connection. Every subsequent run loads the existing database instantly.

---

## Project structure

```
telecom_egypt_chatbot/
├── app.py              # Streamlit UI
├── scraper.py          # te.eg website crawler
├── rag.py              # RAG pipeline (retrieve + generate)
├── document_loader.py  # Parse uploaded files
├── vectorstore.py      # ChromaDB setup & operations
├── requirements.txt
├── .env.example
├── .env                # YOUR API KEY — never commit this!
└── chroma_db/          # Created automatically on first run
```

---

## Uploading your own documents

1. Open the **sidebar** in the app
2. Click **Upload Documents**
3. Select PDF, DOCX, TXT, HTML, or image files
4. Click **Add to Knowledge Base**

The chatbot will immediately be able to answer questions from those documents.

---

## Supported languages

- English
- Arabic (عربي)
- Egyptian dialect (automatically detected by Gemini)

---

## Gemini free tier limits

| Limit | Value |
|---|---|
| Requests per minute | 15 |
| Requests per day | 1,500 |
| Tokens per minute | 1,000,000 |

These are more than enough for personal / demo use.

---

## Troubleshooting

**`GEMINI_API_KEY is not set` error**
→ Make sure you created `.env` (not `.env.example`) and added your key.

**Slow first run**
→ Normal — it's scraping ~60 pages and downloading the embedding model (~120 MB).

**`ModuleNotFoundError`**
→ Run `pip install -r requirements.txt` inside your virtual environment.

**Empty answers**
→ The scraper may have been blocked. Try uploading a PDF or DOCX with te.eg content manually.
