# AI-Powered Document Intelligence Assistant

A Streamlit web app that answers questions about your PDF documents using Retrieval-Augmented Generation (RAG), with Claude as the LLM, local Hugging Face embeddings and a FAISS vector index.

## Overview

Upload one or more PDFs, click **Process Documents**, and ask questions in a chat box. The app finds the passages most relevant to your question and asks Claude to answer using only those passages. Every answer shows the document name, page number and text of the chunks it was based on. If the answer is not in the documents, the app says so instead of guessing.

## Problem Statement

Large language models can answer general questions, but they do not know the contents of your private documents, and when asked anyway they may produce confident but invented answers (hallucinations). Pasting an entire long PDF into a prompt is also slow, costly and limited by the model's context size.

## Solution

The app uses RAG. Documents are split into small chunks, each chunk is converted into an embedding, and the embeddings are stored in FAISS. For each question, only the most similar chunks are retrieved and passed to Claude with strict instructions to answer from that context alone and to cite sources.

## Key Features

- Upload and process multiple PDFs at once
- Page-level text extraction with document name and page number kept as metadata
- Recursive character chunking with overlap
- Local, free embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- In-memory FAISS vector index with similarity search
- Grounded prompt that forbids outside knowledge and requires source citations
- Explicit "not found" response, shown as a warning in the UI
- Retrieved source panel with document, page, chunk text and similarity distance
- Chat interface with conversation display, loading spinners and clear error messages
- Detects PDFs with no extractable text (e.g. scanned images) and reports them

## Architecture

```
INDEXING (on "Process Documents")
PDF files ─► PyPDFLoader (text per page + metadata)
          ─► RecursiveCharacterTextSplitter (chunks, 1000 chars, 200 overlap)
          ─► HuggingFace embeddings (384-dim vectors, local CPU)
          ─► FAISS index (kept in Streamlit session state)

QUESTION ANSWERING (on every question)
Question ─► embed with the same model
         ─► FAISS similarity search (top 4 chunks)
         ─► prompt = rules + labelled chunks + question
         ─► Claude (via LangChain ChatAnthropic)
         ─► answer + source references in the Streamlit UI
```

## Tech Stack

| Technology | Role |
|---|---|
| Python | Application language |
| Anthropic Claude API | LLM that generates grounded answers |
| LangChain | PDF loader, text splitter, FAISS wrapper, prompt template, Claude integration |
| RAG | Overall retrieve-then-generate design |
| Hugging Face Embeddings | Local `all-MiniLM-L6-v2` model via `langchain-huggingface` |
| FAISS | Vector index and similarity search |
| Streamlit | Web interface |

## Project Structure

```
ai-document-intelligence-assistant/
├── app.py                  # Streamlit UI
├── check_setup.py          # One-message test of key, model and credit
├── requirements.txt
├── .env.example            # Template for .env (the real .env is git-ignored)
├── .gitignore
├── README.md
├── data/                   # Reserved for local files (git-ignored)
├── sample_docs/            # Two fictional PDFs for testing
├── docs/
│   ├── CODE_WALKTHROUGH.md # File-by-file explanation and concepts
│   ├── INTERVIEW_GUIDE.md  # How to explain this project
│   └── screenshots/
└── src/
    ├── config.py           # All settings in one place
    ├── document_loader.py  # PDF -> page Documents
    ├── chunking.py         # Pages -> chunks
    ├── embeddings.py       # Embedding model
    ├── vector_store.py     # FAISS index creation
    ├── retriever.py        # Similarity search
    ├── prompts.py          # Grounding prompt + context formatting
    ├── llm.py              # Claude chat model
    └── rag.py              # Ties the pipeline together
```

## How It Works

1. **Load:** each uploaded PDF is written to a temporary file and read page by page. The file name and page number are stored in each page's metadata.
2. **Chunk:** pages are split into ~1000-character chunks with 200 characters of overlap, preferring paragraph and sentence boundaries.
3. **Embed and index:** every chunk is embedded locally and stored in a FAISS index.
4. **Retrieve:** the question is embedded with the same model, and FAISS returns the 4 closest chunks.
5. **Generate:** the chunks, labelled with source and page, are inserted into a prompt that tells Claude to answer only from them, cite sources, and reply with a fixed sentence if the answer is missing.
6. **Display:** the answer appears in the chat. A "not found" reply is shown as a warning, and the retrieved chunks are listed in an expandable section.

## Setup

Requirements: Python 3.11 or 3.12, Git, and an Anthropic API key with credit.

```bash
git clone https://github.com/<your-username>/ai-document-intelligence-assistant.git
cd ai-document-intelligence-assistant
python -m venv venv
```

Activate the environment:

```bash
# Windows (Git Bash)
source venv/Scripts/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The first install downloads PyTorch as a dependency of `sentence-transformers`, so it can take several minutes.

## Environment Variables

Copy `.env.example` to `.env` and fill in your key:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Your key from console.anthropic.com. Never commit it. |
| `ANTHROPIC_MODEL` | The Claude model to use. Change it without editing code. |

## Running the Application

Check the API setup first:

```bash
python check_setup.py
```

Then start the app:

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`. The first time you process documents, the embedding model (~90 MB) is downloaded and cached.

## Example Queries

Using the two PDFs in `sample_docs/` (fictional company documents):

| Type | Question | Expected behaviour |
|---|---|---|
| Direct fact | How many days of paid annual leave do employees get? | 18 days, citing the handbook, page 1 |
| Multiple chunks / documents | What must an employee do before and while working remotely? | Probation and manager approval (handbook) plus VPN rules (security policy) |
| Not in documents | What is NovaTech's maternity leave policy? | The fixed "could not find" message, shown as a warning |
| Summary | Summarize the IT security policy. | Short summary of passwords, devices, remote access, incidents |
| Comparison | Compare the Standard and Premium health plans. | Coverage, sum insured and cost of each, citing handbook page 2 |

## Screenshots

Add screenshots to `docs/screenshots/` and link them here, for example:

```markdown
![Upload and processing](docs/screenshots/upload.png)
![Answer with sources](docs/screenshots/answer.png)
![Not found response](docs/screenshots/not_found.png)
```

## Limitations

- **No OCR:** scanned or image-only PDFs have no extractable text and are reported as unreadable.
- **In-memory index:** the FAISS index is rebuilt on every "Process Documents" click and lost when the app restarts.
- **No conversation memory:** each question is answered independently, so follow-ups like "tell me more about that" do not use earlier turns.
- **Fixed top-k retrieval:** the app always retrieves 4 chunks. Questions needing information spread across many sections may get incomplete answers, and broad "summarize everything" requests only see part of a long document.
- **Grounding depends on the model following instructions:** the prompt strongly reduces hallucination but cannot guarantee it.
- **Tables and complex layouts** may be extracted as jumbled text.
- **Single user, local prototype:** not designed or tested for multiple users or large document collections.

## Future Improvements

- Save and reload the FAISS index to disk so documents need not be reprocessed
- Add conversation memory so follow-up questions work
- Hybrid search (keyword + vector) and a re-ranking step for better retrieval
- OCR support for scanned PDFs
- A relevance threshold to skip the LLM call when no chunk is similar enough
- Streaming answers token by token
- An evaluation set of questions with expected answers to measure retrieval quality
