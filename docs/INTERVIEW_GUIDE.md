# Interview Guide: Document Intelligence Assistant

Only claim what the code does. Everything below matches the implementation.

## Resume entry

**AI-Powered Document Intelligence Assistant**
Python, OpenAI API, LangChain, RAG, FAISS, Hugging Face Embeddings, Streamlit

- Built an LLM-powered document question-answering application using Retrieval-Augmented Generation (RAG) to provide context-aware responses from uploaded PDF documents.
- Implemented document processing, chunking, embedding generation, vector similarity search, and semantic retrieval using LangChain and FAISS.
- Designed prompt templates to combine retrieved document context with LLM responses and provide source-grounded answers through an interactive Streamlit interface.

## 30-second explanation

I built a web app where you upload PDFs and ask questions about them. Instead of sending the whole PDF to the model, it uses RAG: it splits the documents into chunks, turns them into embeddings, and stores them in FAISS. For each question it retrieves the most relevant chunks and has an OpenAI model answer only from them, showing the document and page each answer came from. If the answer isn't in the documents, it says so.

## 1-minute explanation

The problem is that LLMs don't know your private documents and may invent answers when asked about them. My app solves this with Retrieval-Augmented Generation.

When a user uploads PDFs, I extract text page by page with LangChain's PyPDFLoader, keeping the file name and page number as metadata. I split the text into overlapping 1000-character chunks, embed each chunk with a local Hugging Face model, and store the vectors in FAISS.

When the user asks a question, I embed it with the same model, retrieve the four most similar chunks, and insert them into a prompt that tells the model to use only that context, cite sources, and reply with a fixed sentence if the answer is missing. The Streamlit interface shows the answer, a warning for not-found answers, and the retrieved chunks with their pages.

## 2-minute technical explanation

The app has two pipelines.

**Indexing.** Uploaded files are written to temporary files because PyPDFLoader needs a path. It returns one Document per page. I overwrite the source with the real file name and add a 1-based page number. Pages with no text are dropped, and a PDF with no text at all is reported as probably scanned, since there's no OCR. I chunk with RecursiveCharacterTextSplitter, 1000 characters and 200 overlap, which prefers paragraph and sentence boundaries. Embeddings come from all-MiniLM-L6-v2 through langchain-huggingface. It runs locally on CPU, produces 384-dimensional vectors, and I normalize them. The vectors go into an in-memory FAISS index held in Streamlit session state.

**Answering.** The question is embedded with the same model and I call FAISS similarity search with scores for the top four chunks. Each chunk is labelled with its file and page and joined into a context block. A ChatPromptTemplate has a system message with the grounding rules and a human message with the context and question. I pipe it into ChatOpenAI with the model's default temperature, because newer OpenAI models only accept the default. Because FAISS always returns the nearest chunks even for unrelated questions, the not-found decision is made by the model, which must output an exact sentence. The app checks for that sentence to show a warning.

In Streamlit, I cache the embedding model with cache_resource so it loads once, and keep the index and chat history in session state because Streamlit reruns the script on every interaction.

## Architecture

```
Indexing:  PDF → PyPDFLoader → RecursiveCharacterTextSplitter → HF embeddings → FAISS
Answering: question → embed → FAISS top-4 → prompt (rules + chunks + question) → OpenAI LLM → answer + sources
```

Code is split by pipeline stage: loader, chunking, embeddings, vector store, retriever, prompts, LLM, and a `rag.py` that connects them. The UI in `app.py` only calls `build_index` and `answer_question`.

## Why each technology

- **OpenAI API:** follows instructions like "answer only from context" well, and a low-cost model is enough for reading a few chunks. The model name lives in `.env`, so it can be changed without touching code.
- **LangChain:** ready-made PDF loader, text splitter, FAISS wrapper and prompt templates, and a common interface to chat models so I could switch providers by changing one file.
- **Local Hugging Face embeddings:** free, no API key, and private (text never leaves the machine for embedding). OpenAI also sells embeddings, but a local model keeps that step free.
- **FAISS:** fast, simple, runs in-process with no server to set up. Right for a local prototype.
- **Streamlit:** a working web UI in pure Python, so I could focus on the RAG logic.

## Important technical decisions

- **Chunk size 1000 / overlap 200:** a common starting point balancing focused embeddings with enough context per chunk. Overlap prevents losing sentences split at boundaries.
- **Top-k of 4:** enough context for most questions while keeping prompts short and cheap.
- **Same embedding model for chunks and questions:** vectors from different models are not comparable.
- **Default temperature:** newer OpenAI models reject temperature 0, so grounding relies on the strict prompt rules. With an older model that allows it, `LLM_TEMPERATURE = 0` gives more consistent answers.
- **Exact not-found sentence:** makes "answer missing" machine-detectable, so the UI can show it clearly.
- **Metadata carried through every step:** enables page-level source references.
- **In-memory index:** simplest option for a prototype. Persistence is listed as a future improvement.

## Limitations

- No OCR for scanned PDFs.
- Index is not saved; it is rebuilt each time.
- No conversation memory between questions.
- Fixed top-4 retrieval, so broad summaries of long documents are partial.
- Grounding depends on the model following the prompt; it reduces but can't eliminate hallucination.
- Tables and complex layouts may extract poorly.
- Built and tested as a single-user local prototype.

## Future improvements

- Persist the FAISS index to disk.
- Conversation memory for follow-up questions.
- Hybrid keyword + vector search and re-ranking.
- A relevance threshold to skip the LLM when no chunk is close enough.
- OCR for scanned PDFs.
- An evaluation set to measure retrieval and answer quality.

## Interview questions and answers

**1. What is an LLM?**
A Large Language Model is a neural network trained on huge amounts of text to predict the next token. That lets it answer questions, summarize and write. It only knows its training data, so it doesn't know your private documents.

**2. What is RAG, and why is it needed?**
Retrieval-Augmented Generation means retrieving relevant text from your own data and adding it to the prompt before the LLM generates an answer. It's needed because the LLM doesn't know your documents, may hallucinate, and sending entire documents is slow and expensive.

**3. What's the difference between retrieval and generation?**
Retrieval finds the relevant text; in my app FAISS does that and writes nothing. Generation writes the answer from that text; the OpenAI model does that and searches nothing.

**4. What is an embedding?**
A list of numbers representing the meaning of a text. Similar meanings produce vectors that are close together. My model produces 384 numbers per chunk.

**5. What is semantic similarity, and how is it different from keyword search?**
It's similarity of meaning. "How many holidays do I get?" matches "18 days of paid annual leave" even though they share few words. Keyword search would miss that.

**6. What is vector search, and what does FAISS do?**
Vector search finds the stored vectors nearest to a query vector. FAISS is a library from Meta that does this efficiently. I store chunk embeddings in it and query with the question's embedding.

**7. What is chunking, and why does chunk size matter?**
Splitting documents into smaller pieces. Chunks that are too large give vague embeddings and waste prompt space; too small and chunks lose context. I used 1000 characters with 200 overlap so sentences at boundaries aren't lost.

**8. Why must you use the same embedding model for documents and questions?**
Each model has its own vector space. Comparing vectors from different models is meaningless, like comparing coordinates on two different maps.

**9. How does your app reduce hallucination?**
The prompt gives the model the retrieved context and tells it to answer only from it, not use outside knowledge, cite sources, and use a fixed sentence when the answer is missing. It reduces hallucination but can't guarantee zero.

**10. What happens when the answer isn't in the documents?**
FAISS still returns the four nearest chunks, because it returns nearest, not relevant. The model reads them, sees they don't answer the question, and replies with the exact not-found sentence. The app detects it and shows a warning.

**11. How is the prompt constructed?**
A system message with rules, and a human message with the retrieved chunks, each labelled with file name and page, followed by the question. The labels are what let the model cite sources.

**12. How is LangChain used?**
PyPDFLoader for PDFs, RecursiveCharacterTextSplitter for chunking, HuggingFaceEmbeddings and the FAISS wrapper for indexing and search, ChatPromptTemplate for the prompt, and ChatOpenAI for the OpenAI model, connected with a chain using the pipe operator.

**13. How do you show source references?**
Every page Document carries its file name and page number in metadata. The splitter copies metadata to each chunk, and FAISS returns chunks with their metadata, so I can display them and label context for citations.

**14. Why Streamlit, and how does it handle state?**
It builds a web UI in pure Python. It reruns the whole script on every interaction, so I store the FAISS index and chat history in session_state and cache the embedding model with cache_resource so it loads once.

**15. What are the main limitations, and how would you improve it for production?**
No OCR, an in-memory index, no conversation memory, fixed top-k retrieval, and grounding that depends on the model. For production I'd persist the index or use a managed vector database, add hybrid search and re-ranking, add memory, build an evaluation set to measure quality, and handle multiple users.
