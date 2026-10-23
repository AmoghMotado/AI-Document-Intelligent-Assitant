# Code Walkthrough

Read this with the code open. Go in the order below, which follows the data as it moves through the app. For each file, try to answer the "Check yourself" question without looking.

## The big picture

RAG has two pipelines.

**Indexing** runs once when you click Process Documents:
`PDF → pages → chunks → embeddings → FAISS`

**Answering** runs for every question:
`question → embedding → similarity search → top chunks → prompt → LLM → answer`

The key idea: **retrieval finds relevant text, generation writes the answer from it.** FAISS never writes anything. The LLM never searches anything.

## 1. `src/config.py`

Every setting lives here: API key, model name, chunk size, overlap, top-k and the "not found" sentence. `load_dotenv()` reads `.env` into environment variables so no secret appears in code.

Why centralize? To experiment, you change one number in one file.

*Check yourself:* Why is `LLM_TEMPERATURE` set to `None`?
(Newer OpenAI models reject temperature 0 and only accept their default, so `None` tells LangChain not to send a temperature at all. Lower temperature means less randomness, so with a model that allows it, 0 is a good choice for factual answers.)

## 2. `src/document_loader.py`

`PyPDFLoader` returns one LangChain `Document` per page. A Document is just `page_content` (text) plus `metadata` (a dictionary).

Two details matter:
- Streamlit keeps uploads in memory, but `PyPDFLoader` wants a file path. So each upload is written to a temporary file, loaded, then deleted in the `finally` block, which runs even if loading fails.
- PyPDF numbers pages from 0. We add `page_number = page + 1` for humans, and replace the temporary path with the real file name in `source`.

Pages with no text are dropped. If a whole PDF has no text, it is probably a scanned image, and we report that.

*Check yourself:* Why does the metadata matter so much?
(It is the only way we can later show which document and page an answer came from.)

## 3. `src/chunking.py`

**What chunking is:** cutting long text into smaller pieces.

**Why:** an embedding is one vector for a whole piece of text. If a page covers leave policy, working hours and insurance, its vector is a blurry average of all three and matches none of them well. Smaller chunks give sharper vectors. Also, we only send a few chunks to the LLM, so they need to be small.

**Why size matters:**
- Too large: vague embeddings, irrelevant text in the prompt, higher cost.
- Too small: a chunk may hold half a sentence with no context, so even a correct match is not useful.

1000 characters with 200 overlap is a common starting point. **Overlap** repeats the end of one chunk at the start of the next, so a sentence cut at a boundary still appears whole somewhere.

`RecursiveCharacterTextSplitter` tries separators in order: paragraph break, line break, sentence end, space. It only falls back to a smaller separator when a piece is still too long.

*Experiment:* change `CHUNK_SIZE` to 300 and then 3000, reprocess the sample PDFs, and watch the chunk count and answer quality.

## 4. `src/embeddings.py`

**What an embedding is:** a list of numbers (here 384) that represents the meaning of a text. Texts with similar meaning get vectors pointing in similar directions, even with different words. "How many holidays do I get?" and "18 days of paid annual leave" share almost no words but land close together.

This is **semantic similarity**: similarity of meaning, not of exact words. Keyword search would miss that match.

We use `all-MiniLM-L6-v2`: small, fast on a CPU, free, no API key. `normalize_embeddings=True` scales every vector to length 1 so distances reflect direction (meaning) rather than length.

Note: OpenAI also offers paid embedding models, but using a local one keeps this step free and keeps your text on your machine.

## 5. `src/vector_store.py`

**What FAISS does:** stores vectors and quickly finds the ones nearest to a query vector. `FAISS.from_documents` embeds every chunk and stores the vectors together with the chunk text and metadata.

The index lives in memory, in Streamlit's session state. It is rebuilt on each Process click and gone after a restart. That is a deliberate simplicity choice, listed in Limitations.

## 6. `src/retriever.py`

**Retrieval:** embed the question with the *same* model used for chunks, then ask FAISS for the `TOP_K` nearest chunks.

Why the same model? Each model defines its own "meaning space". Vectors from two different models are not comparable.

The score is a distance: **lower means more similar.** The UI shows it so you can see how confident retrieval was.

## 7. `src/prompts.py`

This is where **grounding** happens. The system prompt tells the LLM:
- use only the context
- never use outside knowledge or guess
- if the answer is missing, reply with one exact sentence
- cite sources as `[file, page]`

`format_context` labels each chunk with its file and page, which is how the LLM knows what to cite.

**How this reduces hallucination:** the model is given the facts and told to stay inside them. Instead of recalling (or inventing) an answer, it reads one. It is a large reduction, not a guarantee.

**What happens when information isn't found:** FAISS *always* returns 4 chunks, even for an unrelated question, because it returns the nearest ones, not only good ones. So the "not found" decision is made by the LLM after reading them. The exact sentence lets the app detect it and show a warning.

## 8. `src/llm.py`

Creates `ChatOpenAI` from `langchain-openai`. Because the rest of the app only uses LangChain's chat model interface, switching providers means changing only this file.

## 9. `src/rag.py`

Ties everything together.

- `build_index` runs the indexing pipeline and returns the index plus stats.
- `answer_question` runs retrieval, then builds the prompt, then calls the LLM.

`RAG_PROMPT | get_llm()` is a LangChain **chain**: the `|` pipes the filled-in prompt into the model. `get_text` handles the fact that LangChain may return the reply as a string or as a list of content blocks.

The three steps are literally the letters of RAG: **R**etrieve, **A**ugment (add context to the prompt), **G**enerate.

## 10. `app.py`

Streamlit re-runs the whole script from the top on every click. So:
- `st.session_state` remembers the index and chat history between reruns.
- `@st.cache_resource` loads the embedding model once, not on every click.

The sidebar handles upload and processing. The main area shows the chat. `st.stop()` halts the page early when there is no API key or no processed documents, which keeps the code simple.

## 11. `check_setup.py`

Sends one tiny message to OpenAI. Run it first; if it fails, the problem is the key, model name or credit, not your RAG code.

## Test questions and what should happen

Use the two PDFs in `sample_docs/`.

1. **Direct fact:** "How many days of paid annual leave do employees get?"
   Should answer 18 days and cite the handbook, page 1. The top source should be the leave-policy chunk.

2. **Multiple chunks:** "What must an employee do before and while working remotely?"
   Needs the handbook (3-month probation, max 2 days a week, manager approval) and the security policy (VPN first, public Wi-Fi only on VPN, no home printing). Check the sources panel shows both files.

3. **Not in documents:** "What is NovaTech's maternity leave policy?"
   Should return the exact "could not find" sentence as a yellow warning. Sources will still appear, because FAISS always returns something. Notice their distances are higher.

4. **Summary:** "Summarize the IT security policy."
   Should give a short summary. With only 4 chunks retrieved, a longer document would be summarized only partly. That is a real limitation worth mentioning in interviews.

5. **Comparison:** "Compare the Standard and Premium health plans."
   Should list coverage (employee only vs family), sum insured (3 vs 7 lakh) and cost (free vs 1,200 rupees per month).

Try also: "Who is the CEO of NovaTech?" (not found) and "What is the capital of France?" (should be not found, because the model must not use outside knowledge).
