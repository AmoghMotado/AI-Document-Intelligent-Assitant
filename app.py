"""
AI-Powered Document Intelligence Assistant - Streamlit UI.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.config import ANTHROPIC_MODEL, CHUNK_SIZE, TOP_K, check_api_key
from src.embeddings import get_embedding_model
from src.rag import answer_question, build_index

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
)


# st.cache_resource keeps the embedding model in memory across reruns,
# so it is loaded only once instead of on every button click.
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def load_embedding_model():
    return get_embedding_model()


def init_session_state():
    """Streamlit reruns the whole script on every interaction.
    session_state is how we remember things between reruns."""
    defaults = {
        "vector_store": None,
        "processed_files": [],
        "stats": None,
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_sources(sources):
    """Display the retrieved chunks with document name and page number."""
    with st.expander(f"Retrieved sources ({len(sources)} chunks)"):
        for rank, (doc, score) in enumerate(sources, start=1):
            name = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page_number", "?")
            st.markdown(f"**{rank}. {name} — page {page}**  \n"
                        f"Distance: `{score:.3f}` (lower = more similar)")
            st.caption(doc.page_content)
            if rank < len(sources):
                st.divider()


def show_answer(result):
    """Display one answer, with a warning style if nothing was found."""
    if result["not_found"]:
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])
    show_sources(result["sources"])


init_session_state()

# ---------------- Sidebar: upload and process ----------------
with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button(
        "Process Documents", type="primary", use_container_width=True
    )

    if process_clicked:
        if not uploaded_files:
            st.error("Please upload at least one PDF first.")
        else:
            embedding_model = load_embedding_model()
            with st.spinner("Extracting text, chunking and building the FAISS index..."):
                result = build_index(uploaded_files, embedding_model)

            for err in result["errors"]:
                st.error(err)

            if result["vector_store"] is not None:
                st.session_state.vector_store = result["vector_store"]
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.stats = result
                st.session_state.chat_history = []  # new documents, fresh chat
                st.success("Documents processed. Ask a question!")

    if st.session_state.stats:
        st.header("Indexed documents")
        for name in st.session_state.processed_files:
            st.markdown(f"- {name}")
        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.stats["num_pages"])
        col2.metric("Chunks", st.session_state.stats["num_chunks"])

    st.divider()
    st.caption(
        f"LLM: `{ANTHROPIC_MODEL}`  \n"
        f"Embeddings: all-MiniLM-L6-v2 (local)  \n"
        f"Chunk size: {CHUNK_SIZE} chars · Top-k: {TOP_K}"
    )
    if st.session_state.chat_history and st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- Main area: title and chat ----------------
st.title("📄 AI-Powered Document Intelligence Assistant")
st.markdown(
    "Upload PDFs and ask questions about them. Answers are generated with "
    "**Retrieval-Augmented Generation (RAG)**: the app finds the most relevant "
    "passages with vector search, and Claude answers using only those passages."
)

api_key_error = check_api_key()
if api_key_error:
    st.error(api_key_error)
    st.stop()

if st.session_state.vector_store is None:
    st.info("👈 Upload one or more PDFs and click **Process Documents** to begin.")
    st.stop()

# Replay the conversation so far.
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        show_answer(turn["result"])

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating answer..."):
                result = answer_question(st.session_state.vector_store, question)
            show_answer(result)
            st.session_state.chat_history.append(
                {"question": question, "result": result}
            )
        except Exception as e:
            st.error(
                "Something went wrong while generating the answer. "
                "Check your API key, model name and API credit.\n\n"
                f"Details: {e}"
            )
