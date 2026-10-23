"""
The full RAG pipeline, tying every module together.

Indexing (once per upload):
    PDFs -> pages -> chunks -> embeddings -> FAISS

Answering (every question):
    question -> retrieve chunks -> build prompt -> LLM -> answer
"""

from src.chunking import split_documents
from src.config import NOT_FOUND_MESSAGE
from src.document_loader import load_uploaded_pdfs
from src.llm import get_llm
from src.prompts import RAG_PROMPT, format_context
from src.retriever import retrieve_relevant_chunks
from src.vector_store import create_vector_store


def get_text(content):
    """
    The model's reply is usually a plain string, but LangChain can also return
    a list of content blocks. This collects the text in either case.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def build_index(uploaded_files, embedding_model):
    """
    Run the indexing pipeline.

    Returns a dict with the vector store and some stats for the UI.
    """
    documents, errors = load_uploaded_pdfs(uploaded_files)

    if not documents:
        return {"vector_store": None, "errors": errors, "num_pages": 0, "num_chunks": 0}

    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks, embedding_model)

    return {
        "vector_store": vector_store,
        "errors": errors,
        "num_pages": len(documents),
        "num_chunks": len(chunks),
    }


def answer_question(vector_store, question):
    """
    Run the question-answering pipeline.

    Returns a dict with the answer, the retrieved sources, and whether the
    model said the answer was not found.
    """
    # 1. RETRIEVAL: find the most relevant chunks.
    results = retrieve_relevant_chunks(vector_store, question)

    # 2. AUGMENTATION: put those chunks into the prompt.
    context = format_context(results)

    # 3. GENERATION: the LLM writes an answer from that context only.
    #    "RAG_PROMPT | llm" is a LangChain chain: fill the template,
    #    then pass the resulting messages to the model.
    chain = RAG_PROMPT | get_llm()
    response = chain.invoke({"context": context, "question": question})
    answer = get_text(response.content)

    return {
        "answer": answer,
        "sources": results,
        "not_found": NOT_FOUND_MESSAGE.lower() in answer.lower(),
    }
