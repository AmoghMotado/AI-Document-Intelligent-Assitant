"""
Question-answering pipeline, part 1: RETRIEVAL.

1. The question is embedded with the SAME model used for the chunks.
   (Different models produce incompatible vectors.)
2. FAISS returns the TOP_K closest chunks.

Retrieval only FINDS text. It does not write an answer. That is the
job of the LLM (generation), in rag.py.
"""

from src.config import TOP_K


def retrieve_relevant_chunks(vector_store, question, k=TOP_K):
    """
    Return a list of (Document, score) pairs, most relevant first.

    The score is a distance: LOWER means MORE similar. With normalized
    embeddings it ranges roughly from 0 (identical) to 4 (opposite),
    because FAISS reports the squared distance.
    """
    return vector_store.similarity_search_with_score(question, k=k)
