"""
Step 4 of the indexing pipeline: store vectors in FAISS.

FAISS (by Meta) is a library for fast similarity search over vectors.
Given a question vector, it quickly finds the stored chunk vectors that
are closest to it.

Our index lives in memory (in Streamlit's session state). It is rebuilt
whenever the user clicks "Process Documents". That is simple and fine
for a prototype with a handful of PDFs.
"""

from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """
    Embed every chunk and put the vectors into a new FAISS index.

    LangChain handles the loop for us: it calls the embedding model on
    each chunk's text and stores the vector together with the chunk.
    """
    return FAISS.from_documents(chunks, embedding_model)
