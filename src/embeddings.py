"""
Step 3 of the indexing pipeline: text -> vectors (embeddings).

An embedding model converts text into a list of numbers. Texts with
similar meaning produce vectors that point in similar directions, even
if they use different words ("refund" vs "money back").

We use a Hugging Face model that runs locally:
  - free, no API key
  - the first run downloads it (~90 MB), then it is cached on disk
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


def get_embedding_model():
    """Create the embedding model used for both chunks and questions."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        # Normalizing makes every vector length 1, so the distance FAISS
        # computes reflects the angle between vectors (i.e. meaning).
        encode_kwargs={"normalize_embeddings": True},
    )
