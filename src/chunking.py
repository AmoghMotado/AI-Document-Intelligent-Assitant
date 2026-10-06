"""
Step 2 of the indexing pipeline: pages -> smaller chunks.

Why chunk at all?
  - Retrieval works best on focused pieces of text. A whole page often
    covers several topics, so its embedding becomes a vague "average".
  - We only send the top few chunks to the LLM, which keeps prompts small,
    cheap and on-topic.

Why RecursiveCharacterTextSplitter?
  It tries to split on paragraph breaks first, then line breaks, then
  sentences, then words. So chunks tend to end at natural boundaries
  instead of in the middle of a word.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """Split page Documents into chunk Documents. Metadata is copied over."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Give each chunk a simple id so we can refer to it in the UI.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
