"""
Prompt templates.

This is where "grounding" happens. We give the LLM:
  - strict rules: use only the context, never invent facts
  - the retrieved chunks, each labelled with its source and page
  - the user's question

If the context does not contain the answer, the model must reply with
our exact NOT_FOUND_MESSAGE, so the app can detect it reliably.
"""

from langchain_core.prompts import ChatPromptTemplate

from src.config import NOT_FOUND_MESSAGE

SYSTEM_PROMPT = f"""You are a document question-answering assistant.

Rules:
1. Answer ONLY using the information in the provided context.
2. Do not use outside knowledge and do not guess or invent facts.
3. If the context does not contain the answer, reply with exactly:
   "{NOT_FOUND_MESSAGE}"
4. If the context only partly answers the question, answer the part you
   can and clearly say which part is not covered by the documents.
5. When you use information, mention its source in brackets, like
   [report.pdf, page 3].
6. Be clear and concise. Use bullet points for lists or comparisons."""

HUMAN_PROMPT = """Context from the uploaded documents:
{context}

Question: {question}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def format_context(results):
    """
    Turn retrieved (Document, score) pairs into one labelled text block.

    Each chunk is tagged with its file name and page so the LLM can cite it.
    """
    blocks = []
    for doc, _score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", "?")
        blocks.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)
