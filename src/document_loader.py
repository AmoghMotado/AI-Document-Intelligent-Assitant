"""
Step 1 of the indexing pipeline: PDF -> text.

We use LangChain's PyPDFLoader, which reads a PDF and returns one
Document per page. A LangChain Document has two parts:
  - page_content: the text on that page
  - metadata: extra info, e.g. {"source": "file.pdf", "page": 0}
Keeping the page number in metadata is what lets us show sources later.
"""

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path, display_name=None):
    """Load one PDF from disk and return a list of page Documents."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for page in pages:
        # Use the friendly file name instead of a temporary file path.
        if display_name:
            page.metadata["source"] = display_name
        # PyPDF counts pages from 0. Humans count from 1.
        page.metadata["page_number"] = page.metadata.get("page", 0) + 1

    # Drop pages with no extractable text (e.g. blank or image-only pages).
    return [p for p in pages if p.page_content.strip()]


def load_uploaded_pdfs(uploaded_files):
    """
    Load PDFs uploaded through Streamlit.

    Streamlit gives us file objects held in memory, but PyPDFLoader needs a
    real file path, so we write each upload to a temporary file first.

    Returns (documents, errors):
      documents: all page Documents from every readable PDF
      errors:    list of human-readable messages for files that failed
    """
    documents = []
    errors = []

    for uploaded_file in uploaded_files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            pages = load_pdf(tmp_path, display_name=uploaded_file.name)

            if not pages:
                errors.append(
                    f"{uploaded_file.name}: no text found. It may be a scanned "
                    "image PDF, which this app does not support (no OCR)."
                )
            documents.extend(pages)

        except Exception as e:
            errors.append(f"{uploaded_file.name}: could not be read ({e}).")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return documents, errors
