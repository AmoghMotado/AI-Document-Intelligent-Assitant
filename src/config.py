"""
Central place for all settings.

Every other file imports its settings from here, so if you want to
experiment (e.g. a different chunk size), you only change one file.
"""

import os
from dotenv import load_dotenv

# Read the .env file and put its values into environment variables.
load_dotenv()

# --- LLM (Claude) settings ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# temperature=0 makes answers more deterministic and less "creative",
# which is what we want when answering strictly from documents.
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1024

# --- Embedding settings ---
# A small, free model that runs locally on your CPU. No API key needed.
# It turns text into a 384-number vector that represents its meaning.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking settings ---
# chunk_size is measured in characters (roughly 150-200 words).
# chunk_overlap repeats a little text between neighbouring chunks so a
# sentence cut in half at a boundary still appears whole in one chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval settings ---
# How many chunks to retrieve for each question.
TOP_K = 4

# The exact sentence the LLM must use when the answer is not in the documents.
# The UI checks for this sentence to show a "not found" warning.
NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."


def check_api_key():
    """Return an error message if the API key is missing, else None."""
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("your-"):
        return (
            "ANTHROPIC_API_KEY is missing. Copy .env.example to .env "
            "and paste your real key from console.anthropic.com."
        )
    return None
