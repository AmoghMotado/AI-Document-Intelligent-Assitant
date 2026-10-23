"""
Central place for all settings.

Every other file imports its settings from here, so if you want to
experiment (e.g. a different chunk size), you only change one file.
"""

import os
from dotenv import load_dotenv

# Read the .env file and put its values into environment variables.
load_dotenv()

# --- LLM (OpenAI) settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

# None means "use the model's default temperature".
# Newer OpenAI models reject temperature=0 and only accept the default.
# Grounding comes from the prompt rules (answer only from the context).
# If your model supports it, set this to 0 for more consistent answers.
LLM_TEMPERATURE = None
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
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your-"):
        return (
            "OPENAI_API_KEY is missing. Copy .env.example to .env "
            "and paste your real key from platform.openai.com/api-keys."
        )
    return None
