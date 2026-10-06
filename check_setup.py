"""
Quick setup check. Run this BEFORE the app:

    python check_setup.py

It sends one tiny message to Claude to prove your key, model name
and credit all work.
"""

from src.config import ANTHROPIC_MODEL, check_api_key

error = check_api_key()
if error:
    raise SystemExit(error)

from src.llm import get_llm  # imported after the key check on purpose

print("Model:", ANTHROPIC_MODEL)
reply = get_llm().invoke("Reply with exactly these three words: setup works.")
print("Reply:", reply.content)
