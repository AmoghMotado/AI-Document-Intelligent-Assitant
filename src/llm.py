"""
Creates the OpenAI chat model through LangChain.

ChatOpenAI sends our prompt to OpenAI's API. Because the rest of the code
only talks to a LangChain chat model, switching to a different provider
later means changing just this file (and the settings in config.py).
"""

from langchain_openai import ChatOpenAI

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


def get_llm():
    """Return a configured OpenAI chat model."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
