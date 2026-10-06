"""
Creates the Claude chat model through LangChain.

ChatAnthropic reads ANTHROPIC_API_KEY from the environment. Because the
rest of the code only talks to a LangChain chat model, switching to a
different provider later means changing just this file.
"""

from langchain_anthropic import ChatAnthropic

from src.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


def get_llm():
    """Return a configured Claude chat model."""
    return ChatAnthropic(
        model=ANTHROPIC_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
