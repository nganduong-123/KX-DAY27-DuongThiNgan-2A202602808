"""LLM factory for OpenRouter or Groq's OpenAI-compatible API."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    use_groq = provider == "groq" or (not provider and groq_key and not openrouter_key)
    if use_groq:
        api_key = groq_key
        default_model = "openai/gpt-oss-120b"
        default_base_url = "https://api.groq.com/openai/v1"
    else:
        api_key = openrouter_key
        default_model = "openai/gpt-4o-mini"
        default_base_url = "https://openrouter.ai/api/v1"

    if not api_key:
        raise RuntimeError(
            "No LLM key is configured. Set OPENROUTER_API_KEY or set "
            "LLM_PROVIDER=groq with GROQ_API_KEY in .env."
        )
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", default_model),
        base_url=os.environ.get("LLM_BASE_URL", default_base_url),
        api_key=api_key,
        temperature=temperature,
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
    )
