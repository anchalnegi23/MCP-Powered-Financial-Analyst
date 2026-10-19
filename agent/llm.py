"""Chat model factory.

Isolated so that every other module takes a model as an argument. That is what makes the
graph testable without an API key: tests inject a scripted model and exercise the real
graph, real tools and real MCP servers.

The provider is chosen from the model name, so switching between Anthropic and OpenAI is
a one-line change to LLM_MODEL. Nothing downstream knows which provider answered: the
graph uses only bind_tools() and .tool_calls, which both providers implement.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from services.config import get_settings

# OpenAI's reasoning models reject temperature and max_tokens, so those are omitted for
# them. Everything else takes both.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def build_model(temperature: float = 0.0, **kwargs) -> BaseChatModel:
    """Construct the configured chat model.

    Temperature defaults to 0: an analyst that returns a different answer to the same
    question is not useful, and the numbers come from tools anyway.
    """
    settings = get_settings()
    model = settings.llm_model

    if model.startswith("claude"):
        if not settings.anthropic_api_key:
            raise RuntimeError(
                f"LLM_MODEL is '{model}' but ANTHROPIC_API_KEY is not configured. Set it in "
                "your environment or .env file, choose an OpenAI model, or pass an explicit "
                "model into the agent."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
            max_tokens=8000,
            **kwargs,
        )

    if not settings.openai_api_key:
        raise RuntimeError(
            f"LLM_MODEL is '{model}' but OPENAI_API_KEY is not configured. Set it in your "
            "environment or .env file, choose a Claude model, or pass an explicit model "
            "into the agent."
        )
    from langchain_openai import ChatOpenAI

    if not model.startswith(_REASONING_PREFIXES):
        kwargs.setdefault("temperature", temperature)
        kwargs.setdefault("max_tokens", 8000)

    return ChatOpenAI(model=model, api_key=settings.openai_api_key, **kwargs)
