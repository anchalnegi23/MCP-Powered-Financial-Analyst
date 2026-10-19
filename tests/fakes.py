"""A scripted chat model, so the real graph can be tested without an LLM or an API key.

This runs the *actual* LangGraph, the *actual* MCP client and the *actual* FastMCP
servers. Only the model's choices are scripted, which is exactly the part that would
otherwise be non-deterministic and cost money.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool


class ScriptedChatModel(BaseChatModel):
    """Returns pre-programmed AI messages in order, recording what it was asked and offered.

    `script` may contain AIMessage objects (including ones carrying `tool_calls`) or plain
    strings. Once exhausted, it repeats the last entry, so a stage that loops does not
    crash the test.
    """

    script: list[Any] = []
    index: int = 0
    bound_tool_names: list[list[str]] = []
    prompts_seen: list[str] = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> ScriptedChatModel:
        """Record which tools this stage was offered. Returns self: the script decides."""
        self.bound_tool_names.append([t.name for t in tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts_seen.append(
            "\n".join(str(m.content) for m in messages if isinstance(m.content, str))
        )
        entry = (
            self.script[self.index]
            if self.index < len(self.script)
            else (self.script[-1] if self.script else "")
        )
        self.index += 1
        message = entry if isinstance(entry, AIMessage) else AIMessage(content=str(entry))
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """Build an AI message that requests one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


REPORT_WITH_ALL_SECTIONS = """
## Executive Summary
NVIDIA grew revenue sharply (calculated).

## Financial Performance
Revenue rose to $60.9B in FY2024 (retrieved).

## Revenue Growth
Revenue growth was 125.85% (calculated).

## Profitability
Net margin was 48.85% (calculated).

## Valuation
Trailing P/E of 51.2x (retrieved).

## Recent News
Reuters reported a new data centre GPU (https://example.com/nvidia-gpu-launch).

## Risks
Customer concentration and export controls.

## Bull Case
Data centre demand continues.

## Bear Case
Export restrictions bite and competition intensifies.

## Conclusion
Based on the retrieved financial data, growth has been exceptional and valuation is high.
This is analysis of retrieved data, not investment advice.
""".strip()
