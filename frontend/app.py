"""Streamlit financial analyst interface.

A thin client. It renders what the API returns and never talks to an MCP server, a data
provider or an LLM directly — so there is no credential in this process to leak, and the
UI cannot drift from the API's behaviour.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.getenv("UI_REQUEST_TIMEOUT", "300"))

EXAMPLE_QUERIES = [
    "What's NVIDIA's current stock price?",
    "Compare NVIDIA and AMD revenue growth.",
    "What are the latest developments affecting NVIDIA?",
    "Compare NVIDIA and AMD on revenue growth, profitability, valuation and recent news.",
    "According to NVIDIA's annual report, what are its major business risks?",
]

st.set_page_config(page_title="Financial Analyst", page_icon="📊", layout="wide")


# ── API calls ─────────────────────────────────────────────────────────────────


def call_api(method: str, path: str, **kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Call the backend, returning (payload, error). Never raises into the UI thread."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.request(method, f"{API_BASE_URL}{path}", **kwargs)
    except httpx.ConnectError:
        return None, (
            f"Could not reach the API at {API_BASE_URL}. Is the backend running? "
            "Start it with: uv run uvicorn api.main:app --reload"
        )
    except httpx.TimeoutException:
        return None, "The analysis timed out. Try a narrower question."
    except httpx.HTTPError as exc:
        return None, f"Network error: {type(exc).__name__}"

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail") or response.json().get("error")
        except ValueError:
            detail = response.text[:300]
        return None, f"API error {response.status_code}: {detail}"

    try:
        return response.json(), None
    except ValueError:
        return None, "The API returned a response that could not be parsed."


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_metrics(metrics: list[dict[str, Any]]) -> None:
    """Show calculated metrics as a table, with the formula that produced each one."""
    if not metrics:
        st.caption("No deterministic calculations were performed for this question.")
        return

    st.dataframe(
        [
            {
                "Metric": m.get("label", m.get("metric", "")),
                "Value": f"{m['value']}{'%' if m.get('unit') == 'percent' else ''}",
                "Unit": m.get("unit", ""),
                "Formula": m.get("formula", ""),
                "Inputs": ", ".join(f"{k}={v:,.0f}" for k, v in (m.get("inputs") or {}).items()),
                "Warnings": "; ".join(m.get("warnings") or []) or "—",
            }
            for m in metrics
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Every value above was computed by the Calculator MCP server in Python, not by "
        "the language model. The formula and inputs are shown so each figure can be checked."
    )


def _format_arguments(arguments: dict[str, Any] | None) -> str:
    return ", ".join(f"{k}={v}" for k, v in (arguments or {}).items())[:120]


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    """Show every tool invocation, successful or not. Failures are the important ones."""
    if not tool_calls:
        st.caption("No tools were invoked.")
        return

    st.dataframe(
        [
            {
                "": "✅" if call.get("ok") else "❌",
                "MCP Server": call.get("server", ""),
                "Tool": call.get("tool", ""),
                "Arguments": _format_arguments(call.get("arguments")),
                "ms": round(call.get("duration_ms", 0), 1),
                "Error": (call.get("error") or "—")[:160],
            }
            for call in tool_calls
        ],
        width="stretch",
        hide_index=True,
    )


def render_response(payload: dict[str, Any]) -> None:
    """Render one analysis: the report, then its provenance."""
    st.markdown(payload.get("answer") or "_No report was produced._")

    warnings = payload.get("warnings") or []
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} warning(s) — data that could not be retrieved", True):
            for warning in warnings:
                st.warning(warning)

    servers = payload.get("mcp_servers_used") or []
    tools = payload.get("tools_used") or []
    columns = st.columns(3)
    columns[0].metric("MCP servers used", len(servers))
    columns[1].metric("Tools invoked", len(payload.get("tool_calls") or []))
    columns[2].metric("Response time", f"{payload.get('duration_ms', 0) / 1000:.1f}s")

    if servers:
        st.caption("**MCP servers:** " + " · ".join(f"`{s}`" for s in servers))
    if tools:
        st.caption("**Tools:** " + " · ".join(f"`{t}`" for t in tools))

    tab_metrics, tab_tools, tab_sources = st.tabs(
        ["📐 Calculated metrics", "🔧 Tool calls", "🔗 Sources"]
    )
    with tab_metrics:
        render_metrics(payload.get("metrics") or [])
    with tab_tools:
        render_tool_calls(payload.get("tool_calls") or [])
    with tab_sources:
        sources = payload.get("sources") or []
        if sources:
            for url in sources:
                st.markdown(f"- [{url}]({url})")
        else:
            st.caption("No external sources were cited for this answer.")


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    st.sidebar.title("📊 Financial Analyst")
    st.sidebar.caption("MCP · LangGraph · FastAPI")

    health, error = call_api("GET", "/health")
    if error:
        st.sidebar.error("Backend unreachable")
        st.sidebar.caption(error)
    else:
        status = health.get("status")
        if status == "ok":
            st.sidebar.success("Backend: ok")
        else:
            st.sidebar.warning(f"Backend: {status}")
        for name, connected in (health.get("mcp_servers") or {}).items():
            st.sidebar.caption(f"{'🟢' if connected else '🔴'} {name} MCP")
        for name, reason in (health.get("failed_servers") or {}).items():
            st.sidebar.caption(f"🔴 {name}: {reason[:60]}")
        if not health.get("agent_ready"):
            st.sidebar.error("No LLM configured — /chat is unavailable.")

    st.sidebar.divider()
    st.sidebar.subheader("Company lookup")
    symbol = st.sidebar.text_input("Ticker", placeholder="NVDA", key="ticker_input")
    if st.sidebar.button("Look up", width="stretch") and symbol.strip():
        snapshot, lookup_error = call_api("GET", f"/companies/{symbol.strip()}")
        if lookup_error:
            st.sidebar.error(lookup_error)
        else:
            profile = snapshot.get("profile") or {}
            quote = snapshot.get("quote") or {}
            st.sidebar.metric(
                profile.get("company_name") or snapshot["symbol"],
                f"{quote.get('price', '—')} {quote.get('currency', '')}",
                f"{quote.get('change_percent', 0):+.2f}%" if quote.get("change_percent") else None,
            )
            if profile.get("sector"):
                st.sidebar.caption(f"{profile['sector']} · {profile.get('industry', '')}")
            for warning in snapshot.get("warnings") or []:
                st.sidebar.caption(f"⚠️ {warning}")

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.history = []
        st.session_state.conversation_id = None
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        "This tool produces analysis of retrieved financial data. It is not investment "
        "advice."
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def submit(query: str) -> None:
    """Send one question and append the exchange to the session history."""
    query = (query or "").strip()
    if not query:
        st.warning("Please enter a question.")
        return

    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"), st.spinner("Planning research and calling MCP tools…"):
        payload, error = call_api(
            "POST",
            "/chat",
            json={"query": query, "conversation_id": st.session_state.conversation_id},
        )
        if error:
            st.error(error)
            st.session_state.history.append({"role": "assistant", "error": error})
            return

        st.session_state.conversation_id = payload.get("conversation_id")
        st.session_state.history.append({"role": "assistant", "payload": payload})
        render_response(payload)


def main() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("conversation_id", None)

    render_sidebar()

    st.title("Financial Analyst")
    st.caption(
        "Ask about a company's financials, valuation, news or filings. The agent plans "
        "the research, selects tools across five MCP servers, and shows its working."
    )

    if not st.session_state.history:
        st.subheader("Try one of these")
        for index, example in enumerate(EXAMPLE_QUERIES):
            if st.button(example, key=f"example_{index}", width="stretch"):
                submit(example)
                st.stop()

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            if turn.get("error"):
                st.error(turn["error"])
            elif turn.get("payload"):
                render_response(turn["payload"])
            else:
                st.markdown(turn.get("content", ""))

    if question := st.chat_input("Ask a financial question…"):
        submit(question)


if __name__ == "__main__":
    main()
