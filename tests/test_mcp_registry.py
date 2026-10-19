"""Registry and layering checks for the MCP client. Synchronous: no servers are started."""

from pathlib import Path

from agent.mcp_client import DEFAULT_SERVERS


def test_default_registry_lists_five_servers():
    assert len(DEFAULT_SERVERS) == 5
    assert {s.name for s in DEFAULT_SERVERS} == {
        "calculator",
        "market",
        "financial",
        "news",
        "research",
    }


def test_agent_graph_does_not_import_mcp_machinery():
    """MCP connection logic stays in the client module, not the graph.

    This is the architectural rule of Phase 6 expressed as a test: if a future change
    reaches for fastmcp from inside the graph, this fails.
    """
    assert "fastmcp" in Path("agent/mcp_client.py").read_text()  # the client owns it...

    for other in Path("agent").glob("*.py"):
        if other.name in ("mcp_client.py", "__init__.py"):
            continue
        assert "fastmcp" not in other.read_text(), f"{other} imports MCP machinery directly"
