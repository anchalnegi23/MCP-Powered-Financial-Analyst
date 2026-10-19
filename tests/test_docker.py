"""Docker configuration checks that do not require a Docker daemon.

The live stack is verified by actually running it (see docs/deployment.md); these guard
the configuration itself so a bad edit fails in CI rather than on a deploy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

COMPOSE = yaml.safe_load(Path("docker-compose.yml").read_text())
DOCKERFILE = Path("Dockerfile").read_text()
SERVICES = COMPOSE["services"]

EXPECTED_SERVICES = {
    "postgres",
    "redis",
    "calculator-mcp",
    "market-mcp",
    "financial-mcp",
    "news-mcp",
    "research-mcp",
    "api",
    "ui",
}
MCP_SERVICES = {s for s in EXPECTED_SERVICES if s.endswith("-mcp")}


# ── Services ──────────────────────────────────────────────────────────────────


def test_every_component_has_a_service():
    assert set(SERVICES) == EXPECTED_SERVICES


def test_all_five_mcp_servers_are_separate_services():
    """The architectural claim, checked: five servers means five containers."""
    assert len(MCP_SERVICES) == 5


@pytest.mark.parametrize("service", sorted(EXPECTED_SERVICES))
def test_every_service_has_a_healthcheck(service):
    assert "healthcheck" in SERVICES[service], f"{service} has no healthcheck"


@pytest.mark.parametrize("service", sorted(MCP_SERVICES))
def test_mcp_healthchecks_perform_a_real_protocol_handshake(service):
    """A bare GET /mcp returns 400, so it cannot tell healthy from broken."""
    test = " ".join(SERVICES[service]["healthcheck"]["test"])
    assert "initialize" in test
    assert "jsonrpc" in test


# ── Wiring ────────────────────────────────────────────────────────────────────


def test_the_api_waits_for_every_dependency_to_be_healthy():
    depends = SERVICES["api"]["depends_on"]
    assert set(depends) == {"postgres", "redis", *MCP_SERVICES}
    assert all(d["condition"] == "service_healthy" for d in depends.values())


def test_the_api_is_pointed_at_every_mcp_container():
    environment = SERVICES["api"]["environment"]
    for name, port in [
        ("CALCULATOR", 8001),
        ("MARKET", 8002),
        ("FINANCIAL", 8003),
        ("NEWS", 8004),
        ("RESEARCH", 8005),
    ]:
        assert environment[f"{name}_MCP_URL"].endswith(f":{port}/mcp")


def test_the_ui_talks_to_the_api_by_service_name():
    assert SERVICES["ui"]["environment"]["API_BASE_URL"] == "http://api:8000"
    assert SERVICES["ui"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_postgres_and_the_research_store_have_persistent_volumes():
    assert "pgdata:/var/lib/postgresql/data" in SERVICES["postgres"]["volumes"]
    assert set(COMPOSE["volumes"]) == {"pgdata", "chromadata"}


def test_only_the_research_server_mounts_the_document_store():
    """The API reaches documents through MCP, never by reading the vector store itself."""
    assert any("chromadata" in v for v in SERVICES["research-mcp"]["volumes"])
    assert not any("chromadata" in v for v in SERVICES["api"].get("volumes", []))
    assert "CHROMA_PERSIST_DIR" not in SERVICES["api"]["environment"]


# ── Secrets ───────────────────────────────────────────────────────────────────


def test_no_secret_is_hardcoded_in_the_compose_file():
    raw = Path("docker-compose.yml").read_text()
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for marker in ("API_KEY:", "PASSWORD:", "SECRET:", "TOKEN:"):
            if marker in stripped:
                # The value must be an interpolation, never a literal.
                value = stripped.split(marker, 1)[1].strip()
                assert value.startswith("${"), f"literal secret in compose: {stripped}"


def test_the_password_is_required_rather_than_defaulted():
    """A default password is a production incident waiting to happen."""
    raw = Path("docker-compose.yml").read_text()
    assert "${POSTGRES_PASSWORD:?" in raw


def test_no_secret_is_baked_into_the_dockerfile():
    for pattern in (r"ENV\s+\w*API_KEY", r"ENV\s+\w*PASSWORD", r"ARG\s+\w*SECRET", "sk-ant-"):
        assert not re.search(pattern, DOCKERFILE), f"Dockerfile contains {pattern}"


def test_the_env_file_is_excluded_from_the_build_context():
    ignored = Path(".dockerignore").read_text()
    assert ".env" in ignored
    assert "!.env.example" in ignored


# ── Image ─────────────────────────────────────────────────────────────────────


def test_the_image_runs_as_a_non_root_user():
    assert "USER analyst" in DOCKERFILE
    assert "useradd" in DOCKERFILE


def test_the_dependency_layer_is_cached_separately_from_the_code():
    """Editing a service module must not reinstall 100 packages."""
    deps_at = DOCKERFILE.index("uv sync --frozen --no-install-project")
    code_at = DOCKERFILE.index("COPY agent/")
    assert deps_at < code_at


def test_the_lockfile_is_used_so_builds_are_reproducible():
    assert "uv.lock" in DOCKERFILE
    assert "--frozen" in DOCKERFILE


def test_tests_are_not_shipped_in_the_image():
    assert "tests/" in Path(".dockerignore").read_text()
