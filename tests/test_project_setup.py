"""Phase 1 acceptance checks: structure, importability, and no committed secrets."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "servers/market_mcp",
    "servers/financial_mcp",
    "servers/news_mcp",
    "servers/calculator_mcp",
    "servers/research_mcp",
    "agent",
    "api",
    "frontend",
    "services",
    "models",
    "tests",
    "docs",
]

REQUIRED_FILES = [".env.example", ".gitignore", "pyproject.toml", "README.md"]


@pytest.mark.parametrize("rel", REQUIRED_DIRS)
def test_required_directory_exists(rel: str) -> None:
    assert (ROOT / rel).is_dir(), f"missing directory: {rel}"


@pytest.mark.parametrize("rel", REQUIRED_FILES)
def test_required_file_exists(rel: str) -> None:
    assert (ROOT / rel).is_file(), f"missing file: {rel}"


def test_env_file_is_not_committed() -> None:
    assert not (ROOT / ".env").exists() or ".env" in (ROOT / ".gitignore").read_text()


def test_env_example_has_no_filled_secrets() -> None:
    """Every key-ish variable in .env.example must be left blank."""
    secretish = ("api_key", "token", "secret")
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if any(s in name.lower() for s in secretish):
            assert value == "", f"{name} must be blank in .env.example"


def test_settings_import_and_defaults() -> None:
    from services.config import get_settings

    settings = get_settings()
    assert settings.llm_model
    assert settings.cache_ttl_seconds >= 0


def test_packages_import_cleanly() -> None:
    import importlib

    for module in (
        "agent",
        "api",
        "models",
        "services",
        "servers.calculator_mcp",
        "servers.market_mcp",
        "servers.financial_mcp",
        "servers.news_mcp",
        "servers.research_mcp",
    ):
        importlib.import_module(module)
