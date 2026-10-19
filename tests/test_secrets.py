"""No credential is hardcoded anywhere in the source tree. Synchronous by design."""

from pathlib import Path

import pytest

SOURCE_DIRS = ["agent", "api", "frontend", "models", "servers", "services"]

# Placeholders that are allowed to appear because they are obviously not real.
ALLOWED = ("change-me", "REDACTED", "your-key", "test-key", "example")


@pytest.mark.parametrize("directory", SOURCE_DIRS)
def test_no_hardcoded_credentials(directory: str):
    for path in sorted(Path(directory).rglob("*.py")):
        source = path.read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or any(a in stripped for a in ALLOWED):
                continue  # comments and obvious placeholders are not credentials
            for marker in ("postgresql://", "postgres://", "redis://"):
                if marker in stripped and "@" in stripped:
                    raise AssertionError(f"{path}: possible hardcoded credential: {stripped}")


@pytest.mark.parametrize("directory", SOURCE_DIRS)
def test_no_api_keys_are_embedded(directory: str):
    """Every key must arrive through services/config.py, never as a literal."""
    for path in sorted(Path(directory).rglob("*.py")):
        source = path.read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or any(a in stripped for a in ALLOWED):
                continue
            for prefix in ("sk-ant-", "sk-live-", "sk_test_"):
                assert prefix not in stripped, f"{path}: embedded key: {stripped}"


def test_env_file_is_git_ignored():
    ignored = Path(".gitignore").read_text()
    assert ".env" in ignored
    assert "!.env.example" in ignored
