"""Shared pytest fixtures: FakeLLMProvider and temp-repo builders."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.llm import LLMProvider, LlmResponse  # noqa: E402


class FakeLLMProvider(LLMProvider):
    """Scripted provider for deterministic tests -- no live API calls."""

    def __init__(self, responses: list, model: str = "fake-model"):
        self._responses = list(responses)
        self._idx = 0
        self._model = model
        self.calls = []

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages, max_tokens=4096, temperature=0.7, tools=None) -> LlmResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if self._idx < len(self._responses):
            item = self._responses[self._idx]
            self._idx += 1
        else:
            item = self._responses[-1] if self._responses else '{"name": "think", "arguments": {"question": "done"}}'

        if isinstance(item, dict):
            text = item.get("text", "")
            tool_calls = item.get("tool_calls", [])
        else:
            text = item
            tool_calls = []

        return LlmResponse(
            text=text, input_tokens=100, output_tokens=50, cost_usd=0.0001,
            model=self._model, tool_calls=tool_calls,
        )


@pytest.fixture
def tmp_repo(tmp_path):
    """A minimal python repo with a bug and a failing test, plus git init."""
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "calc.py").write_text(
        "def add(a, b):\n"
        "    return a - b  # BUG: should be +\n"
    )
    (repo / "test_calc.py").write_text(
        "from calc import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    return str(repo)


@pytest.fixture
def fake_provider():
    return FakeLLMProvider(responses=[])
