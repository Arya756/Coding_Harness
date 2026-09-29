"""Live end-to-end smoke test specifically against the Groq provider."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(not os.environ.get("AI_API_KEY") or os.environ.get("HARNESS_LLM_PROVIDER") != "groq",
                     reason="requires AI_API_KEY and HARNESS_LLM_PROVIDER=groq")
def test_groq_fix_flow():
    from harness.llm import create_provider
    from harness.orchestrator import Orchestrator

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = os.path.join(tmpdir, "repo")
        os.makedirs(repo)
        with open(os.path.join(repo, "calc.py"), "w") as f:
            f.write("def add(a, b):\n    return a - b  # BUG\n")
        with open(os.path.join(repo, "test_calc.py"), "w") as f:
            f.write("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "e2e@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "E2E"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

        provider = create_provider(provider="groq")
        orch = Orchestrator(repo_path=repo, issue="calc.add() returns the wrong value", provider=provider,
                             max_steps=20)
        result = orch.run()
        assert result["status"] in ("resolved", "failed", "stuck")
