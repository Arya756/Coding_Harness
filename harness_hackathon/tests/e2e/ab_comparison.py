"""A/B comparison harness for two providers/models on the same issue.

Not part of the automated test suite; run manually:
    python tests/e2e/ab_comparison.py --repo /path/to/repo --issue "..." \
        --model-a openai:gpt-4o-mini --model-b groq:llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.llm import create_provider  # noqa: E402
from harness.orchestrator import Orchestrator  # noqa: E402


def _run_one(spec: str, repo: str, issue: str, max_steps: int) -> dict:
    provider_name, model = spec.split(":", 1)
    provider = create_provider(provider=provider_name, model=model)
    orch = Orchestrator(repo_path=repo, issue=issue, provider=provider, max_steps=max_steps)
    result = orch.run()
    return {
        "spec": spec,
        "status": result["status"],
        "summary": result["summary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    result_a = _run_one(args.model_a, args.repo, args.issue, args.max_steps)
    result_b = _run_one(args.model_b, args.repo, args.issue, args.max_steps)

    print("=== A ===")
    print(result_a)
    print("=== B ===")
    print(result_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
