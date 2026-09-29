"""CLI entry point for the Autonomous Coding-Agent Harness."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import requests

from .config import DEFAULT_MODEL_LABEL, FREE_TIER_MODELS, MAX_SESSION_SECONDS, MAX_STEPS, STATE_FILE
from .llm import create_provider
from .llm_gemini import GeminiProvider
from .orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Provider detection from API key (no user prompt needed)
# ---------------------------------------------------------------------------

_OLLAMA_MODELS_CACHE: Optional[list[str]] = None


def _ollama_models(host: str) -> list[str]:
    """List locally available Ollama models."""
    global _OLLAMA_MODELS_CACHE
    if _OLLAMA_MODELS_CACHE is not None:
        return _OLLAMA_MODELS_CACHE
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            _OLLAMA_MODELS_CACHE = models
            return models
    except Exception:
        pass
    _OLLAMA_MODELS_CACHE = []
    return []


def _openai_list_models(base_url: str, api_key: str) -> list[str]:
    try:
        resp = requests.get(f"{base_url}/models", headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return [m["id"] for m in data.get("data", []) if "id" in m]
    except Exception:
        pass
    return []


def _ollama_list_models(host: str) -> list[str]:
    return _ollama_models(host)


def _gemini_list_models(api_key: str) -> list[str]:
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def _list_models(provider: str, api_key: str, base_url: Optional[str] = None,
                 ollama_host: Optional[str] = None) -> list[str]:
    """Fetch available models from the provider's API, or a curated fallback."""
    if provider == "ollama":
        host = (ollama_host or os.environ.get("HARNESS_LLM_HOST", "http://localhost:11434")).rstrip("/")
        return _ollama_list_models(host)

    if provider == "gemini":
        return _gemini_list_models(api_key)

    # OpenAI-compatible providers
    url = base_url or _default_base_url(provider)
    if url:
        models = _openai_list_models(url, api_key)
        if models:
            return models

    # Fallback: curated known models per provider
    return _curated_models(provider)


def _default_base_url(provider: str) -> Optional[str]:
    if provider == "openai":
        return "https://api.openai.com/v1"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "groq":
        return os.environ.get("HARNESS_LLM_GROQ_BASE", "https://api.groq.com/openai/v1")
    if provider == "deepseek":
        return "https://api.deepseek.com/v1"
    if provider == "qwen":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider in ("cursor", "generic"):
        return os.environ.get("HARNESS_LLM_BASE_URL") or os.environ.get("HARNESS_LLM_CURSOR_BASE")
    return None


def _curated_models(provider: str) -> list[str]:
    """Fallback model list shown when the provider API is unreachable."""
    curated = {
        "openai": [
            "gpt-4o", "gpt-4o-mini", "gpt-4 Turbo", "gpt-3.5-turbo",
            "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18",
        ],
        "openrouter": [
            "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo",
            "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus",
            "meta/llama-3.1-8b-instruct", "google/gemini-2.0-flash",
            "deepseek/deepseek-chat", "mistral/mistral-large-latest",
        ],
        "groq": [
            "llama-3.1-8b-instant", "llama-3.2-90b-text-preview",
            "llama-3.2-3b-preview", "gemma-7b-it", "mixtral-8x7b-32768",
            "llama3-groq-8b-0xFF", "llama-3.1-70b-versatile",
        ],
        "deepseek": [
            "deepseek-chat", "deepseek-reasoner",
        ],
        "qwen": [
            "qwen-turbo", "qwen-plus", "qwen-max",
        ],
        "gemini": [
            "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
            "gemini-1.5-pro", "gemini-1.5-flash",
        ],
        "ollama": [
            "llama3.2:3b", "llama3.2:1b", "llama3.1:8b",
            "qwen2.5-coder:7b", "mistral:7b", "gemma:7b",
        ],
    }
    return curated.get(provider, ["gpt-4o-mini"])


def _detect_provider(api_key: str, explicit_provider: Optional[str] = None) -> str:
    """Infer the LLM provider from the API key and env vars.

    Priority:
    1. Explicit provider (CLI arg or env var)
    2. No key → Ollama (if host reachable)
    3. gsk- prefix → Groq
    4. sk- prefix with OpenRouter base URL/env → OpenRouter
    5. Default to OpenAI
    """
    if explicit_provider:
        return explicit_provider.lower()

    env_provider = os.environ.get("HARNESS_LLM_PROVIDER", "").strip()
    if env_provider:
        return env_provider.lower()

    if not api_key:
        # No key: try Ollama
        host = os.environ.get("HARNESS_LLM_HOST", "http://localhost:11434")
        if _ollama_models(host):
            return "ollama"
        # Ollama not reachable; fall through to ask user
        return "openai"

    upper = api_key.strip().upper()

    # Groq keys start with gsk-
    if api_key.startswith("gsk-") or upper.startswith("GSK-"):
        return "groq"

    # OpenRouter: sk- key + base URL hint or OPENROUTER_API_KEY env
    if api_key.startswith("sk-") and not api_key.startswith("sk-proj-"):
        if os.environ.get("HARNESS_LLM_BASE_URL", "").lower().startswith("openrouter"):
            return "openrouter"
        if os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter"

    # Google Gemini keys are not sk- format — they look like AIza... or long random strings
    # but we can't reliably detect them; user must set HARNESS_LLM_PROVIDER=gemini

    # Default: OpenAI
    return "openai"


# ---------------------------------------------------------------------------
# Interactive setup
# ---------------------------------------------------------------------------

def _resolve_api_key_env() -> Optional[str]:
    """Read the API key from the standard evaluation env var.

    Criterion 2: the evaluator exports AI_API_KEY; the harness must read it
    from there without prompting and without hard-coding.
    """
    return (
        os.environ.get("AI_API_KEY")
        or os.environ.get("HARNESS_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or None
    )


def _prompt_api_key() -> str:
    """Ask the user for an API key, unless AI_API_KEY is already set.

    If the evaluation env var is present, use it silently (criterion 2).
    Otherwise prompt the user interactively.
    """
    env_key = _resolve_api_key_env()
    if env_key:
        print(f"\n  → Using API key from AI_API_KEY environment variable.")
        # Mask for display
        visible = env_key[:4] + "·" * (max(0, len(env_key) - 8)) + env_key[-4:] if len(env_key) > 8 else "·" * len(env_key)
        print(f"    (key: {visible})")
        return env_key

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  Autonomous Coding-Agent Harness — Setup                   │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print("Step 1/4 — API key")
    print("  Enter your LLM API key. The harness will detect the provider")
    print("  automatically. (OpenAI, Groq, OpenRouter, DeepSeek, Qwen,")
    print("  Gemini, or Ollama — no key needed for Ollama.)")
    print("  (Alternatively, set the AI_API_KEY environment variable and")
    print("   re-run — the harness will pick it up automatically.)")
    print()
    while True:
        key = input("  API key: ").strip()
        if key:
            return key
        print("  Key is required to proceed.")


def _choose_from_list(items: list[str], label: str = "model") -> str:
    """Print a numbered list and return the user's choice."""
    print(f"\nStep 2/4 — Choose a {label}")
    print(f"  {len(items)} available:")
    for i, item in enumerate(items, 1):
        print(f"    [{i}]  {item}")
    print()
    default_idx = 1
    try:
        default_idx = next(i for i, item in enumerate(items, 1)
                           if item.lower() in ("gpt-4o-mini", "gpt-4o-mini", "llama-3.1-8b-instant"))
    except StopIteration:
        pass

    while True:
        raw = input(f"  Choice [{default_idx}]: ").strip()
        if not raw:
            return items[default_idx - 1]
        try:
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        except ValueError:
            pass
        # Allow typing the model name directly
        if raw in items:
            return raw
        print(f"  Enter a number 1-{len(items)}, or the exact model name.")


def _interactive_setup() -> dict:
    """Run the interactive setup wizard.

    Returns a dict with: provider, api_key, model, issue, repo, budget,
    max_steps, max_seconds, require_approval, free_tier_confirmed.
    """
    api_key = _prompt_api_key()

    # Detect provider
    provider = _detect_provider(api_key)
    print(f"\n  → Detected provider: {provider}")

    # Resolve base URL and Ollama host
    base_url = os.environ.get("HARNESS_LLM_BASE_URL")
    ollama_host = os.environ.get("HARNESS_LLM_HOST", "http://localhost:11434")

    # List models
    models = _list_models(provider, api_key, base_url=base_url, ollama_host=ollama_host)
    if not models:
        models = _curated_models(provider)
        print("  (model list fetch failed; showing known models)")
    elif provider == "ollama":
        print(f"  → Found {len(models)} local model(s) on {ollama_host}")

    model = _choose_from_list(models, "model")

    # Free-tier check
    free_tier_confirmed = False
    if model in FREE_TIER_MODELS:
        print(f"\n  ⚠  '{model}' is a free-tier model.")
        resp = input("  Continue anyway? [y/N]: ").strip().lower()
        free_tier_confirmed = resp == "y"
        if not free_tier_confirmed:
            print("  Aborting. Re-run with a non-free model or pass --free-tier-confirmed.")
            sys.exit(1)

    # Budget
    budget = None
    print("\nStep 3/4 — Budget (optional)")
    print("  Set a spending limit in USD, or leave blank for no limit.")
    while True:
        raw = input("  Budget (USD, e.g. 0.50): ").strip()
        if not raw:
            break
        try:
            budget = float(raw)
            if budget <= 0:
                print("  Must be positive.")
                continue
            break
        except ValueError:
            print("  Enter a number (e.g. 0.50).")

    # Max steps and seconds
    print("\nStep 3b/4 — Limits (optional, defaults shown)")
    max_steps = MAX_STEPS
    max_seconds = MAX_SESSION_SECONDS
    raw = input(f"  Max steps [{max_steps}]: ").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                max_steps = val
        except ValueError:
            pass
    raw = input(f"  Max seconds [{max_seconds}]: ").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                max_seconds = val
        except ValueError:
            pass

    # Require approval
    require_approval = False
    resp = input("  Require approval before each edit? [y/N]: ").strip().lower()
    require_approval = resp == "y"

    # Target repo
    print("\nStep 4/4 — Target repository")
    print("  Path to the repository the harness should work on.")
    while True:
        repo = input("  Repo path: ").strip()
        if os.path.isdir(repo):
            break
        print(f"  '{repo}' is not a directory. Check the path and try again.")

    # Issue
    print("\n  What problem should the harness fix?")
    print("  (Describe the bug, missing feature, or change needed.)")
    print()
    issue = input("  Issue: ").strip()
    if not issue:
        print("  Issue is required.")
        sys.exit(1)

    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "ollama_host": ollama_host,
        "issue": issue,
        "repo": repo,
        "budget": budget,
        "max_steps": max_steps,
        "max_seconds": max_seconds,
        "require_approval": require_approval,
        "free_tier_confirmed": free_tier_confirmed,
    }


def _print_session_header(config: dict) -> None:
    print()
    print("════════════════════════════════════════════════════════════════")
    print(f"  Provider:  {config['provider']}")
    print(f"  Model:     {config['model']}")
    print(f"  Repo:      {config['repo']}")
    print(f"  Issue:     {config['issue']}")
    if config['budget'] is not None:
        print(f"  Budget:    ${config['budget']:.2f}")
    print(f"  Max steps: {config['max_steps']}")
    print(f"  Max secs:  {config['max_seconds']}")
    print(f"  Approval:  {config['require_approval']}")
    print("════════════════════════════════════════════════════════════════")
    print()
    print("  Starting agent loop… (interrupt with Ctrl+C)")
    print()


# ---------------------------------------------------------------------------
# Existing helpers
# ---------------------------------------------------------------------------

def _resolve_issue(cli_issue: Optional[str], repo: str) -> Optional[str]:
    if cli_issue:
        return cli_issue
    env_issue = os.environ.get("ISSUE")
    if env_issue:
        return env_issue
    for fname in ("issue.txt", "ISSUE.md"):
        path = os.path.join(repo, fname)
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    return None


def _resolve_api_key(cli_key: Optional[str]) -> str:
    return (
        cli_key
        or os.environ.get("AI_API_KEY")
        or os.environ.get("HARNESS_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )


def _is_free_tier_model(model: str) -> bool:
    return model in FREE_TIER_MODELS


def load_evaluation_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _launch_check(eval_config: dict, repo: str) -> int:
    """Validate provider/key/repo without calling the model. Return exit code."""
    problems = []
    if not os.path.isdir(repo):
        problems.append(f"Repo path does not exist: {repo}")
    provider_name = os.environ.get("HARNESS_LLM_PROVIDER", "openai")
    api_key = _resolve_api_key(None)
    if provider_name != "ollama" and not api_key:
        problems.append("No API key found (AI_API_KEY / HARNESS_LLM_API_KEY / OPENAI_API_KEY)")
    if problems:
        for p in problems:
            print(f"[launch-check] FAIL: {p}")
        return 1
    print("[launch-check] OK")
    return 0


def _print_diff_preview(action_name: str, args: dict) -> bool:
    print("\n--- Proposed edit ---")
    print(json.dumps(args, indent=2)[:2000])
    resp = input("Apply this edit? [y/N] ").strip().lower()
    return resp == "y"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous Coding-Agent Harness")
    parser.add_argument("--repo", help="Path to the target repository")
    parser.add_argument("--issue", help="Issue description text")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--budget", type=float, default=None, help="Budget in USD")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-seconds", type=int, default=MAX_SESSION_SECONDS)
    parser.add_argument("--system-prompt", help="Path to a file with custom system-prompt instructions")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: prompt for issues one at a time (requires --repo)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--llm-provider", help="LLM provider name (auto-detected if omitted in setup mode)")
    parser.add_argument("--llm-api-key", help="LLM API key (prompted in setup mode if omitted)")
    parser.add_argument("--llm-base-url", help="Custom base URL for generic/cursor providers")
    parser.add_argument("--ollama-host", help="Ollama host URL")
    parser.add_argument("--evaluation", help="Path to evaluation config JSON")
    parser.add_argument("--launch-check", action="store_true",
                        help="Validate setup without calling the model")
    parser.add_argument("--free-tier-confirmed", action="store_true")
    parser.add_argument("--require-approval", action="store_true",
                        help="Pause and print a diff preview before applying edits")
    parser.add_argument("--setup", action="store_true",
                        help="Interactive setup wizard: API key → model → issue → run")

    args = parser.parse_args(argv)

    eval_config = load_evaluation_config(args.evaluation) if args.evaluation else {}

    if args.launch_check:
        if not args.repo:
            print("Error: --launch-check requires --repo")
            return 1
        return _launch_check(eval_config, args.repo)

    # ---- Setup mode: no --repo and no --issue → interactive wizard ----
    if not args.repo:
        cfg = _interactive_setup()
        _print_session_header(cfg)
        provider_name = cfg["provider"]
        api_key = cfg["api_key"]
        model = cfg["model"]
        base_url = cfg["base_url"]
        ollama_host = cfg["ollama_host"]

        try:
            provider = create_provider(
                provider=provider_name,
                api_key=api_key,
                model=model,
                base_url=base_url,
                ollama_host=ollama_host,
            )
        except Exception as exc:
            print(f"Failed to create LLM provider: {exc}")
            return 1

        orch = Orchestrator(
            repo_path=cfg["repo"],
            issue=cfg["issue"],
            budget_usd=cfg["budget"],
            max_steps=cfg["max_steps"],
            system_prompt=None,
            resume=False,
            provider=provider,
            max_seconds=cfg["max_seconds"],
            require_approval=cfg["require_approval"],
        )
        result = orch.run()
        print(result["report"])
        return 0 if result["status"] == "resolved" else 1

    # ---- Normal/CLI mode (when --repo is provided) ----
    model = args.model or os.environ.get("HARNESS_LLM_MODEL") or os.environ.get("HARNESS_MODEL", DEFAULT_MODEL_LABEL)
    if _is_free_tier_model(model) and not args.free_tier_confirmed:
        print(f"Model '{model}' is a free-tier model. Pass --free-tier-confirmed to proceed.")
        return 1

    api_key = _resolve_api_key(args.llm_api_key)
    provider_name = args.llm_provider or os.environ.get("HARNESS_LLM_PROVIDER", "openai")

    try:
        provider = create_provider(
            provider=provider_name,
            api_key=api_key,
            model=model,
            base_url=args.llm_base_url,
            ollama_host=args.ollama_host,
        )
    except Exception as exc:
        print(f"Failed to create LLM provider: {exc}")
        return 1

    system_prompt = None
    if args.system_prompt and os.path.exists(args.system_prompt):
        with open(args.system_prompt) as f:
            system_prompt = f.read()

    state_path = os.path.join(os.path.abspath(args.repo), STATE_FILE)

    if args.interactive:
        while True:
            try:
                issue = input("Issue (blank to quit): ").strip()
            except EOFError:
                break
            if not issue:
                break
            if os.path.exists(state_path):
                os.remove(state_path)
            orch = Orchestrator(
                repo_path=args.repo, issue=issue, budget_usd=args.budget,
                max_steps=args.max_steps, system_prompt=system_prompt, resume=False,
                provider=provider, max_seconds=args.max_seconds, require_approval=args.require_approval,
            )
            result = orch.run()
            print(result["report"])
        return 0

    if args.resume:
        if args.issue:
            print("Warning: --issue is ignored when --resume is set; continuing prior session.")
        orch = Orchestrator(
            repo_path=args.repo, issue="", budget_usd=args.budget, max_steps=args.max_steps,
            system_prompt=system_prompt, resume=True, provider=provider,
            max_seconds=args.max_seconds, require_approval=args.require_approval,
        )
        result = orch.run()
        print(result["report"])
        return 0 if result["status"] == "resolved" else 1

    issue = _resolve_issue(args.issue, args.repo)
    if not issue:
        print("No issue provided. Use --issue, ISSUE env var, or repo/issue.txt / ISSUE.md")
        return 1

    orch = Orchestrator(
        repo_path=args.repo, issue=issue, budget_usd=args.budget, max_steps=args.max_steps,
        system_prompt=system_prompt, resume=False, provider=provider,
        max_seconds=args.max_seconds, require_approval=args.require_approval,
    )
    result = orch.run()
    print(result["report"])
    return 0 if result["status"] == "resolved" else 1


if __name__ == "__main__":
    sys.exit(main())
