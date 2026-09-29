# Autonomous Coding-Agent Harness — Final Build Specification (PRD v3)

**Purpose:** This is a complete, self-contained build specification for the
`harness_hackathon/` project. It combines the original architecture with
corrected bugs, hardened security, and wired-up features.

**Status of this version:** All known bugs from prior versions are specified
as required correct behavior here. Features that were previously "defined but
unused" are required to be wired in, and are (see `harness/` for the
implementation): verification is called from the edit path, checkpoint/
rollback is implemented, context compaction is real, cross-session memory is
used, and escalation is reachable.

## 1. Project Overview

- Name: Autonomous Coding-Agent Harness
- Language: Python 3.10+
- Type: Autonomous software-engineering agent
- Root path: `harness_hackathon/`

Given a repository and an issue description, the harness reads repo structure
surgically, generates a validated step-by-step plan, navigates the codebase,
edits with checkpoint/rollback, verifies each edit (syntax -> lint -> tests),
recovers from failures, and only commits when tests actually pass, producing a
telemetry report.

It is model-agnostic (OpenAI, OpenRouter, Ollama, Groq, DeepSeek, Qwen,
Gemini, or any OpenAI-compatible endpoint) via environment variables only.

Tool calls are parsed from the LLM's text response (JSON extraction + regex +
inference fallbacks), not solely via provider function-calling, to maximize
model compatibility; native `tool_calls` are still used as the first-priority
parse path when a provider returns them.

## 2. Guiding Principles

1. Never claim success without proof — a commit only happens after a real,
   passing test run (or an explicit "no tests exist" acknowledgment).
2. Every edit is reversible — file contents are snapshotted before editing,
   with rollback on regression or syntax failure.
3. Fail safe, not silent — classification, telemetry, and cost accounting
   must be correct even when "harmless if wrong".
4. Contain the blast radius — shell execution is allow-listed, chaining is
   blocked, destructive git operations are denied, and secrets are stripped
   from subprocess environments.
5. Token frugality is a feature — outlines over full reads, a read-cache to
   avoid re-reading unchanged files, and real (not no-op) context compaction.
6. Deterministic testability — core logic is unit-testable with a
   `FakeLLMProvider`; `make test` requires no live API access.

## 3. High-Level Architecture

```
CLI (main.py)
    |
    |-- create_provider() -> LLMProvider (llm.py / llm_gemini.py)
    |
    +-- Orchestrator (orchestrator.py)
            |
            |-- RepoNavigator (navigator/repo_navigator.py)
            |-- ContextManager (context/manager.py)
            |-- TelemetryCollector + TokenTracker (telemetry/collector.py)
            |-- RecoveryEngine (recovery/engine.py)
            |-- Verifier (verification/verifier.py)   <-- wired into edit path
            |-- Reporter (telemetry/reporter.py)
            |-- Checkpoint/Rollback (in orchestrator)
            |
            +-- Tool Dispatch (tools/registry.py -> tools/handlers.py)
                    read_file | grep | run_command | edit_file
                    edit_file_multi | run_tests | list_dir
                    get_file_info | think
```

## 4–19. Module Contracts

The full module-by-module specification (config constants and command guards,
protocol types and parsing, task state and persistence, the orchestrator main
loop, the LLM provider layer and pricing table, the repo navigator, the nine
tool definitions and their handlers, the tool registry, the context manager,
the recovery engine and failure taxonomy, telemetry collection, the ASCII
reporter, and the syntax/lint verifier) is implemented under `harness/` with
one module per concern, matching the directory map below.

```
harness_hackathon/
|-- harness/
|   |-- __init__.py
|   |-- main.py
|   |-- config.py
|   |-- models.py
|   |-- state.py
|   |-- orchestrator.py
|   |-- llm.py
|   |-- llm_gemini.py
|   |-- navigator/repo_navigator.py
|   |-- tools/{registry.py, handlers.py}
|   |-- context/manager.py
|   |-- recovery/engine.py
|   |-- telemetry/{collector.py, reporter.py}
|   +-- verification/verifier.py
|-- config/evaluation.json
|-- tests/ (+ tests/e2e/)
|-- .env.example
|-- Makefile
|-- requirements.txt
+-- README.md
```

Key required-correct behaviors baked into this implementation:

- `classify_failure` returns `NONE` on `status == "success"` (not `UNKNOWN`).
- `edit_file` creates a new file when the path doesn't exist and
  `old_str == ""`.
- `edit_file_multi` applies multiple hunks atomically (all-or-nothing).
- `run_command` blocks command chaining (`;`, `&&`, `||`, backticks, `$( )`,
  redirects, pipes), denies destructive `git` subcommands even though `git`
  itself is allow-listed, and strips `*_API_KEY`/`*_TOKEN`/`*_SECRET` from the
  subprocess environment.
- `run_tests` returns `{"passed": true, "no_tests": true}` when no test
  framework is detected, so test-less repos can still be legitimately
  resolved (and this is recorded in telemetry/report).
- Every edit is checkpointed before being applied; `verify_edit` (syntax then
  lint) runs immediately after; syntax failures and test regressions trigger
  automatic rollback via the recovery engine's `rollback` strategy.
- `done` is gated on a real, current test run — it never trusts the model's
  claim, and refuses to commit (feeding the failure back into context) until
  tests pass.
- Context compaction is real: after 10+ tool results accumulate past the
  token threshold, old results are collapsed into a textual digest and only
  the most recent 10 are kept verbatim; a `compaction` telemetry event
  records `saved_tokens`.
- A force-edit nudge is injected after 5 consecutive read-like tool calls.
- A per-provider shared retry helper (`LLMProvider._with_retry`) backs off
  exponentially on 429/500/502/503/504 and timeouts; the Groq-only
  `max_tokens<=900` cap lives solely in `GroqProvider`.
- Budget and wall-clock session caps (`--budget`, `--max-seconds`) abort the
  loop cleanly with `failure_reason` set to `"budget"` or `"session_timeout"`.
- Escalation is reachable: once a failure signature persists past the normal
  retry/replan cycle, the engine writes `.harness/needs_human.md` and returns
  the `escalate` action.

## 20–27. CLI, Env Vars, Testing, Dependencies, Acceptance Criteria

See `README.md` for CLI usage and `tests/` for the required coverage
(handlers, parsing, recovery, orchestrator, context). `tests/e2e/` holds
live-API scripts that are excluded from `make test` via the `e2e` pytest
marker and only run via `make test-e2e`.

Acceptance criteria (definition of done) are enforced by the test suite in
`tests/`:

1. `make setup && make test` passes with no live API calls.
2. A known-bug fixture repo is only committed after tests genuinely pass.
3. A model claiming "done" while tests fail never produces a commit.
4. A syntax-breaking edit is caught by `verify_edit` and rolled back.
5. An edit that regresses a previously-passing test is rolled back.
6. `run_command` rejects destructive git, `rm -rf /`, chained commands, and
   non-allow-listed binaries; subprocess env excludes secrets.
7. Switching providers is purely an environment-variable change.
8. Transient 503s are retried with backoff.
9. Large sessions measurably shrink their token estimate via compaction.
10. Telemetry cost accounting includes the initial plan call.
11. Budget-limited runs abort with `failure_reason == "budget"`.
12. Sessions exceeding `--max-seconds` abort with `status in {"stuck",
    "failed"}` and `failure_reason == "session_timeout"`.
