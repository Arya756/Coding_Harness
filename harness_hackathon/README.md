# Autonomous Coding-Agent Harness

A model-agnostic, autonomous software-engineering agent. Give it a repository
and an issue description and it will plan, navigate the codebase, make
exact-string edits, verify them (syntax -> lint -> tests), recover from
failures, and only commit once tests actually pass.

Built strictly to `PRD.md` (Build Specification v3).

## Features

- **Model-agnostic**: OpenAI, OpenRouter, Ollama, Groq, DeepSeek, Qwen, Gemini
  (native v1beta), or any OpenAI-compatible endpoint via `generic`/`cursor`.
  Switching providers requires only environment variables.
- **Text-in/text-out tool calling**: parses JSON actions from free-form model
  text (with native `tool_calls` support as first-priority path), so it works
  even with models that don't support function calling.
- **Token-efficient navigation**: outline -> signatures -> full read escalation,
  with a per-session read-cache to avoid redundant re-reads.
- **Safe edits**: exact-string `edit_file` / multi-hunk `edit_file_multi`,
  checkpointed before every edit, verified (syntax then lint) and rolled back
  automatically on syntax errors or test regressions.
- **Hardened shell execution**: allow-listed commands, no command chaining,
  destructive `git` operations blocked, secrets stripped from the subprocess
  environment.
- **Real recovery loop**: classifies failures (test failure, command error,
  edit failure, regression, syntax error, stuck, etc.) and retries, replans,
  rolls back, compacts context, or escalates to a `needs_human.md` file.
- **Gated commits**: `done` is refused until a real test run passes (or the
  repo genuinely has no test suite, which is recorded transparently).
- **Real context compaction** once token estimates cross a threshold, plus
  minimal cross-session memory of prior attempts on the same issue.
- **Full telemetry**: JSONL event log + an ASCII summary report per run.

## Install

```bash
cd harness_hackathon
make setup
cp .env.example .env   # fill in AI_API_KEY etc.
```

## Run

```bash
python harness/main.py --repo /path/to/target/repo --issue "Fix the login bug"
```

Or drop an `issue.txt` / `ISSUE.md` file in the target repo and omit `--issue`.

Useful flags: `--budget 0.50`, `--max-steps 40`, `--max-seconds 900`,
`--resume`, `--interactive`, `--require-approval`, `--llm-provider groq`,
`--model llama-3.1-8b-instant`.

## Test

```bash
make test       # unit/integration tests with a FakeLLMProvider, no live API calls
make test-e2e   # live-API smoke tests (needs real credentials)
```

## Layout

See `PRD.md` for the full module-by-module specification. Source lives under
`harness/`, tests under `tests/`, and per-run artifacts are written to the
*target* repo under `.harness/` (`state.json`, `telemetry/events.jsonl`,
`report.txt`, `memory/decisions.jsonl`, and `needs_human.md` on escalation).
