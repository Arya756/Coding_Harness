# Autonomous Coding-Agent Harness

A model-agnostic, autonomous software-engineering agent. Give it a repository
and an issue description and it will plan, navigate the codebase, make
exact-string edits, verify them (syntax → lint → tests), recover from
failures, and only commit once tests actually pass.

Built strictly to the [Build Specification (PRD v3)](harness_hackathon/PRD.md).

## Quick start (evaluator workflow)

```bash
git clone <this-repo>
cd <this-repo>
export AI_API_KEY="<provided-key>"
make setup
make run
```

`make run` enters an interactive wizard that asks for:
1. **Model** — auto-detected from your API key, or pick from the list
2. **Budget** (optional) — spending limit in USD
3. **Target repo** — path to the repository the harness should work on
4. **Issue** — the problem the harness should fix

After setup, the harness runs the full loop: plan → navigate → edit →
verify → test → commit → report. Artifacts appear in the target repo under
`.harness/` (state, telemetry, report).

## Makefile commands

| Command | Purpose |
|---|---|
| `make setup` | Install Python deps + nltk data |
| `make run` | Launch the harness in interactive setup mode |
| `make test` | Run unit/integration tests (no live API calls) |
| `make clean` | Remove generated artefacts |

## API key

The harness reads the API key from the `AI_API_KEY` environment variable.
**Do not hard-code keys in source code, the Makefile, .env files, or docs.**

For local development you may create a `.env` file from the example:

```bash
cp harness_hackathon/.env.example harness_hackathon/.env
# then fill in AI_API_KEY=...
```

The `.env` file is git-ignored and must never contain a real credential.

## Supported providers

OpenAI, OpenRouter, Ollama, Groq, DeepSeek, Qwen, Gemini (native v1beta),
or any OpenAI-compatible endpoint via `generic`/`cursor`. Switching providers
requires only environment variables — no code changes.

## Model requirement

The harness uses **text-only** language models. It accepts text input (repo
content + issue description) and produces text output (plan, edits, reports).
No image, audio, video, or multimodal processing is required or used.

## Project layout

```
autonomous-coding-agent-specification/
├── Makefile               ← REQUIRED — standard evaluation interface
├── README.md              ← this file
├── .env.example           ← env var template (NO real credentials)
└── harness_hackathon/     ← source code
    ├── harness/           ← agent core (main, orchestrator, llm, tools, …)
    ├── tests/             ← unit + integration tests
    ├── requirements.txt   ← Python dependencies
    ├── Makefile           ← harness-level convenience targets
    ├── .env.example       ← env var template
    └── PRD.md             ← full build specification
```

## Configuring the model

Set these environment variables before running `make run`:

```bash
export HARNESS_LLM_PROVIDER=openai      # openai | openrouter | ollama | groq | deepseek | qwen | gemini
export HARNESS_LLM_MODEL=gpt-4o-mini     # any valid model for the chosen provider
export HARNESS_LLM_BASE_URL=            # optional: custom base URL for generic/cursor providers
export HARNESS_LLM_HOST=http://localhost:11434   # Ollama host (only for ollama provider)
```

For Ollama (no API key needed):

```bash
export HARNESS_LLM_PROVIDER=ollama
export HARNESS_LLM_MODEL=llama3.2:3b
export HARNESS_LLM_HOST=http://localhost:11434
make run
```

## Running tests

```bash
make test          # unit + integration tests, no live API calls
cd harness_hackathon && make test-e2e   # live-API smoke tests (needs AI_API_KEY)
```

## Interactive mode (manual use)

When you run `make run` or `python -m harness.main` without `--repo`, the
interactive setup wizard guides you through the full configuration. You can
also use the CLI directly for scripted runs:

```bash
cd harness_hackathon
python -m harness.main \
  --repo /path/to/target/repo \
  --issue "fix: the login function returns None for valid users" \
  --max-steps 40 \
  --budget 0.50
```
