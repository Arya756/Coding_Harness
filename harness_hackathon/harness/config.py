"""Settings, constants, tool definitions, and command guards for the harness."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 50
MAX_RETRIES_PER_FAILURE = 3
COMPACTION_TOKEN_THRESHOLD = 15000
CONTEXT_RESERVE_TOKENS = 4096
PROMPT_COMPRESS_CHAR_THRESHOLD = 8000
DEFAULT_COMMAND_TIMEOUT = 30
DEFAULT_TEST_TIMEOUT = 60
DEFAULT_BUDGET_USD: Optional[float] = None
MAX_SESSION_SECONDS = _env_int("HARNESS_MAX_SESSION_SECONDS", 1800)
CONSECUTIVE_READ_LIMIT = 5
TELEMETRY_DIR = ".harness/telemetry"
STATE_FILE = ".harness/state.json"
MEMORY_DIR = ".harness/memory"
DEFAULT_MODEL_LABEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Command guards
# ---------------------------------------------------------------------------

ALLOWED_COMMANDS = [
    "git", "npm", "yarn", "pip", "pytest", "python", "python3", "node",
    "tsc", "gofmt", "go", "cargo", "ruff", "eslint", "make", "cat", "head",
    "tail", "wc", "find", "rg", "grep", "ls", "pwd", "echo", "env", "which",
]

DENIED_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "dd if=",
    ":>",
    "mkfs",
    "shutdown",
    "reboot",
    "init 0",
    "poweroff",
    "halt",
]

DENIED_GIT_SUBCOMMANDS = [
    "reset --hard",
    "clean -f",
    "clean -fd",
    "clean -fdx",
    "push --force",
    "push -f",
    "checkout -- .",
    "checkout .",
    "rebase",
    "filter-branch",
    "update-ref -d",
]

CHAIN_METACHARS = [";", "&&", "||", "`", "$(", ">", ">>", "|"]

FREE_TIER_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemma-2-27b",
    "gemma-2-9b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "llama-3.1-8b",
}


def _prefix_allowed(first_token: str, allowed: list[str]) -> bool:
    for entry in allowed:
        if entry.endswith("*"):
            if first_token.startswith(entry[:-1]):
                return True
        elif first_token == entry:
            return True
    return False


def _is_command_allowed(cmd: str) -> tuple[bool, str]:
    """Validate a shell command against the allow/deny lists.

    Returns (allowed, reason). reason is empty when allowed.
    """
    if not cmd or not cmd.strip():
        return False, "empty command"

    if any(sep in cmd for sep in CHAIN_METACHARS):
        return False, "chaining not allowed"

    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        return False, f"unparsable command: {exc}"

    if not tokens:
        return False, "empty command"

    normalized = " ".join(tokens)
    lowered = normalized.lower()

    for bad in DENIED_COMMANDS:
        if bad in lowered:
            return False, f"denied: matches blocked pattern '{bad}'"

    first = tokens[0]

    if first == "git":
        for g in DENIED_GIT_SUBCOMMANDS:
            if g in normalized:
                return False, f"destructive git operation blocked: '{g}'"

    if not _prefix_allowed(first, ALLOWED_COMMANDS):
        return False, f"command '{first}' is not allow-listed"

    return True, ""


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    required: list
    handler: Optional[str] = None


TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description=(
            "Read a file. Use mode='outline' first, then 'signatures', then "
            "'full'. Never read full unless outline/signatures insufficient."
        ),
        parameters={
            "path": {"type": "string", "description": "File path relative to repo root"},
            "mode": {
                "type": "string",
                "enum": ["outline", "signatures", "full"],
                "default": "outline",
            },
            "start": {"type": "integer", "description": "Start line (full mode only)"},
            "end": {"type": "integer", "description": "End line (full mode only)"},
        },
        required=["path"],
        handler="handlers.read_file_handler",
    ),
    ToolDefinition(
        name="grep",
        description="Search repo for a pattern; returns file:line matches, not contents.",
        parameters={
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "options": {"type": "string"},
            "max_results": {"type": "integer", "default": 100},
        },
        required=["pattern"],
        handler="handlers.grep_handler",
    ),
    ToolDefinition(
        name="run_command",
        description=(
            "Run a whitelisted shell command; returns stdout/stderr/exit code. "
            "Chaining blocked; destructive git denied."
        ),
        parameters={
            "cmd": {"type": "string"},
            "timeout": {"type": "integer", "default": DEFAULT_COMMAND_TIMEOUT},
        },
        required=["cmd"],
        handler="handlers.run_command_handler",
    ),
    ToolDefinition(
        name="edit_file",
        description=(
            "Replace an exact string. old_str must match exactly. First "
            "occurrence only. If path doesn't exist and old_str=='' creates "
            "the file with new_str as full content."
        ),
        parameters={
            "path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        },
        required=["path", "old_str", "new_str"],
        handler="handlers.edit_file_handler",
    ),
    ToolDefinition(
        name="edit_file_multi",
        description=(
            "Apply multiple {old_str, new_str} hunks to one file in sequence "
            "(each first-occurrence). Reduces round trips for multi-hunk fixes."
        ),
        parameters={
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                    },
                },
            },
        },
        required=["path", "edits"],
        handler="handlers.edit_file_multi_handler",
    ),
    ToolDefinition(
        name="run_tests",
        description="Run the project's test suite (or subset). Returns pass/fail and output.",
        parameters={
            "pattern": {"type": "string"},
            "timeout": {"type": "integer", "default": DEFAULT_TEST_TIMEOUT},
        },
        required=[],
        handler="handlers.run_tests_handler",
    ),
    ToolDefinition(
        name="list_dir",
        description="List directory contents.",
        parameters={
            "path": {"type": "string", "default": "."},
            "recursive": {"type": "boolean", "default": False},
        },
        required=[],
        handler="handlers.list_dir_handler",
    ),
    ToolDefinition(
        name="get_file_info",
        description="Metadata: size, language, line count, top-level symbols.",
        parameters={"path": {"type": "string"}},
        required=["path"],
        handler="handlers.get_file_info_handler",
    ),
    ToolDefinition(
        name="think",
        description="Internal reasoning. Not shown to user.",
        parameters={"question": {"type": "string"}},
        required=["question"],
        handler="handlers.think_handler",
    ),
]

TOOL_NAMES = [t.name for t in TOOL_DEFINITIONS]
