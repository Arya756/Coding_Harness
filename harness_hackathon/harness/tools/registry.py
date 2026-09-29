"""Tool registry: schema export and dispatch with per-tool call limits."""

from __future__ import annotations

from typing import Any, Optional

from ..config import TOOL_DEFINITIONS
from . import handlers

_HANDLERS = {
    "read_file": handlers.read_file_handler,
    "grep": handlers.grep_handler,
    "run_command": handlers.run_command_handler,
    "edit_file": handlers.edit_file_handler,
    "edit_file_multi": handlers.edit_file_multi_handler,
    "run_tests": handlers.run_tests_handler,
    "list_dir": handlers.list_dir_handler,
    "get_file_info": handlers.get_file_info_handler,
    "think": handlers.think_handler,
}

_DEFAULT_TOOL_MAX = 200

_budget_tracker: dict = {}


def get_tool_schemas() -> list:
    """Return OpenAI-function-format schemas for all tools.

    Attempts schema compression via an optional third-party compressor;
    falls back to the raw (still-compact) schema when unavailable.
    """
    raw_schemas = []
    for td in TOOL_DEFINITIONS:
        raw_schemas.append({
            "name": td.name,
            "description": td.description,
            "parameters": {
                "type": "object",
                "properties": td.parameters,
                "required": td.required,
            },
        })

    try:
        from llm_prompt_refiner import SchemaCompressor  # type: ignore

        compressor = SchemaCompressor()
        compressed = compressor.compress(raw_schemas)
        if compressed:
            return compressed
    except Exception:
        pass

    return raw_schemas


def reset_budget_tracker(tool_maxes: Optional[dict] = None) -> dict:
    global _budget_tracker
    _budget_tracker = {"counts": {}, "maxes": tool_maxes or {}}
    return _budget_tracker


def execute_tool(tool_name: str, args: dict, repo_path: str, budget_tracker: Optional[dict] = None) -> dict:
    global _budget_tracker
    tracker = budget_tracker if budget_tracker is not None else _budget_tracker
    if not tracker:
        tracker = reset_budget_tracker()

    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return {
            "status": "error",
            "output": f"Unknown tool: {tool_name}",
            "tool": tool_name,
            "args": args,
        }

    counts = tracker.setdefault("counts", {})
    maxes = tracker.setdefault("maxes", {})
    max_calls = maxes.get(f"{tool_name}_max", _DEFAULT_TOOL_MAX)
    counts[tool_name] = counts.get(tool_name, 0) + 1
    if counts[tool_name] > max_calls:
        return {
            "status": "error",
            "output": f"Call limit exceeded for tool '{tool_name}' ({max_calls} max)",
            "tool": tool_name,
            "args": args,
        }

    try:
        result = handler(repo_path, args or {})
    except Exception as exc:  # noqa: BLE001 - tool handlers must never crash the loop
        result = {"status": "error", "output": f"Handler exception: {exc}"}

    result.setdefault("tool", tool_name)
    result.setdefault("args", args)
    return result
