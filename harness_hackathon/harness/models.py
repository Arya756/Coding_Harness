"""Protocol types, error taxonomy, and parsing helpers."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import TOOL_NAMES


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ModelError(Exception):
    """Base error for LLM provider failures."""


class ModelHTTPError(ModelError):
    def __init__(self, status: int, body: str = ""):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:300]}")


class ProtocolError(ModelError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class ServiceUnavailableError(ModelError):
    pass


# ---------------------------------------------------------------------------
# Action (pydantic-validated)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict = Field(default_factory=dict)
    goal: str = ""

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_args(cls, v: Any) -> dict:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {"_raw": v}
        return {"_raw": v}


class ModelAdapter(ABC):
    """Lower-level adapter interface, one step below LLMProvider."""

    @abstractmethod
    def generate(self, messages: list, **kwargs) -> dict:
        ...


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_NAME_ALIASES = ("name", "action", "tool")
_ARGS_ALIASES = ("arguments", "args", "parameters", "kwargs")
_GOAL_ALIASES = ("goal", "description", "reason")


def _decode_json(text: str) -> Optional[str]:
    """Strip markdown fences and return the raw JSON string, or None."""
    if not text:
        return None
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate
    text = text.strip()
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    return text[start:]


def _extract_balanced(text: str) -> Optional[str]:
    """Extract the first balanced {...} or [...] block from text."""
    start_idx = None
    open_ch = None
    close_ch = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start_idx = i
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start_idx is None:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _first_field(d: dict, aliases: tuple) -> Any:
    for a in aliases:
        if a in d:
            return d[a]
    return None


def _dict_to_action(d: dict) -> Optional[Action]:
    name = _first_field(d, _NAME_ALIASES)
    if not name or not isinstance(name, str):
        return None
    arguments = _first_field(d, _ARGS_ALIASES)
    goal = _first_field(d, _GOAL_ALIASES) or ""
    try:
        return Action(name=name, arguments=arguments or {}, goal=str(goal))
    except Exception:
        return None


def _action(text: str) -> Optional[Action]:
    """Extract an Action from free-form LLM text."""
    if not text:
        return None

    raw = _decode_json(text)
    if raw:
        balanced = _extract_balanced(raw) or raw
        try:
            parsed = json.loads(balanced)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            if isinstance(parsed, dict):
                act = _dict_to_action(parsed)
                if act:
                    return act
            elif isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict):
                    act = _dict_to_action(first)
                    if act:
                        return act
    return None


def action_protocol_instruction() -> str:
    return (
        "Respond with a single JSON object describing your next action. "
        "Format:\n"
        '{"name": "<tool_name>", "arguments": {...}, "goal": "<why>"}\n'
        f"Valid tool names: {', '.join(TOOL_NAMES)}, done, replan.\n"
        "Wrap the JSON in a ```json code fence. Do not include any other "
        "JSON-looking text outside the action object."
    )
