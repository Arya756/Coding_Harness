"""TaskState dataclass: progress tracking, read-cache, checkpoints, persistence."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from .config import MAX_RETRIES_PER_FAILURE


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskState:
    issue: str
    repo_path: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=now_iso)

    step: int = 0
    plan: Optional[list] = None
    status: str = "running"  # running | resolved | failed | stuck

    files_read: list = field(default_factory=list)
    files_modified: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    read_cache: dict = field(default_factory=dict)

    failure_history: list = field(default_factory=list)
    retry_counts: dict = field(default_factory=dict)

    resolved: bool = False
    failure_reason: Optional[str] = None
    final_commit: Optional[str] = None
    tests_passed: Optional[bool] = None
    tests_output: Optional[str] = None
    no_tests: bool = False

    # In-memory only, never persisted.
    checkpoints: dict = field(default_factory=dict, repr=False, compare=False)

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("checkpoints", None)
        return json.dumps(d, indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> "TaskState":
        d = json.loads(text)
        d.pop("checkpoints", None)
        return cls(**d)

    def record_tool_call(self, tool: str, args: dict, result: dict) -> None:
        entry_call = {"tool": tool, "args": args, "timestamp": now_iso()}
        self.tool_calls.append(entry_call)
        entry_result = {"tool": tool, "args": args, "result": result, "timestamp": now_iso()}
        self.tool_results.append(entry_result)

    def record_llm_call(self, model: str, prompt_preview: str, in_tok: int, out_tok: int, cost_usd: float) -> None:
        self.llm_calls.append({
            "model": model,
            "prompt_preview": prompt_preview[:200],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost_usd,
            "timestamp": now_iso(),
        })

    def record_decision(self, decision: str, rationale: str = "") -> None:
        self.decisions.append({
            "decision": decision,
            "rationale": rationale,
            "timestamp": now_iso(),
        })

    def record_failure(self, failure_type: str, details: str, resolution: str = "") -> None:
        self.failure_history.append({
            "type": failure_type,
            "attempt": self.retry_counts.get(self._signature(failure_type, details), 0) + 1,
            "timestamp": now_iso(),
            "resolution": resolution,
        })
        sig = self._signature(failure_type, details)
        self.retry_counts[sig] = self.retry_counts.get(sig, 0) + 1

    @staticmethod
    def _signature(failure_type: str, details: str) -> str:
        return f"{failure_type}:{(details or '')[:50]}"

    def is_stuck(self, failure_type: str, details: str) -> bool:
        sig = self._signature(failure_type, details)
        return self.retry_counts.get(sig, 0) >= MAX_RETRIES_PER_FAILURE

    def mark_read(self, path: str, mode: str, step: int, digest: str) -> None:
        self.read_cache[f"{path}:{mode}"] = {"step": step, "digest": digest}

    def was_read(self, path: str, mode: str) -> Optional[dict]:
        return self.read_cache.get(f"{path}:{mode}")

    def invalidate_read_cache(self, path: str) -> None:
        keys = [k for k in self.read_cache if k.startswith(f"{path}:")]
        for k in keys:
            self.read_cache.pop(k, None)

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "issue": self.issue[:200],
            "status": self.status,
            "step": self.step,
            "resolved": self.resolved,
            "failure_reason": self.failure_reason,
            "final_commit": self.final_commit,
            "tests_passed": self.tests_passed,
            "no_tests": self.no_tests,
            "files_read": len(self.files_read),
            "files_modified": len(self.files_modified),
            "llm_calls": len(self.llm_calls),
            "tool_calls": len(self.tool_calls),
        }

    # -- persistence -------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "TaskState":
        with open(path) as f:
            return cls.from_json(f.read())
