"""Token tracking, cost estimation, and telemetry event collection."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import TELEMETRY_DIR
from ..llm import estimate_cost as _estimate_cost


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenTracker:
    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model
        self._spend = 0.0
        self._enc_cache = {}

        try:
            import tiktoken  # type: ignore

            self._tiktoken = tiktoken
        except Exception:
            self._tiktoken = None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        if not text:
            return 0
        model = model or self.default_model
        if self._tiktoken:
            try:
                if model not in self._enc_cache:
                    try:
                        self._enc_cache[model] = self._tiktoken.encoding_for_model(model)
                    except Exception:
                        self._enc_cache[model] = self._tiktoken.get_encoding("cl100k_base")
                return len(self._enc_cache[model].encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def estimate_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        return _estimate_cost(model, in_tok, out_tok)

    def check_budget(self, estimated_cost: float, budget_usd: Optional[float]) -> bool:
        if budget_usd is None:
            return True
        return (self._spend + estimated_cost) <= budget_usd

    def add_spend(self, amount: float) -> None:
        self._spend += amount

    def total_spend(self) -> float:
        return self._spend


class TelemetryCollector:
    def __init__(self, repo_path: str, state, token_tracker: TokenTracker):
        self.repo_path = repo_path
        self.state = state
        self.token_tracker = token_tracker
        self.events: list = []

    def _log(self, event_type: str, **fields) -> None:
        self.events.append({
            "timestamp": _now_iso(),
            "event_type": event_type,
            "session_id": self.state.session_id,
            "step": self.state.step,
            **fields,
        })

    def log_session_start(self, model: str, budget_usd: Optional[float]) -> None:
        self._log("session_start", repo_path=self.repo_path, issue=self.state.issue[:200],
                   model=model, budget_usd=budget_usd)

    def log_session_end(self) -> None:
        summary = self.get_summary()
        self._log(
            "session_end",
            status=self.state.status,
            total_steps=self.state.step,
            total_llm_calls=summary["llm_calls"],
            total_tokens=summary["total_tokens"],
            total_cost_usd=summary["total_cost_usd"],
            resolved=self.state.resolved,
            failure_reason=self.state.failure_reason,
            no_tests=self.state.no_tests,
        )

    def log_llm_call(self, model: str, in_tok: int, out_tok: int, cost: float,
                      prompt_preview: str = "", response_preview: str = "") -> None:
        self._log("llm_call", model=model, input_tokens=in_tok, output_tokens=out_tok,
                   cost_usd=cost, prompt_preview=prompt_preview[:200],
                   response_preview=response_preview[:200])

    def log_tool_call(self, tool: str, args: dict, status: str, tokens: int = 0,
                       duration_ms: int = 0, output_preview: str = "") -> None:
        self._log("tool_call", tool=tool, args=args, status=status, tokens=tokens,
                   duration_ms=duration_ms, output_preview=output_preview[:200])

    def log_test_run(self, passed: bool, exit_code: Optional[int], duration_ms: int,
                      output_preview: str = "", no_tests: bool = False) -> None:
        self._log("test_run", passed=passed, exit_code=exit_code, duration_ms=duration_ms,
                   output_preview=output_preview[:300], no_tests=no_tests)

    def log_edit(self, path: str, old_str_preview: str, new_str_preview: str) -> None:
        self._log("edit", path=path, old_str_preview=old_str_preview[:200],
                   new_str_preview=new_str_preview[:200])

    def log_recovery(self, failure_type: str, action: str, message: str) -> None:
        self._log("recovery", failure_type=failure_type, action=action, message=message[:500])

    def log_rollback(self, path: str, reason: str) -> None:
        self._log("rollback", path=path, reason=reason)

    def log_compaction(self, before: int, after: int) -> None:
        self._log("compaction", before=before, after=after, saved_tokens=max(0, before - after))

    def log_verify(self, path: str, passed: bool, summary: str) -> None:
        self._log("verify", path=path, passed=passed, summary=summary[:300])

    def log_state_save(self) -> None:
        self._log("state_save")

    # ------------------------------------------------------------------
    def flush(self, filepath: Optional[str] = None) -> None:
        filepath = filepath or os.path.join(self.repo_path, TELEMETRY_DIR, "events.jsonl")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")
        self.events = []

    def get_summary(self) -> dict:
        llm_calls = len(self.state.llm_calls)
        tool_calls = len(self.state.tool_calls)
        test_runs = sum(1 for tc in self.state.tool_calls if tc.get("tool") == "run_tests")
        edits = sum(1 for tc in self.state.tool_calls if tc.get("tool") in ("edit_file", "edit_file_multi"))
        rollbacks = sum(1 for e in self.events if e.get("event_type") == "rollback")
        recoveries = len(self.state.failure_history)

        tokens_in = sum(c.get("input_tokens", 0) for c in self.state.llm_calls)
        tokens_out = sum(c.get("output_tokens", 0) for c in self.state.llm_calls)
        total_cost = sum(c.get("cost_usd", 0.0) for c in self.state.llm_calls)

        tool_usage: dict = {}
        for tc in self.state.tool_calls:
            t = tc.get("tool", "?")
            tool_usage[t] = tool_usage.get(t, 0) + 1

        return {
            "session_id": self.state.session_id,
            "status": self.state.status,
            "total_steps": self.state.step,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "test_runs": test_runs,
            "edits": edits,
            "recoveries": recoveries,
            "rollbacks": rollbacks,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "total_cost_usd": round(total_cost, 6),
            "tool_usage": tool_usage,
            "files_read": len(self.state.files_read),
            "files_modified": len(self.state.files_modified),
            "resolved": self.state.resolved,
            "failure_reason": self.state.failure_reason,
            "no_tests": self.state.no_tests,
        }
