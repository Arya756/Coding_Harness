"""Failure classification and recovery strategy selection."""

from __future__ import annotations

import os
from typing import Optional


class FailureType:
    TEST_FAILURE = "test_failure"
    COMMAND_ERROR = "command_error"
    EDIT_FAILURE = "edit_failure"
    CONTEXT_OVERFLOW = "context_overflow"
    STUCK = "stuck"
    UNKNOWN = "unknown"
    FILE_NOT_FOUND = "file_not_found"
    SYNTAX_ERROR = "syntax_error"
    PERMISSION_DENIED = "permission_denied"
    REGRESSION = "regression"
    NONE = "none"


def classify_failure(result: dict, state) -> str:
    if result.get("status") == "success":
        return FailureType.NONE

    hint = result.get("failure_hint")
    if hint == "regression":
        return FailureType.REGRESSION
    if hint == "syntax":
        return FailureType.SYNTAX_ERROR

    tool = result.get("tool")
    output = str(result.get("output", "")).lower()

    if tool == "run_tests" and (result.get("passed") is False or result.get("status") == "error"):
        return FailureType.TEST_FAILURE
    if tool == "run_command":
        if "timed out" in output:
            return FailureType.COMMAND_ERROR
        if result.get("exit_code") not in (None, 0):
            return FailureType.COMMAND_ERROR
    if tool in ("edit_file", "edit_file_multi") and "not found" in output:
        return FailureType.EDIT_FAILURE
    if "denied" in output:
        return FailureType.PERMISSION_DENIED
    if "not found" in output and tool in ("read_file", "get_file_info"):
        return FailureType.FILE_NOT_FOUND

    details = str(result.get("output", ""))
    failure_type_guess = FailureType.UNKNOWN
    if state.is_stuck(failure_type_guess, details):
        return FailureType.STUCK

    return FailureType.UNKNOWN


class RecoveryEngine:
    def __init__(self, state):
        self.state = state

    def handle(self, result: dict) -> dict:
        failure_type = classify_failure(result, self.state)
        details = str(result.get("output", ""))
        tool = result.get("tool", "?")

        if failure_type == FailureType.NONE:
            return {"action": "none", "failure_type": failure_type, "message": "", "prompt_addition": ""}

        self.state.record_failure(failure_type, details)

        if self.state.is_stuck(failure_type, details) and failure_type != FailureType.STUCK:
            # Escalation reachable: stuck signature persisting past normal retries.
            if self.state.retry_counts.get(self.state._signature(failure_type, details), 0) > 3:
                return self._strategy_escalate(failure_type, tool, details)
            return self._strategy_replan(failure_type, tool, details)

        if failure_type == FailureType.STUCK:
            return self._strategy_replan(failure_type, tool, details)
        if failure_type == FailureType.REGRESSION:
            return self._strategy_rollback(failure_type, tool, details, "caused a test regression")
        if failure_type == FailureType.SYNTAX_ERROR:
            if self.state.checkpoints:
                return self._strategy_rollback(failure_type, tool, details, "broke syntax")
            return self._strategy_retry(failure_type, tool, details)
        if failure_type == FailureType.CONTEXT_OVERFLOW:
            return self._strategy_compact(failure_type, tool, details)

        return self._strategy_retry(failure_type, tool, details)

    # -- strategies ----------------------------------------------------
    def _strategy_retry(self, failure_type, tool, details) -> dict:
        retry_count = self.state.retry_counts.get(self.state._signature(failure_type, details), 1)
        msg = (
            f"Tool '{tool}' failed ({failure_type}), attempt {retry_count}. "
            f"Output: {details[:500]}\nTry a different approach."
        )
        return {"action": "retry", "failure_type": failure_type, "message": msg, "prompt_addition": msg}

    def _strategy_replan(self, failure_type, tool, details) -> dict:
        counts = self.state.retry_counts.get(self.state._signature(failure_type, details), 0)
        msg = (
            f"STUCK: repeated failures of type '{failure_type}' on tool '{tool}' "
            f"({counts} attempts). Details: {details[:1000]}\n"
            f"Original issue: {self.state.issue}\n"
            "You must try a genuinely different approach. Consider:\n"
            "1. A different file or module may be the real source of the bug.\n"
            "2. Re-read the exact current file contents (mode='full') before editing again.\n"
            "3. Break the fix into smaller, independently-testable edits.\n"
            "4. Check whether your assumption about the failing behavior is correct."
        )
        return {"action": "replan", "failure_type": failure_type, "message": msg, "prompt_addition": msg}

    def _strategy_rollback(self, failure_type, tool, details, reason) -> dict:
        msg = (
            f"Reverted your last edit because it {reason}. "
            f"Details: {details[:500]}\nRe-examine before editing again."
        )
        return {"action": "rollback", "failure_type": failure_type, "message": msg, "prompt_addition": msg}

    def _strategy_compact(self, failure_type, tool, details) -> dict:
        return {
            "action": "compact",
            "failure_type": failure_type,
            "message": "Context too large; compacting older tool results.",
            "prompt_addition": "",
        }

    def _strategy_escalate(self, failure_type, tool, details) -> dict:
        msg = (
            f"ESCALATION: failure type '{failure_type}' on tool '{tool}' persists after "
            f"replanning. Last error: {details[:500]}. Manual intervention needed."
        )
        try:
            harness_dir = os.path.join(self.state.repo_path, ".harness")
            os.makedirs(harness_dir, exist_ok=True)
            with open(os.path.join(harness_dir, "needs_human.md"), "w") as f:
                f.write("# Needs Human Intervention\n\n")
                f.write(f"Issue: {self.state.issue}\n\n")
                f.write(f"Failure type: {failure_type}\n")
                f.write(f"Tool: {tool}\n\n")
                f.write(f"Details:\n{details}\n\n")
                f.write(f"Retry counts: {self.state.retry_counts}\n")
        except OSError:
            pass
        return {"action": "escalate", "failure_type": failure_type, "message": msg, "prompt_addition": msg}
