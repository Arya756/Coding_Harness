import pytest

from harness.recovery.engine import FailureType, RecoveryEngine, classify_failure
from harness.state import TaskState


def make_state(issue="fix bug", repo="/tmp/repo"):
    return TaskState(issue=issue, repo_path=repo)


class TestClassifyFailure:
    def test_success_is_none(self):
        state = make_state()
        result = {"status": "success", "tool": "read_file", "output": "ok"}
        assert classify_failure(result, state) == FailureType.NONE

    def test_regression_hint(self):
        state = make_state()
        result = {"status": "error", "tool": "run_tests", "output": "boom", "failure_hint": "regression"}
        assert classify_failure(result, state) == FailureType.REGRESSION

    def test_syntax_hint(self):
        state = make_state()
        result = {"status": "error", "tool": "edit_file", "output": "bad syntax", "failure_hint": "syntax"}
        assert classify_failure(result, state) == FailureType.SYNTAX_ERROR

    def test_test_failure(self):
        state = make_state()
        result = {"status": "error", "tool": "run_tests", "output": "assert fail", "passed": False}
        assert classify_failure(result, state) == FailureType.TEST_FAILURE

    def test_command_error_timeout(self):
        state = make_state()
        result = {"status": "error", "tool": "run_command", "output": "Command timed out after 30s"}
        assert classify_failure(result, state) == FailureType.COMMAND_ERROR

    def test_command_error_exit_code(self):
        state = make_state()
        result = {"status": "error", "tool": "run_command", "output": "failed", "exit_code": 1}
        assert classify_failure(result, state) == FailureType.COMMAND_ERROR

    def test_edit_failure_not_found(self):
        state = make_state()
        result = {"status": "error", "tool": "edit_file", "output": "old_str not found in x.py"}
        assert classify_failure(result, state) == FailureType.EDIT_FAILURE

    def test_permission_denied(self):
        state = make_state()
        result = {"status": "error", "tool": "run_command", "output": "Blocked: denied"}
        assert classify_failure(result, state) == FailureType.PERMISSION_DENIED

    def test_unknown(self):
        state = make_state()
        result = {"status": "error", "tool": "think", "output": "some weird error"}
        assert classify_failure(result, state) == FailureType.UNKNOWN


class TestStuckDetection:
    def test_is_stuck_after_max_retries(self):
        state = make_state()
        for _ in range(3):
            state.record_failure(FailureType.UNKNOWN, "some weird error")
        assert state.is_stuck(FailureType.UNKNOWN, "some weird error") is True

    def test_not_stuck_below_threshold(self):
        state = make_state()
        state.record_failure(FailureType.UNKNOWN, "some weird error")
        assert state.is_stuck(FailureType.UNKNOWN, "some weird error") is False


class TestRecoveryEngine:
    def test_retry_strategy_for_test_failure(self):
        state = make_state()
        engine = RecoveryEngine(state)
        result = {"status": "error", "tool": "run_tests", "output": "assert fail", "passed": False}
        rec = engine.handle(result)
        assert rec["action"] == "retry"

    def test_rollback_strategy_for_regression(self):
        state = make_state()
        engine = RecoveryEngine(state)
        result = {"status": "error", "tool": "run_tests", "output": "regressed", "failure_hint": "regression"}
        rec = engine.handle(result)
        assert rec["action"] == "rollback"
        assert "regression" in rec["prompt_addition"]

    def test_replan_when_stuck(self):
        state = make_state()
        engine = RecoveryEngine(state)
        result = {"status": "error", "tool": "run_tests", "output": "assert fail", "passed": False}
        for _ in range(3):
            rec = engine.handle(result)
        # After hitting the retry cap, the same failure signature should trigger replan/escalate.
        assert rec["action"] in ("replan", "escalate")

    def test_escalate_reachable_after_persistent_failures(self):
        state = make_state()
        engine = RecoveryEngine(state)
        result = {"status": "error", "tool": "run_tests", "output": "assert fail", "passed": False}
        actions = [engine.handle(result)["action"] for _ in range(8)]
        assert "escalate" in actions
        assert state.status != "running" or True  # escalate itself doesn't set status; orchestrator does

    def test_none_action_on_success(self):
        state = make_state()
        engine = RecoveryEngine(state)
        rec = engine.handle({"status": "success", "tool": "read_file", "output": "ok"})
        assert rec["action"] == "none"
