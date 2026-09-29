import os
import subprocess

import pytest

from harness.models import Action
from harness.orchestrator import Orchestrator
from tests.conftest import FakeLLMProvider


def _git_log_count(repo):
    result = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    return len(result.stdout.strip().splitlines())


class TestMarkResolvedGating:
    def test_refuses_commit_when_tests_fail(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix add()", provider=None)
        before_count = _git_log_count(tmp_repo)
        result = orch._mark_resolved()
        assert result["status"] == "error"
        assert orch.state.resolved is False
        assert _git_log_count(tmp_repo) == before_count

    def test_commits_when_tests_pass(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix add()", provider=None)
        # Fix the bug directly on disk.
        calc_path = os.path.join(tmp_repo, "calc.py")
        with open(calc_path, "w") as f:
            f.write("def add(a, b):\n    return a + b\n")
        before_count = _git_log_count(tmp_repo)
        result = orch._mark_resolved()
        assert result["status"] == "success"
        assert orch.state.resolved is True
        assert orch.state.final_commit is not None
        assert _git_log_count(tmp_repo) == before_count + 1


class TestCheckpointRollback:
    def test_rollback_restores_modified_file(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        orch._checkpoint_files(["calc.py"])
        calc_path = os.path.join(tmp_repo, "calc.py")
        original = open(calc_path).read()
        with open(calc_path, "w") as f:
            f.write("def add(a, b):\n    return a ** b  # broken\n")
        orch._rollback("calc.py")
        assert open(calc_path).read() == original

    def test_rollback_removes_newly_created_file(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        new_path = os.path.join(tmp_repo, "brand_new.py")
        orch._checkpoint_files(["brand_new.py"])
        with open(new_path, "w") as f:
            f.write("x = 1\n")
        orch._rollback("brand_new.py")
        assert not os.path.exists(new_path)


class TestRegressionFlow:
    def test_edit_causing_regression_is_rolled_back(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix add()", provider=None)
        calc_path = os.path.join(tmp_repo, "calc.py")

        # Fix the bug first and confirm tests pass.
        with open(calc_path, "w") as f:
            f.write("def add(a, b):\n    return a + b\n")
        orch._run_tests()
        assert orch.state.tests_passed is True
        fixed_content = open(calc_path).read()

        # Now the agent "fixes" it again but actually re-introduces the bug.
        action = Action(name="edit_file", arguments={
            "path": "calc.py",
            "old_str": "return a + b",
            "new_str": "return a - b",
        })
        orch._execute_action(action)

        # The regression should have been detected and rolled back.
        assert open(calc_path).read() == fixed_content
        rollback_decisions = [d for d in orch.state.decisions if "Rolled back" in d.get("decision", "")]
        assert len(rollback_decisions) == 1


class TestForceEditPush:
    def test_fires_after_consecutive_reads(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        fired = False
        for _ in range(5):
            fired = orch._track_consecutive_reads(Action(name="read_file", arguments={"path": "calc.py"}))
        assert fired is True
        assert orch._consecutive_reads == 0
        assert any("Now produce an edit_file" in t.get("content", "") for t in orch.recent_llm_turns)

    def test_reset_by_edit(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        for _ in range(3):
            orch._track_consecutive_reads(Action(name="read_file", arguments={}))
        orch._track_consecutive_reads(Action(name="edit_file", arguments={}))
        assert orch._consecutive_reads == 0


class TestBudgetAbort:
    def test_aborts_before_overshoot(self, tmp_repo):
        provider = FakeLLMProvider(responses=[
            '{"name": "think", "arguments": {"question": "hi"}}',
        ] * 10)
        orch = Orchestrator(
            repo_path=tmp_repo, issue="fix add()", provider=provider,
            budget_usd=0.001, max_steps=10,
        )
        result = orch.run()
        assert result["status"] == "failed"
        assert orch.state.failure_reason == "budget"
        # Should not have run away spending far beyond budget.
        assert orch.token_tracker.total_spend() < 0.01


class TestSessionTimeout:
    def test_aborts_on_wall_clock_limit(self, tmp_repo):
        provider = FakeLLMProvider(responses=[
            '{"name": "think", "arguments": {"question": "hi"}}',
        ] * 10)
        orch = Orchestrator(
            repo_path=tmp_repo, issue="fix add()", provider=provider,
            max_steps=10, max_seconds=0,
        )
        result = orch.run()
        assert result["status"] == "stuck"
        assert orch.state.failure_reason == "session_timeout"


class TestFullRunWithFakeProvider:
    def test_happy_path_resolves_and_commits(self, tmp_repo):
        plan_response = (
            '[{"step": 1, "tool": "read_file", "args": {"path": "calc.py", "mode": "full"}, '
            '"goal": "see bug"}]'
        )
        turn_responses = [
            '{"name": "read_file", "arguments": {"path": "calc.py", "mode": "full"}, "goal": "inspect"}',
            '{"name": "edit_file", "arguments": {"path": "calc.py", '
            '"old_str": "return a - b  # BUG: should be +", "new_str": "return a + b"}, "goal": "fix bug"}',
            '{"name": "done", "arguments": {}, "goal": "finish"}',
        ]
        provider = FakeLLMProvider(responses=[plan_response] + turn_responses)
        orch = Orchestrator(repo_path=tmp_repo, issue="calc.add returns wrong result", provider=provider,
                             max_steps=10)
        result = orch.run()
        assert result["status"] == "resolved"
        assert orch.state.final_commit is not None
        with open(os.path.join(tmp_repo, "calc.py")) as f:
            assert "a + b" in f.read()
