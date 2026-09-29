import pytest

from harness.context.manager import ContextManager
from harness.state import TaskState
from harness.orchestrator import Orchestrator


def make_state():
    return TaskState(issue="Fix the login bug in auth.py", repo_path="/tmp/repo")


class TestAssemble:
    def test_message_ordering(self):
        state = make_state()
        cm = ContextManager(state, "/tmp/repo")
        tools = [{"name": "read_file", "description": "x"}]
        recent_tools = [{"tool": "read_file", "args": {"path": "a.py"},
                          "result": {"status": "success", "output": "content"}}]
        llm_turns = [{"role": "assistant", "content": "previous turn"}]
        ctx = cm.assemble(tools, recent_tools, llm_turns)
        roles = [m["role"] for m in ctx["messages"]]
        assert roles[0] == "system"
        assert roles[1] == "user"  # issue prompt
        assert "assistant" in roles
        assert ctx["estimated_tokens"] > 0

    def test_tool_result_truncation(self):
        state = make_state()
        cm = ContextManager(state, "/tmp/repo")
        long_output = "x" * 5000
        recent_tools = [{"tool": "read_file", "args": {}, "result": {"status": "success", "output": long_output}}]
        ctx = cm.assemble([], recent_tools, [])
        combined = "\n".join(m["content"] for m in ctx["messages"])
        assert "truncated" in combined

    def test_issue_included(self):
        state = make_state()
        cm = ContextManager(state, "/tmp/repo")
        ctx = cm.assemble([], [], [])
        assert "Fix the login bug" in ctx["messages"][1]["content"]


class TestEstimateTokens:
    def test_estimate_tokens_grows_with_content(self):
        state = make_state()
        cm = ContextManager(state, "/tmp/repo")
        small = cm.estimate_tokens([{"a": "b"}])
        large = cm.estimate_tokens([{"a": "b" * 1000}])
        assert large > small


class TestReadCache:
    def test_read_cache_short_circuits(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        orch.state.mark_read("calc.py", "outline", step=1, digest="def add(...)")
        cached = orch.state.was_read("calc.py", "outline")
        assert cached is not None
        assert cached["step"] == 1

    def test_invalidate_on_edit(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        orch.state.mark_read("calc.py", "outline", step=1, digest="digest")
        orch.state.invalidate_read_cache("calc.py")
        assert orch.state.was_read("calc.py", "outline") is None


class TestCompaction:
    def test_summarize_results_produces_digest(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        old = [
            {"tool": "read_file", "args": {"path": "a.py"}, "result": {"status": "success", "output": "x"}},
            {"tool": "edit_file", "args": {"path": "a.py"}, "result": {"status": "success", "output": "y"}},
            {"tool": "run_tests", "args": {}, "result": {"status": "error", "passed": False, "output": "fail"}},
        ]
        digest = orch._summarize_results(old)
        assert "a.py" in digest
        assert "failed test" in digest

    def test_compact_context_reduces_tokens(self, tmp_repo):
        orch = Orchestrator(repo_path=tmp_repo, issue="fix", provider=None)
        for i in range(20):
            orch.state.tool_results.append({
                "tool": "read_file", "args": {"path": f"file{i}.py"},
                "result": {"status": "success", "output": "x" * 500},
                "timestamp": "now",
            })
        before = orch.context_mgr.estimate_tokens(orch.state.tool_results)
        orch._compact_context()
        after = orch.context_mgr.estimate_tokens(orch.state.tool_results)
        assert after < before
