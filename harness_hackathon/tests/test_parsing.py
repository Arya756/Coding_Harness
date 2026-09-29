import pytest
from pydantic import ValidationError

from harness.models import Action, _action, action_protocol_instruction
from harness.orchestrator import Orchestrator


class TestActionModel:
    def test_valid_action(self):
        a = Action(name="read_file", arguments={"path": "x.py"}, goal="check")
        assert a.name == "read_file"
        assert a.arguments == {"path": "x.py"}

    def test_arguments_as_json_string_is_coerced(self):
        a = Action(name="read_file", arguments='{"path": "x.py"}')
        assert a.arguments == {"path": "x.py"}

    def test_arguments_as_bad_string_wrapped(self):
        a = Action(name="think", arguments="not json")
        assert a.arguments == {"_raw": "not json"}

    def test_default_arguments(self):
        a = Action(name="list_dir")
        assert a.arguments == {}


class TestActionParsingFromText:
    def test_fenced_json(self):
        text = '```json\n{"name": "grep", "arguments": {"pattern": "foo"}, "goal": "search"}\n```'
        act = _action(text)
        assert act is not None
        assert act.name == "grep"
        assert act.arguments["pattern"] == "foo"

    def test_raw_json(self):
        text = '{"name": "run_tests", "arguments": {}}'
        act = _action(text)
        assert act.name == "run_tests"

    def test_field_aliases_tool_args(self):
        text = '{"tool": "edit_file", "args": {"path": "a.py"}, "reason": "fix"}'
        act = _action(text)
        assert act.name == "edit_file"
        assert act.arguments == {"path": "a.py"}
        assert act.goal == "fix"

    def test_action_alias(self):
        text = '{"action": "done", "parameters": {}}'
        act = _action(text)
        assert act.name == "done"

    def test_no_json_returns_none(self):
        assert _action("I think we should look at the file") is None

    def test_list_wrapped_json(self):
        text = '[{"name": "think", "arguments": {"question": "hmm"}}]'
        act = _action(text)
        assert act.name == "think"

    def test_protocol_instruction_mentions_tools(self):
        instr = action_protocol_instruction()
        assert "read_file" in instr
        assert "json" in instr.lower()


class TestInferenceFallback:
    def _orch(self, repo, issue="Fix the login bug"):
        return Orchestrator(repo_path=repo, issue=issue, provider=None)

    def test_infer_read_file(self, tmp_repo):
        orch = self._orch(tmp_repo)
        act = orch._infer_action("I should read_file calc.py to understand it")
        assert act.name == "read_file"

    def test_infer_grep(self, tmp_repo):
        orch = self._orch(tmp_repo)
        act = orch._infer_action("Let me grep for the relevant function")
        assert act.name == "grep"

    def test_infer_tests(self, tmp_repo):
        orch = self._orch(tmp_repo)
        act = orch._infer_action("Now let's run test to verify")
        assert act.name == "run_tests"

    def test_infer_done(self, tmp_repo):
        orch = self._orch(tmp_repo)
        act = orch._infer_action("The issue is done and resolved")
        assert act.name == "done"

    def test_infer_replan(self, tmp_repo):
        orch = self._orch(tmp_repo)
        act = orch._infer_action("I need a new plan for this")
        assert act.name == "replan"

    def test_fallback_think(self, tmp_repo):
        orch = self._orch(tmp_repo)
        act = orch._infer_action("asdkjaslkdjalksjd")
        assert act.name == "think"
