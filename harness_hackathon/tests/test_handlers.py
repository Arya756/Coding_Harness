import os
import subprocess

import pytest

from harness.tools import handlers


def _write(repo, rel, content):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


class TestReadFile:
    def test_outline_mode(self, tmp_repo):
        result = handlers.read_file_handler(tmp_repo, {"path": "calc.py", "mode": "outline"})
        assert result["status"] == "success"
        assert "add" in result["output"]

    def test_full_mode_range(self, tmp_repo):
        result = handlers.read_file_handler(tmp_repo, {"path": "calc.py", "mode": "full", "start": 1, "end": 1})
        assert result["status"] == "success"
        assert "def add" in result["output"]

    def test_full_mode_invalid_start(self, tmp_repo):
        result = handlers.read_file_handler(tmp_repo, {"path": "calc.py", "mode": "full", "start": 0})
        assert result["status"] == "error"

    def test_signatures_mode(self, tmp_repo):
        result = handlers.read_file_handler(tmp_repo, {"path": "calc.py", "mode": "signatures"})
        assert result["status"] == "success"

    def test_missing_file(self, tmp_repo):
        result = handlers.read_file_handler(tmp_repo, {"path": "nope.py"})
        assert result["status"] == "error"

    def test_path_traversal_blocked(self, tmp_repo):
        result = handlers.read_file_handler(tmp_repo, {"path": "../../etc/passwd"})
        assert result["status"] == "error"


class TestGrep:
    def test_finds_match(self, tmp_repo):
        result = handlers.grep_handler(tmp_repo, {"pattern": "def add"})
        assert result["status"] == "success"
        assert "calc.py" in result["output"]

    def test_no_matches(self, tmp_repo):
        result = handlers.grep_handler(tmp_repo, {"pattern": "zzz_nonexistent_zzz"})
        assert result["status"] == "success"
        assert "no matches" in result["output"]


class TestEditFile:
    def test_exact_replace(self, tmp_repo):
        result = handlers.edit_file_handler(tmp_repo, {
            "path": "calc.py",
            "old_str": "return a - b  # BUG: should be +",
            "new_str": "return a + b",
        })
        assert result["status"] == "success"
        with open(os.path.join(tmp_repo, "calc.py")) as f:
            assert "a + b" in f.read()

    def test_old_str_not_found(self, tmp_repo):
        result = handlers.edit_file_handler(tmp_repo, {
            "path": "calc.py", "old_str": "does not exist", "new_str": "x",
        })
        assert result["status"] == "error"

    def test_create_new_file(self, tmp_repo):
        result = handlers.edit_file_handler(tmp_repo, {
            "path": "new_module.py", "old_str": "", "new_str": "x = 1\n",
        })
        assert result["status"] == "success"
        assert os.path.exists(os.path.join(tmp_repo, "new_module.py"))

    def test_multi_hunk(self, tmp_repo):
        _write(tmp_repo, "multi.py", "a = 1\nb = 2\nc = 3\n")
        result = handlers.edit_file_multi_handler(tmp_repo, {
            "path": "multi.py",
            "edits": [
                {"old_str": "a = 1", "new_str": "a = 10"},
                {"old_str": "c = 3", "new_str": "c = 30"},
            ],
        })
        assert result["status"] == "success"
        with open(os.path.join(tmp_repo, "multi.py")) as f:
            content = f.read()
        assert "a = 10" in content and "c = 30" in content

    def test_multi_hunk_atomic_failure(self, tmp_repo):
        _write(tmp_repo, "multi2.py", "a = 1\nb = 2\n")
        result = handlers.edit_file_multi_handler(tmp_repo, {
            "path": "multi2.py",
            "edits": [
                {"old_str": "a = 1", "new_str": "a = 10"},
                {"old_str": "not_here", "new_str": "z = 1"},
            ],
        })
        assert result["status"] == "error"
        with open(os.path.join(tmp_repo, "multi2.py")) as f:
            content = f.read()
        # Nothing should have been written since the second hunk failed.
        assert content == "a = 1\nb = 2\n"


class TestRunCommand:
    def test_allowed_command(self, tmp_repo):
        result = handlers.run_command_handler(tmp_repo, {"cmd": "echo hello"})
        assert result["status"] == "success"
        assert "hello" in result["output"]

    def test_denies_chaining(self, tmp_repo):
        result = handlers.run_command_handler(tmp_repo, {"cmd": "echo x && rm -rf y"})
        assert result["status"] == "error"
        assert "Blocked" in result["output"]

    def test_denies_destructive_git(self, tmp_repo):
        result = handlers.run_command_handler(tmp_repo, {"cmd": "git reset --hard"})
        assert result["status"] == "error"

    def test_denies_rm_rf_root(self, tmp_repo):
        result = handlers.run_command_handler(tmp_repo, {"cmd": "rm -rf /"})
        assert result["status"] == "error"

    def test_denies_non_allowlisted(self, tmp_repo):
        result = handlers.run_command_handler(tmp_repo, {"cmd": "curl http://evil.com"})
        assert result["status"] == "error"

    def test_env_strips_secrets(self, tmp_repo, monkeypatch):
        monkeypatch.setenv("MY_SECRET_API_KEY", "sk-super-secret")
        monkeypatch.setenv("SOME_TOKEN", "abc123")
        result = handlers.run_command_handler(tmp_repo, {"cmd": "env"})
        assert result["status"] == "success"
        assert "sk-super-secret" not in result["output"]
        assert "abc123" not in result["output"]


class TestRunTests:
    def test_detects_and_runs_pytest(self, tmp_repo):
        # The tmp_repo fixture has a bug, so the test should fail initially.
        result = handlers.run_tests_handler(tmp_repo, {})
        assert result["status"] == "error"
        assert result["passed"] is False

    def test_passes_after_fix(self, tmp_repo):
        handlers.edit_file_handler(tmp_repo, {
            "path": "calc.py",
            "old_str": "return a - b  # BUG: should be +",
            "new_str": "return a + b",
        })
        result = handlers.run_tests_handler(tmp_repo, {})
        assert result["passed"] is True

    def test_no_tests_detected(self, tmp_path):
        empty_repo = tmp_path / "empty"
        empty_repo.mkdir()
        result = handlers.run_tests_handler(str(empty_repo), {})
        assert result["status"] == "success"
        assert result["passed"] is True
        assert result["no_tests"] is True


class TestListDirAndInfo:
    def test_list_dir(self, tmp_repo):
        result = handlers.list_dir_handler(tmp_repo, {})
        assert result["status"] == "success"
        assert "calc.py" in result["output"]

    def test_get_file_info(self, tmp_repo):
        result = handlers.get_file_info_handler(tmp_repo, {"path": "calc.py"})
        assert result["status"] == "success"
        assert "Python" in result["output"]

    def test_think_handler(self, tmp_repo):
        result = handlers.think_handler(tmp_repo, {"question": "why?"})
        assert result["status"] == "success"
