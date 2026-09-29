"""Post-edit verification: syntax check then lint check (warnings only)."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

_AUTO_PASS_EXTS = set()


def _run(cmd: list, cwd: str, timeout: int = 20) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        return ok, (proc.stdout + "\n" + proc.stderr).strip()
    except FileNotFoundError:
        return True, f"tool not available: {cmd[0]} (skipped)"
    except subprocess.TimeoutExpired:
        return False, "verification timed out"
    except OSError as exc:
        return False, str(exc)


def _syntax_check(repo_path: str, abs_path: str, ext: str) -> tuple[bool, str]:
    if ext == ".py":
        return _run(["python3", "-m", "py_compile", abs_path], repo_path)
    if ext in (".js", ".jsx"):
        if shutil.which("node"):
            return _run(["node", "--check", abs_path], repo_path)
        return True, "node not available (skipped)"
    if ext in (".ts", ".tsx"):
        if shutil.which("npx"):
            return _run(["npx", "--yes", "tsc", "--noEmit", abs_path], repo_path, timeout=40)
        return True, "tsc not available (skipped)"
    if ext == ".go":
        if shutil.which("go"):
            return _run(["go", "build", "-n", abs_path], repo_path)
        return True, "go not available (skipped)"
    if ext == ".rs":
        if shutil.which("rustc"):
            return _run(["rustc", "--edition", "2021", "--emit", "metadata", "-o", "/tmp/_verify.rmeta", abs_path], repo_path)
        return True, "rustc not available (skipped)"
    if ext == ".java":
        if shutil.which("javac"):
            return _run(["javac", "-d", "/tmp", abs_path], repo_path)
        return True, "javac not available (skipped)"
    return True, "auto-pass (unsupported extension)"


def detect_test_framework(repo_path: str) -> Optional[str]:
    def exists(name: str) -> bool:
        return os.path.exists(os.path.join(repo_path, name))

    if exists("package.json"):
        return "npm"
    if exists("Cargo.toml"):
        return "cargo"
    if exists("go.mod"):
        return "go"
    for f in ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini"):
        if exists(f):
            return "pytest"
    if exists("Makefile"):
        return "make"
    return None


def _lint_check(repo_path: str, abs_path: str, ext: str) -> tuple[bool, str]:
    if ext == ".py":
        for linter, args in (
            ("ruff", ["ruff", "check", abs_path]),
            ("flake8", ["flake8", abs_path]),
            ("pyflakes", ["pyflakes", abs_path]),
        ):
            if shutil.which(linter):
                ok, out = _run(args, repo_path)
                return ok, out
        return True, "no python linter available (skipped)"
    if ext in (".js", ".jsx", ".ts", ".tsx"):
        if shutil.which("eslint"):
            return _run(["eslint", abs_path], repo_path)
        if shutil.which("npx"):
            return _run(["npx", "--yes", "eslint", abs_path], repo_path, timeout=40)
        return True, "eslint not available (skipped)"
    return True, "auto-pass (no linter configured)"


def verify_edit(repo_path: str, file_path: str) -> dict:
    abs_path = os.path.join(repo_path, file_path)
    ext = os.path.splitext(file_path)[1].lower()
    checks = []

    if not os.path.isfile(abs_path):
        return {"passed": False, "checks": [{"name": "exists", "passed": False}],
                "summary": f"File not found after edit: {file_path}"}

    syntax_ok, syntax_out = _syntax_check(repo_path, abs_path, ext)
    checks.append({"name": "syntax", "passed": syntax_ok, "output": syntax_out[:500]})

    if not syntax_ok:
        return {
            "passed": False,
            "checks": checks,
            "summary": f"Syntax check failed for {file_path}: {syntax_out[:300]}",
        }

    lint_ok, lint_out = _lint_check(repo_path, abs_path, ext)
    checks.append({"name": "lint", "passed": lint_ok, "output": lint_out[:500]})

    summary = f"Syntax OK for {file_path}."
    if not lint_ok:
        summary += f" Lint warnings: {lint_out[:300]}"

    return {"passed": True, "checks": checks, "summary": summary}
