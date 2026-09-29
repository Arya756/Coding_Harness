"""Tool handler implementations. Each handler(repo_path, args) -> dict."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import Any, Optional

from ..config import (
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
    _is_command_allowed,
)

_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_path(repo_path: str, rel: str) -> str:
    rel = rel or "."
    base = os.path.abspath(repo_path)
    target = os.path.abspath(os.path.join(base, rel))
    if not (target == base or target.startswith(base + os.sep)):
        raise ValueError(f"Path traversal blocked: {rel}")
    return target


def _truncated(text: str, max_len: int = 2000) -> str:
    if text is None:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated {len(text) - max_len} chars]"


def _lang_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
        ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go",
        ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".php": "PHP",
        ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".cs": "C#",
        ".md": "Markdown", ".json": "JSON", ".yml": "YAML", ".yaml": "YAML",
    }.get(ext, "Unknown")


def _count_lines(path: str) -> int:
    try:
        with open(path, "r", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


_OUTLINE_PATTERNS = {
    ".py": re.compile(r"^(\s*)(def|class)\s+(\w+)"),
    ".js": re.compile(r"^\s*(export\s+)?(async\s+)?(function\s+(\w+)|class\s+(\w+)|const\s+(\w+)\s*=\s*(\(|async))"),
    ".jsx": None,
    ".ts": re.compile(r"^\s*(export\s+)?(async\s+)?(function\s+(\w+)|class\s+(\w+)|interface\s+(\w+)|type\s+(\w+)|const\s+(\w+)\s*=\s*(\(|async))"),
    ".tsx": None,
    ".go": re.compile(r"^\s*func\s+(\(\w+\s+\*?\w+\)\s+)?(\w+)|^\s*type\s+(\w+)\s+(struct|interface)"),
    ".rs": re.compile(r"^\s*(pub\s+)?(fn|struct|impl|trait|enum)\s+(\w+)"),
    ".java": re.compile(r"^\s*(public|private|protected)?\s*(static\s+)?(class|interface)\s+(\w+)|^\s*(public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\("),
}
_OUTLINE_PATTERNS[".jsx"] = _OUTLINE_PATTERNS[".js"]
_OUTLINE_PATTERNS[".tsx"] = _OUTLINE_PATTERNS[".ts"]


def extract_outline(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    pattern = _OUTLINE_PATTERNS.get(ext)
    outline = []
    if pattern is None:
        return outline
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if pattern.match(line):
                    outline.append({"line": i, "text": line.strip()})
    except OSError:
        pass
    return outline


def _restricted_env() -> dict:
    env = dict(os.environ)
    for key in list(env.keys()):
        upper = key.upper()
        if any(upper.endswith(suf) for suf in _SECRET_SUFFIXES):
            env.pop(key, None)
    safe_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    env["PATH"] = env.get("PATH", safe_path)
    return env


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def read_file_handler(repo_path: str, args: dict) -> dict:
    path = args.get("path")
    mode = args.get("mode", "outline")
    if not path:
        return {"status": "error", "output": "path is required"}
    try:
        abs_path = _repo_path(repo_path, path)
    except ValueError as exc:
        return {"status": "error", "output": str(exc)}

    if not os.path.isfile(abs_path):
        return {"status": "error", "output": f"File not found: {path}"}

    total_lines = _count_lines(abs_path)

    if mode == "full":
        start = args.get("start")
        end = args.get("end")
        try:
            with open(abs_path, "r", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            return {"status": "error", "output": str(exc)}
        start = int(start) if start is not None else 1
        end = int(end) if end else len(lines)
        if start < 1:
            return {"status": "error", "output": "start must be >= 1"}
        if end > len(lines):
            end = len(lines)
        if start > end:
            return {"status": "error", "output": "start must be <= end"}
        snippet = "".join(lines[start - 1:end])
        output = f"# {path} lines {start}-{end} of {len(lines)}\n{snippet}"
        return {"status": "success", "output": _truncated(output, 8000), "mode": mode}

    if mode == "signatures":
        outline = extract_outline(abs_path)
        if not outline:
            try:
                with open(abs_path, "r", errors="replace") as f:
                    lines = f.readlines()
            except OSError as exc:
                return {"status": "error", "output": str(exc)}
            snippet = "".join(lines[:100])
            return {"status": "success", "output": _truncated(f"# {path} (first 100 lines)\n{snippet}"), "mode": mode}
        try:
            with open(abs_path, "r", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            return {"status": "error", "output": str(exc)}
        chunks = []
        for sym in outline:
            ln = sym["line"]
            lo = max(0, ln - 3)
            hi = min(len(lines), ln + 3)
            chunk = "".join(lines[lo:hi])
            chunks.append(f"-- around line {ln} --\n{chunk}")
        output = f"# {path} signatures ({len(outline)} symbols)\n" + "\n".join(chunks)
        return {"status": "success", "output": _truncated(output, 4000), "mode": mode}

    # default: outline
    outline = extract_outline(abs_path)
    if not outline:
        try:
            with open(abs_path, "r", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            return {"status": "error", "output": str(exc)}
        snippet = "".join(lines[:50])
        return {"status": "success", "output": _truncated(f"# {path} (first 50 lines of {total_lines})\n{snippet}"), "mode": mode}
    table = "\n".join(f"  L{s['line']}: {s['text']}" for s in outline)
    output = f"# {path} outline ({total_lines} lines, {len(outline)} symbols)\n{table}"
    return {"status": "success", "output": _truncated(output, 2000), "mode": mode}


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

def grep_handler(repo_path: str, args: dict) -> dict:
    pattern = args.get("pattern")
    if not pattern:
        return {"status": "error", "output": "pattern is required"}
    path = args.get("path", ".")
    max_results = int(args.get("max_results", 100))
    try:
        search_path = _repo_path(repo_path, path)
    except ValueError as exc:
        return {"status": "error", "output": str(exc)}

    rg_bin = shutil.which("rg")
    try:
        if rg_bin:
            cmd = [rg_bin, "--no-heading", "--with-filename", "-n",
                   "--max-count", str(max_results), pattern, search_path]
        else:
            cmd = ["grep", "-rn", "--include=*", pattern, search_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, env=_restricted_env())
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "grep timed out"}
    except OSError as exc:
        return {"status": "error", "output": str(exc)}

    if proc.returncode == 1 and not proc.stdout.strip():
        return {"status": "success", "output": "(no matches)"}
    if proc.returncode not in (0, 1):
        return {"status": "error", "output": proc.stderr.strip() or "grep failed"}

    out_lines = proc.stdout.strip().splitlines()
    rel_lines = [line.replace(repo_path.rstrip("/") + "/", "") for line in out_lines]
    rel_lines = rel_lines[:max_results]
    return {"status": "success", "output": _truncated("\n".join(rel_lines), 4000)}


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------

def run_command_handler(repo_path: str, args: dict) -> dict:
    cmd = args.get("cmd")
    if not cmd:
        return {"status": "error", "output": "cmd is required"}
    timeout = int(args.get("timeout", DEFAULT_COMMAND_TIMEOUT))

    allowed, reason = _is_command_allowed(cmd)
    if not allowed:
        return {"status": "error", "output": f"Blocked: {reason}"}

    try:
        start = time.time()
        proc = subprocess.run(
            cmd, shell=True, cwd=repo_path, timeout=timeout,
            capture_output=True, text=True, env=_restricted_env(),
        )
        duration_ms = int((time.time() - start) * 1000)
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": f"Command timed out after {timeout}s"}
    except OSError as exc:
        return {"status": "error", "output": str(exc)}

    stdout = _truncated(proc.stdout, 5000)
    stderr = _truncated(proc.stderr, 2000)
    output = f"exit_code={proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    status = "success" if proc.returncode == 0 else "error"
    return {"status": status, "output": output, "exit_code": proc.returncode, "duration_ms": duration_ms}


# ---------------------------------------------------------------------------
# edit_file / edit_file_multi
# ---------------------------------------------------------------------------

def edit_file_handler(repo_path: str, args: dict) -> dict:
    path = args.get("path")
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")
    if path is None:
        return {"status": "error", "output": "path is required"}
    try:
        abs_path = _repo_path(repo_path, path)
    except ValueError as exc:
        return {"status": "error", "output": str(exc)}

    if not os.path.exists(abs_path) and old_str == "":
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(new_str)
        return {
            "status": "success",
            "output": f"Created {path} ({len(new_str)} chars)",
            "diff_summary": f"created file with {new_str.count(chr(10)) + 1} lines",
        }

    if not os.path.isfile(abs_path):
        return {"status": "error", "output": f"File not found: {path}"}

    with open(abs_path, "r", errors="replace") as f:
        content = f.read()

    if old_str not in content:
        return {
            "status": "error",
            "output": f"old_str not found in {path}. Re-read with mode='full' for exact content.",
        }

    new_content = content.replace(old_str, new_str, 1)
    with open(abs_path, "w") as f:
        f.write(new_content)

    return {
        "status": "success",
        "output": f"Edited {path}",
        "diff_summary": f"-{old_str[:80]!r} +{new_str[:80]!r}",
    }


def edit_file_multi_handler(repo_path: str, args: dict) -> dict:
    path = args.get("path")
    edits = args.get("edits", [])
    if not path:
        return {"status": "error", "output": "path is required"}
    if not edits:
        return {"status": "error", "output": "edits is required (non-empty list)"}
    try:
        abs_path = _repo_path(repo_path, path)
    except ValueError as exc:
        return {"status": "error", "output": str(exc)}
    if not os.path.isfile(abs_path):
        return {"status": "error", "output": f"File not found: {path}"}

    with open(abs_path, "r", errors="replace") as f:
        content = f.read()

    working = content
    applied = 0
    for edit in edits:
        old_str = edit.get("old_str", "")
        new_str = edit.get("new_str", "")
        if old_str not in working:
            return {
                "status": "error",
                "output": f"Hunk {applied + 1} old_str not found in {path}; no changes written (atomic).",
            }
        working = working.replace(old_str, new_str, 1)
        applied += 1

    with open(abs_path, "w") as f:
        f.write(working)

    return {
        "status": "success",
        "output": f"Applied {applied} edits to {path}",
        "diff_summary": f"{applied} hunks applied",
    }


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------

def _detect_test_command(repo_path: str, pattern: Optional[str] = None) -> Optional[str]:
    def exists(name: str) -> bool:
        return os.path.exists(os.path.join(repo_path, name))

    if exists("package.json"):
        try:
            import json
            with open(os.path.join(repo_path, "package.json")) as f:
                pkg = json.load(f)
            if "test" in pkg.get("scripts", {}):
                cmd = "npm test"
                if pattern:
                    cmd += f" -- --testPathPattern={pattern}"
                return cmd
        except (OSError, ValueError):
            pass
    if exists("Cargo.toml"):
        cmd = "cargo test"
        if pattern:
            cmd += f" {pattern}"
        return cmd
    if exists("go.mod"):
        cmd = "go test ./..."
        if pattern:
            cmd = f"go test -run {pattern} ./..."
        return cmd
    for f in ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini"):
        if exists(f):
            cmd = "pytest"
            if pattern:
                cmd += f" -k {pattern}"
            return cmd
    for f in ("jest.config.js", "jest.config.ts", ".jestrc"):
        if exists(f):
            cmd = "npm test"
            if pattern:
                cmd += f" -- --testPathPattern={pattern}"
            return cmd
    if exists("Makefile"):
        return "make test"
    return None


def run_tests_handler(repo_path: str, args: dict) -> dict:
    pattern = args.get("pattern")
    timeout = int(args.get("timeout", DEFAULT_TEST_TIMEOUT))

    cmd = _detect_test_command(repo_path, pattern)
    if cmd is None:
        return {
            "status": "success",
            "output": "No test framework detected",
            "passed": True,
            "no_tests": True,
            "duration_ms": 0,
        }

    allowed, reason = _is_command_allowed(cmd)
    if not allowed:
        return {"status": "error", "output": f"Blocked test command: {reason}", "passed": False}

    try:
        start = time.time()
        proc = subprocess.run(
            cmd, shell=True, cwd=repo_path, timeout=timeout,
            capture_output=True, text=True, env=_restricted_env(),
        )
        duration_ms = int((time.time() - start) * 1000)
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": f"Tests timed out after {timeout}s", "passed": False}
    except OSError as exc:
        return {"status": "error", "output": str(exc), "passed": False}

    output = _truncated(proc.stdout + "\n" + proc.stderr, 6000)
    passed = proc.returncode == 0
    return {
        "status": "success" if passed else "error",
        "output": output,
        "passed": passed,
        "exit_code": proc.returncode,
        "duration_ms": duration_ms,
    }


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

def list_dir_handler(repo_path: str, args: dict) -> dict:
    path = args.get("path", ".")
    recursive = bool(args.get("recursive", False))
    try:
        abs_path = _repo_path(repo_path, path)
    except ValueError as exc:
        return {"status": "error", "output": str(exc)}
    if not os.path.isdir(abs_path):
        return {"status": "error", "output": f"Not a directory: {path}"}

    if not recursive:
        try:
            entries = sorted(os.listdir(abs_path))
        except OSError as exc:
            return {"status": "error", "output": str(exc)}
        lines = []
        for e in entries:
            suffix = "/" if os.path.isdir(os.path.join(abs_path, e)) else ""
            lines.append(f"{e}{suffix}")
        return {"status": "success", "output": "\n".join(lines)}

    lines = []
    count = 0
    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel_root = os.path.relpath(root, abs_path)
        for f in sorted(files):
            if f.startswith("."):
                continue
            rel = f if rel_root == "." else os.path.join(rel_root, f)
            lines.append(rel)
            count += 1
            if count >= 200:
                lines.append("... [truncated, 200+ entries]")
                return {"status": "success", "output": "\n".join(lines)}
    return {"status": "success", "output": "\n".join(lines)}


# ---------------------------------------------------------------------------
# get_file_info
# ---------------------------------------------------------------------------

def get_file_info_handler(repo_path: str, args: dict) -> dict:
    path = args.get("path")
    if not path:
        return {"status": "error", "output": "path is required"}
    try:
        abs_path = _repo_path(repo_path, path)
    except ValueError as exc:
        return {"status": "error", "output": str(exc)}
    if not os.path.isfile(abs_path):
        return {"status": "error", "output": f"File not found: {path}"}

    size = os.path.getsize(abs_path)
    lines = _count_lines(abs_path)
    lang = _lang_from_ext(abs_path)
    outline = extract_outline(abs_path)[:15]
    symbols = [s["text"] for s in outline]

    output = (
        f"path={path}\nlanguage={lang}\nsize_bytes={size}\nlines={lines}\n"
        f"symbols:\n" + "\n".join(f"  {s}" for s in symbols)
    )
    return {"status": "success", "output": output}


# ---------------------------------------------------------------------------
# think
# ---------------------------------------------------------------------------

def think_handler(repo_path: str, args: dict) -> dict:
    question = args.get("question", "")
    return {"status": "success", "output": f"(thinking) {question}"}
