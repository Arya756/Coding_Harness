"""RepoNavigator: compact repo overview, search, and surgical reads."""

from __future__ import annotations

import os
from typing import Any

from ..tools import handlers

_LANG_MARKERS = [
    ("package.json", "JS/Node"),
    ("yarn.lock", "JS (Yarn)"),
    ("pnpm-lock.yaml", "JS (pnpm)"),
    ("requirements.txt", "Python"),
    ("Pipfile", "Python (Pipenv)"),
    ("pyproject.toml", "Python"),
    ("poetry.lock", "Python (Poetry)"),
    ("Cargo.toml", "Rust"),
    ("go.mod", "Go"),
    ("Gemfile", "Ruby"),
    ("composer.json", "PHP"),
    ("CMakeLists.txt", "C/C++"),
    ("Makefile", "C/C++/generic"),
    ("pom.xml", "Java (Maven)"),
    ("build.gradle", "Java (Gradle)"),
]

_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".harness", "dist", "build"}


class RepoNavigator:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _detect_lang(self) -> list[str]:
        langs = []
        try:
            entries = set(os.listdir(self.repo_path))
        except OSError:
            return langs
        for marker, label in _LANG_MARKERS:
            if marker in entries:
                langs.append(label)
        for entry in entries:
            if entry.endswith(".csproj"):
                langs.append("C#")
                break
        return langs or ["Unknown"]

    def overview(self, max_dirs: int = 5, max_files_per_dir: int = 10) -> dict:
        try:
            root_entries = sorted(os.listdir(self.repo_path))
        except OSError as exc:
            return {"status": "error", "output": str(exc)}

        root_files = [e for e in root_entries if os.path.isfile(os.path.join(self.repo_path, e))]
        root_dirs = [
            e for e in root_entries
            if os.path.isdir(os.path.join(self.repo_path, e)) and e not in _IGNORE_DIRS
        ]

        lines = [f"Languages detected: {', '.join(self._detect_lang())}", ""]
        lines.append("Root files:")
        for f in root_files[:20]:
            lines.append(f"  {f}")
        if len(root_files) > 20:
            lines.append(f"  ... and {len(root_files) - 20} more files")

        lines.append("")
        lines.append("Top-level directories:")
        for d in root_dirs[:max_dirs]:
            lines.append(f"  {d}/")
            sub_path = os.path.join(self.repo_path, d)
            try:
                sub_entries = sorted(os.listdir(sub_path))
            except OSError:
                sub_entries = []
            sub_entries = [e for e in sub_entries if e not in _IGNORE_DIRS]
            for e in sub_entries[:max_files_per_dir]:
                suffix = "/" if os.path.isdir(os.path.join(sub_path, e)) else ""
                lines.append(f"    {e}{suffix}")
            if len(sub_entries) > max_files_per_dir:
                lines.append(f"    ... and {len(sub_entries) - max_files_per_dir} more")
        if len(root_dirs) > max_dirs:
            lines.append(f"  ... and {len(root_dirs) - max_dirs} more directories")

        return {"status": "success", "output": "\n".join(lines)}

    def find(self, query: str, max_results: int = 20) -> dict:
        result = handlers.grep_handler(self.repo_path, {"pattern": query, "max_results": max_results})
        result["query"] = query
        result["count"] = result.get("output", "").count("\n") + 1 if result.get("status") == "success" else 0
        return result

    def read(self, path: str, mode: str = "outline", start: Any = None, end: Any = None) -> dict:
        args = {"path": path, "mode": mode}
        if start is not None:
            args["start"] = start
        if end is not None:
            args["end"] = end
        result = handlers.read_file_handler(self.repo_path, args)
        result["mode"] = mode
        result["path"] = path
        return result
