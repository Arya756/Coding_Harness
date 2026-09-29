"""Context assembly, compression, and token estimation for the LLM prompt."""

from __future__ import annotations

import json
from typing import Any, Optional


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ContextManager:
    def __init__(self, state, repo_path: str):
        self.state = state
        self.repo_path = repo_path

    # ------------------------------------------------------------------
    def assemble(self, tool_schemas: list, recent_tools: list, recent_llm_turns: list,
                 custom_instructions: Optional[str] = None) -> dict:
        system_prompt = self._build_system_prompt(tool_schemas, custom_instructions)
        issue_prompt = self._build_issue_prompt()

        messages = [{"role": "user", "content": issue_prompt}]

        for tr in recent_tools[-5:]:
            content = self._format_tool_result(tr)
            messages.append({"role": "user", "content": content})

        for turn in recent_llm_turns[-3:]:
            messages.append(turn)

        total_text = system_prompt + "".join(m.get("content", "") for m in messages)
        estimated_tokens = _rough_tokens(total_text)

        return {
            "system": system_prompt,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "estimated_tokens": estimated_tokens,
        }

    def _format_tool_result(self, tr: dict) -> str:
        tool = tr.get("tool", "?")
        result = tr.get("result", {})
        output = str(result.get("output", ""))
        if len(output) > 3000:
            output = output[:3000] + "... [truncated]"
        status = result.get("status", "?")
        return f"[tool_result tool={tool} status={status}]\n{output}"

    # ------------------------------------------------------------------
    def _build_system_prompt(self, tool_schemas: list, custom_instructions: Optional[str]) -> str:
        tool_names = [t.get("name", "?") for t in tool_schemas] if tool_schemas else []
        sections = [
            "You are an autonomous coding agent operating inside a real repository.",
            "",
            "## Available Tools",
            "Priority order: list_dir/read_file(outline) -> grep -> read_file(signatures) "
            "-> read_file(full) -> edit_file/edit_file_multi -> run_tests -> done.",
            f"Tools: {', '.join(tool_names) if tool_names else '(see schemas)'}",
            "",
            "## Workflow",
            "1. Understand the repo structure before editing.",
            "2. Locate the exact code relevant to the issue using grep/read_file.",
            "3. Make a minimal, exact-string edit_file or edit_file_multi change.",
            "4. Your edit will be automatically syntax/lint verified before tests run.",
            "5. Run tests after every edit. Only call 'done' once tests actually pass.",
            "6. If an edit breaks syntax or regresses a passing test, it will be rolled "
            "back automatically -- re-examine before retrying.",
            "",
            "## Token Efficiency",
            "Prefer outline/signature reads over full reads. Do not re-read unchanged files.",
            "",
            "## Recovery",
            "If a tool fails repeatedly, you will be asked to replan. Try a genuinely "
            "different approach, not a minor variation.",
        ]
        if tool_schemas:
            sections.append("")
            sections.append("## Tool Schemas")
            sections.append(json.dumps(tool_schemas)[:3000])
        if custom_instructions:
            sections.append("")
            sections.append("## Custom Instructions")
            sections.append(custom_instructions)
        sections.append("")
        sections.append("## Output Format")
        sections.append(
            'Respond with exactly one JSON action: {"name": "<tool>", "arguments": {...}, '
            '"goal": "..."} wrapped in a ```json fence. Use name="done" when finished, '
            'name="replan" to revise your plan, or name="think" to reason privately.'
        )
        return "\n".join(sections)

    def _build_issue_prompt(self) -> str:
        state = self.state
        lines = [f"## Issue\n{state.issue}", ""]
        if state.plan:
            lines.append("## Plan")
            for step in state.plan:
                lines.append(f"  {step.get('step')}. [{step.get('tool')}] {step.get('goal', '')}")
            lines.append("")
        lines.append("## Progress")
        lines.append(f"  step={state.step} files_read={len(state.files_read)} "
                      f"files_modified={len(state.files_modified)} tests_passed={state.tests_passed}")
        lines.append("")
        if state.decisions:
            lines.append("## Recent Decisions")
            for d in state.decisions[-3:]:
                lines.append(f"  - {d.get('decision')}: {d.get('rationale', '')}")
            lines.append("")
        prior_notes = getattr(state, "_prior_session_notes", None)
        if prior_notes:
            lines.append("## Prior-Session Notes")
            for note in prior_notes[:5]:
                lines.append(f"  - {note}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def estimate_tokens(self, obj: Any) -> int:
        try:
            text = json.dumps(obj, default=str)
        except (TypeError, ValueError):
            text = str(obj)
        return _rough_tokens(text)

    def compress_history(self, messages: list, keep_last: int = 2) -> list:
        try:
            from contextzip import compress_messages  # type: ignore

            compressed = compress_messages(messages)
            if compressed:
                return compressed
        except Exception:
            pass
        return messages[-keep_last:]

    def compress_prompt(self, text: str) -> str:
        try:
            from less_tokens import compress  # type: ignore

            result = compress(text)
            if result:
                return result
        except Exception:
            pass
        return text
