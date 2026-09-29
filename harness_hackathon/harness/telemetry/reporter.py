"""ASCII telemetry report generator."""

from __future__ import annotations

import os
from typing import Optional


def _box_line(text: str, width: int = 78) -> str:
    return "| " + text.ljust(width - 4) + " |"


class Reporter:
    def __init__(self, repo_path: str, state, telemetry_collector):
        self.repo_path = repo_path
        self.state = state
        self.telemetry = telemetry_collector

    def generate_report(self, output_file: Optional[str] = None) -> str:
        s = self.telemetry.get_summary()
        width = 78
        lines = []
        border = "+" + "-" * (width - 2) + "+"
        lines.append(border)
        lines.append(_box_line(f"Issue: {self.state.issue[:75]}"))
        lines.append(_box_line(f"Session: {self.state.session_id}"))
        lines.append(_box_line(f"Repo: {self.repo_path}"))
        lines.append(border)

        status_symbol = "\u2713" if self.state.resolved else "\u2717"
        status_word = {
            "resolved": "RESOLVED",
            "failed": "FAILED",
            "stuck": "STUCK",
            "running": "RUNNING",
        }.get(self.state.status, self.state.status.upper())
        lines.append(_box_line(f"Status: {status_symbol} {status_word}"))
        if self.state.failure_reason:
            lines.append(_box_line(f"Failure reason: {self.state.failure_reason}"))
        if s.get("no_tests") and self.state.resolved:
            lines.append(_box_line("NOTE: Resolved WITHOUT tests -- no test suite detected"))
        lines.append(border)

        lines.append(_box_line("Metrics"))
        lines.append(_box_line(f"  Steps: {s['total_steps']}  LLM calls: {s['llm_calls']}  "
                                f"Tool calls: {s['tool_calls']}"))
        lines.append(_box_line(f"  Test runs: {s['test_runs']}  Edits: {s['edits']}  "
                                f"Rollbacks: {s['rollbacks']}  Recoveries: {s['recoveries']}"))
        lines.append(border)

        lines.append(_box_line("Tokens & Cost"))
        lines.append(_box_line(f"  In: {s['tokens_in']}  Out: {s['tokens_out']}  "
                                f"Total: {s['total_tokens']}  Cost: ${s['total_cost_usd']:.6f}"))
        lines.append(border)

        lines.append(_box_line("Tool Usage"))
        for tool, count in sorted(s["tool_usage"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(_box_line(f"  {tool}: {count}"))
        lines.append(border)

        lines.append(_box_line(f"Files: read={s['files_read']}  modified={s['files_modified']}"))
        for f in self.state.files_modified[-5:]:
            lines.append(_box_line(f"  - {f.get('path')}"))
        lines.append(border)

        lines.append(_box_line("Key Decisions"))
        for d in self.state.decisions[-5:]:
            lines.append(_box_line(f"  - {d.get('decision', '')[:70]}"))
        lines.append(border)

        lines.append(_box_line("Recovery Events"))
        for f in self.state.failure_history[-5:]:
            lines.append(_box_line(f"  - {f.get('type')} attempt={f.get('attempt')} "
                                    f"{str(f.get('resolution', ''))[:75]}"))
        lines.append(border)

        lines.append(_box_line("Result"))
        lines.append(_box_line(f"  Commit: {self.state.final_commit or '(none)'}"))
        lines.append(_box_line(f"  Tests passed: {self.state.tests_passed}"))
        if self.state.tests_output:
            lines.append(_box_line(f"  Output: {self.state.tests_output[:75]}"))
        lines.append(border)

        report = "\n".join(lines)

        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report)

        return report
