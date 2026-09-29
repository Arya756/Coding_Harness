"""Main agent loop: plans, executes tools, verifies edits, recovers, commits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from typing import Any, Optional

from .config import (
    CONSECUTIVE_READ_LIMIT,
    COMPACTION_TOKEN_THRESHOLD,
    DEFAULT_MODEL_LABEL,
    MAX_SESSION_SECONDS,
    MAX_STEPS,
    MEMORY_DIR,
    PROMPT_COMPRESS_CHAR_THRESHOLD,
    STATE_FILE,
    TOOL_NAMES,
)
from .context.manager import ContextManager
from .models import Action, _action
from .navigator.repo_navigator import RepoNavigator
from .recovery.engine import FailureType, RecoveryEngine, classify_failure
from .state import TaskState, now_iso
from .telemetry.collector import TelemetryCollector, TokenTracker
from .telemetry.reporter import Reporter
from .tools.registry import execute_tool, get_tool_schemas, reset_budget_tracker
from .verification.verifier import verify_edit

_READ_LIKE_TOOLS = {"read_file", "grep", "list_dir", "get_file_info"}


class Orchestrator:
    def __init__(
        self,
        repo_path: str,
        issue: str,
        budget_usd: Optional[float] = None,
        max_steps: int = MAX_STEPS,
        system_prompt: Optional[str] = None,
        resume: bool = False,
        provider=None,
        max_seconds: Optional[int] = None,
        require_approval: bool = False,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.budget_usd = budget_usd
        self.max_steps = max_steps
        self.custom_system_prompt = system_prompt
        self.provider = provider
        self.max_seconds = max_seconds if max_seconds is not None else MAX_SESSION_SECONDS
        self.require_approval = require_approval

        state_path = os.path.join(self.repo_path, STATE_FILE)
        if resume and os.path.exists(state_path):
            self.state = TaskState.load(state_path)
        else:
            self.state = TaskState(issue=issue, repo_path=self.repo_path)

        self.navigator = RepoNavigator(self.repo_path)
        self.context_mgr = ContextManager(self.state, self.repo_path)
        self.token_tracker = TokenTracker(DEFAULT_MODEL_LABEL)
        self.telemetry = TelemetryCollector(self.repo_path, self.state, self.token_tracker)
        self.recovery = RecoveryEngine(self.state)
        self.reporter = Reporter(self.repo_path, self.state, self.telemetry)
        self.budget_tracker = reset_budget_tracker()

        self._pre_edit_content: dict = {}
        self._consecutive_reads = 0
        self.recent_llm_turns: list = []
        self._session_start_time = time.time()
        self._load_prior_session_notes()

    # ------------------------------------------------------------------
    def _abs(self, rel_path: str) -> str:
        return os.path.join(self.repo_path, rel_path)

    def _state_path(self) -> str:
        return os.path.join(self.repo_path, STATE_FILE)

    def _memory_path(self) -> str:
        return os.path.join(self.repo_path, MEMORY_DIR, "decisions.jsonl")

    def _issue_hash(self) -> str:
        return hashlib.sha256(self.state.issue.encode("utf-8")).hexdigest()[:16]

    def _load_prior_session_notes(self) -> None:
        path = self._memory_path()
        notes: list = []
        if os.path.exists(path):
            try:
                issue_hash = self._issue_hash()
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if entry.get("issue_hash") == issue_hash:
                            notes.append(entry.get("note", ""))
            except OSError:
                pass
        self.state._prior_session_notes = notes[-5:]  # type: ignore[attr-defined]

    def _save_prior_session_note(self) -> None:
        path = self._memory_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            outcome = "resolved" if self.state.resolved else (self.state.status or "unknown")
            note = f"Session {self.state.session_id[:8]} outcome={outcome}"
            if self.state.decisions:
                note += f"; last decision: {self.state.decisions[-1].get('decision', '')[:120]}"
            entry = {"issue_hash": self._issue_hash(), "note": note, "timestamp": now_iso()}
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    def run(self) -> dict:
        self._session_start_time = time.time()
        self.telemetry.log_session_start(self.provider.model_name if self.provider else DEFAULT_MODEL_LABEL,
                                          self.budget_usd)

        if not self.state.plan:
            self._generate_initial_plan()

        while self.state.step < self.max_steps and self.state.status == "running":
            self.state.step += 1

            elapsed = time.time() - self._session_start_time
            if elapsed > self.max_seconds:
                self.state.status = "stuck"
                self.state.failure_reason = "session_timeout"
                self.state.record_decision("Session wall-clock limit exceeded", f"elapsed={elapsed:.0f}s")
                break

            if self.budget_usd is not None:
                projected = self._estimate_next_call_cost()
                if self.token_tracker.total_spend() + projected > self.budget_usd:
                    self.state.status = "failed"
                    self.state.failure_reason = "budget"
                    self.state.record_decision("Budget exceeded", f"spend={self.token_tracker.total_spend():.4f}")
                    break

            tool_schemas = get_tool_schemas()
            recent_tools = self.state.tool_results[-5:]

            ctx = self.context_mgr.assemble(tool_schemas, recent_tools, self.recent_llm_turns[-3:],
                                             self.custom_system_prompt)

            total_len = sum(len(m.get("content", "")) for m in ctx["messages"])
            if total_len > PROMPT_COMPRESS_CHAR_THRESHOLD:
                for m in ctx["messages"]:
                    if m["role"] == "user":
                        m["content"] = self.context_mgr.compress_prompt(m["content"])

            llm_result = self._call_llm(ctx["messages"], tools=tool_schemas)
            if llm_result is None:
                self.state.status = "failed"
                self.state.failure_reason = "llm_error"
                break

            self.recent_llm_turns.append({"role": "assistant", "content": llm_result["response"].text})

            action = self._parse_action(llm_result["response"])

            self._track_consecutive_reads(action)

            self._execute_action(action)

            if ctx["estimated_tokens"] > COMPACTION_TOKEN_THRESHOLD:
                self._compact_context()

            self._save_state()

            if self.state.status != "running":
                break

        self.telemetry.log_session_end()
        self.telemetry.flush()
        report = self.reporter.generate_report(
            output_file=os.path.join(self.repo_path, ".harness", "report.txt")
        )
        self._save_prior_session_note()
        self._save_state()

        return {
            "status": self.state.status,
            "report": report,
            "summary": self.telemetry.get_summary(),
            "state": self.state,
        }

    # ------------------------------------------------------------------
    def _track_consecutive_reads(self, action: Action) -> bool:
        """Update the consecutive-read counter; inject a force-edit nudge at the limit.

        Returns True if the force-edit push fired this call.
        """
        if action.name in _READ_LIKE_TOOLS:
            self._consecutive_reads += 1
        elif action.name in ("edit_file", "edit_file_multi"):
            self._consecutive_reads = 0

        if self._consecutive_reads >= CONSECUTIVE_READ_LIMIT:
            self.recent_llm_turns.append({
                "role": "user",
                "content": "You have read the relevant files. Now produce an edit_file "
                           "tool call to fix the issue.",
            })
            self._consecutive_reads = 0
            return True
        return False

    def _estimate_next_call_cost(self) -> float:
        model = self.provider.model_name if self.provider else DEFAULT_MODEL_LABEL
        # Conservative estimate for the upcoming call.
        return self.token_tracker.estimate_cost(model, 4000, 1500)

    # ------------------------------------------------------------------
    def _generate_initial_plan(self) -> None:
        overview = self.navigator.overview()
        tool_schemas = get_tool_schemas()
        plan_prompt = (
            f"Issue:\n{self.state.issue}\n\n"
            f"Repository overview:\n{overview.get('output', '')}\n\n"
            f"Available tools: {json.dumps(tool_schemas)[:3000]}\n\n"
            "Produce a JSON array of plan steps, each: "
            '{"step": <int>, "tool": "<tool_name>", "args": {...}, "goal": "<why>"}. '
            "Respond with ONLY the JSON array, optionally wrapped in a ```json fence."
        )

        plan_text = ""
        if self.provider:
            try:
                resp = self.provider.chat([{"role": "user", "content": plan_prompt}], max_tokens=1500)
                plan_text = resp.text
                cost = self.token_tracker.estimate_cost(self.provider.model_name, resp.input_tokens, resp.output_tokens)
                self.token_tracker.add_spend(cost)
                self.state.record_llm_call(self.provider.model_name, plan_prompt[:200],
                                            resp.input_tokens, resp.output_tokens, cost)
                self.telemetry.log_llm_call(
                    model=self.provider.model_name,
                    in_tok=resp.input_tokens, out_tok=resp.output_tokens,
                    cost=cost, prompt_preview=plan_prompt[:200], response_preview=plan_text[:200],
                )
            except Exception as exc:
                self.state.record_decision("Initial plan LLM call failed", str(exc)[:200])

        plan = self._parse_plan_text(plan_text)
        if not plan:
            plan = self._default_plan()
        self.state.plan = plan

    def _parse_plan_text(self, text: str) -> Optional[list]:
        if not text:
            return None
        from .models import _decode_json, _extract_balanced

        raw = _decode_json(text)
        if not raw:
            return None
        balanced = _extract_balanced(raw) or raw
        try:
            parsed = json.loads(balanced)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None

        cleaned = []
        for i, step in enumerate(parsed, 1):
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            if tool not in TOOL_NAMES:
                self.state.record_decision(f"Dropped invalid plan step tool={tool}", "not a known tool")
                continue
            cleaned.append({
                "step": step.get("step", i),
                "tool": tool,
                "args": step.get("args", {}),
                "goal": step.get("goal", ""),
            })
        return cleaned or None

    def _default_plan(self) -> list:
        return [
            {"step": 1, "tool": "list_dir", "args": {}, "goal": "Understand repo structure"},
            {"step": 2, "tool": "grep", "args": {"pattern": self.state.issue[:50]}, "goal": "Search relevant code"},
            {"step": 3, "tool": "think", "args": {"question": f"How to solve: {self.state.issue}"}, "goal": "Diagnose"},
        ]

    # ------------------------------------------------------------------
    def _call_llm(self, messages: list, tools: Optional[list] = None) -> Optional[dict]:
        if not self.provider:
            return None
        try:
            resp = self.provider.chat(messages, max_tokens=4096, temperature=0.7, tools=tools)
        except Exception as exc:
            self.state.record_decision("LLM call failed", str(exc)[:300])
            return None

        cost = self.token_tracker.estimate_cost(self.provider.model_name, resp.input_tokens, resp.output_tokens)
        self.token_tracker.add_spend(cost)
        self.state.record_llm_call(self.provider.model_name, "(loop turn)", resp.input_tokens, resp.output_tokens, cost)
        self.telemetry.log_llm_call(model=self.provider.model_name, in_tok=resp.input_tokens,
                                     out_tok=resp.output_tokens, cost=cost,
                                     prompt_preview="(loop turn)", response_preview=resp.text[:200])
        return {"response": resp, "tool_calls": resp.tool_calls}

    # ------------------------------------------------------------------
    def _parse_action(self, response) -> Action:
        if response.tool_calls:
            tc = response.tool_calls[0]
            try:
                return Action(name=tc.get("name", "think"), arguments=tc.get("args", {}), goal="")
            except Exception:
                pass

        act = _action(response.text)
        if act is not None:
            return act

        return self._infer_action(response.text)

    def _infer_action(self, text: str) -> Action:
        low = (text or "").lower()
        if "read_file" in low or "read file" in low:
            m = re.search(r"[\w./\-]+\.\w+", text)
            path = m.group(0) if m else ""
            return Action(name="read_file", arguments={"path": path, "mode": "outline"},
                          goal="inferred from text")
        if "grep" in low or "search for" in low or "find" in low:
            return Action(name="grep", arguments={"pattern": self.state.issue[:50]},
                          goal="inferred from text")
        if "edit" in low or "change" in low or "fix" in low:
            return Action(name="think", arguments={"question": "Need exact old_str/new_str to edit."},
                          goal="inferred from text - insufficient detail to edit safely")
        if "run test" in low or "test" in low:
            return Action(name="run_tests", arguments={}, goal="inferred from text")
        if "done" in low or "resolved" in low or "fixed" in low:
            return Action(name="done", arguments={}, goal="inferred from text")
        if "replan" in low or "new plan" in low:
            return Action(name="replan", arguments={}, goal="inferred from text")
        return Action(name="think",
                       arguments={"question": f"Could not parse an action from model output: {text[:300]}"},
                       goal="fallback")

    # ------------------------------------------------------------------
    def _execute_action(self, action: Action) -> None:
        if action.name == "think":
            self._log_thought(action.arguments.get("question", ""))
            return
        if action.name == "done":
            result = self._mark_resolved()
            if result.get("status") == "error":
                self.recent_llm_turns.append({"role": "user", "content": result["output"]})
            return
        if action.name == "replan":
            self._handle_replan()
            return

        if action.name not in TOOL_NAMES:
            self.recent_llm_turns.append({
                "role": "user",
                "content": f"Unknown action '{action.name}'. Valid tools: {', '.join(TOOL_NAMES)}, done, replan.",
            })
            return

        self._execute_tool_action(action)

    def _log_thought(self, question: str) -> None:
        self.state.record_decision("thought", question[:300])

    def _handle_replan(self) -> None:
        overview = self.navigator.overview()
        prompt = (
            f"Issue:\n{self.state.issue}\n\nCurrent overview:\n{overview.get('output', '')}\n\n"
            "Produce a revised JSON array plan as before."
        )
        result = self._call_llm([{"role": "user", "content": prompt}])
        if result:
            plan = self._parse_plan_text(result["response"].text)
            if plan:
                self.state.plan = plan
                self.state.record_decision("Replanned", "generated new plan")

    def _execute_tool_action(self, action: Action) -> None:
        tool = action.name
        args = action.arguments

        if tool == "read_file":
            path = args.get("path")
            mode = args.get("mode", "outline")
            if path:
                cached = self.state.was_read(path, mode)
                if cached is not None:
                    msg = (f"Already read {path} ({mode}) at step {cached['step']}. "
                           f"Summary: {cached['digest']}. Re-read only if you need exact current lines.")
                    result = {"status": "success", "output": msg, "tool": tool, "args": args}
                    self.state.record_tool_call(tool, args, result)
                    self.telemetry.log_tool_call(tool, args, "success", output_preview=msg)
                    self.recent_llm_turns.append({"role": "user", "content": self._format_tool_feedback(tool, result)})
                    return

        if tool in ("edit_file", "edit_file_multi"):
            paths = [args.get("path")] if tool == "edit_file" else [args.get("path")]
            self._checkpoint_files([p for p in paths if p])

        result = execute_tool(tool, args, self.repo_path, self.budget_tracker)
        self.state.record_tool_call(tool, args, result)
        self.telemetry.log_tool_call(tool, args, result.get("status", "?"),
                                      duration_ms=result.get("duration_ms", 0),
                                      output_preview=str(result.get("output", "")))

        if tool == "read_file" and result.get("status") == "success":
            path = args.get("path")
            mode = args.get("mode", "outline")
            if path:
                digest = str(result.get("output", ""))[:150].replace("\n", " ")
                self.state.mark_read(path, mode, self.state.step, digest)
                self.state.files_read.append({"path": path, "mode": mode, "timestamp": now_iso()})

        if result.get("status") == "error":
            rec = self.recovery.handle(result)
            self.telemetry.log_recovery(rec["failure_type"], rec["action"], rec["message"])
            self._apply_recovery(rec, result)
            return

        if tool in ("edit_file", "edit_file_multi"):
            self._handle_post_edit(args)
            return

        self.recent_llm_turns.append({"role": "user", "content": self._format_tool_feedback(tool, result)})

    def _format_tool_feedback(self, tool: str, result: dict) -> str:
        output = str(result.get("output", ""))
        return f"[result of {tool}] status={result.get('status')}\n{output[:2000]}"

    # ------------------------------------------------------------------
    def _checkpoint_files(self, paths: list) -> None:
        for p in paths:
            abs_p = self._abs(p)
            self._pre_edit_content[p] = open(abs_p).read() if os.path.exists(abs_p) else None
            self.state.checkpoints[p] = self._pre_edit_content[p]

    def _rollback(self, path: str) -> None:
        prev = self._pre_edit_content.get(path, self.state.checkpoints.get(path))
        abs_p = self._abs(path)
        if prev is None:
            if os.path.exists(abs_p):
                os.remove(abs_p)
        else:
            with open(abs_p, "w") as f:
                f.write(prev)
        self.state.invalidate_read_cache(path)
        self.state.record_decision(f"Rolled back edit to {path}",
                                    "Edit caused syntax error or test regression")
        self.telemetry.log_rollback(path, "syntax error or test regression")

    def _is_regression(self, tests_before: Optional[bool], test_result: dict) -> bool:
        return tests_before is True and test_result.get("passed") is False

    # ------------------------------------------------------------------
    def _handle_post_edit(self, args: dict) -> None:
        path = args.get("path")
        old_preview = str(args.get("old_str", ""))
        new_preview = str(args.get("new_str", ""))
        if "edits" in args:
            old_preview = json.dumps(args.get("edits", []))[:200]
            new_preview = ""

        self.state.files_modified.append({"path": path, "timestamp": now_iso(),
                                           "diff_summary": f"{old_preview[:80]} -> {new_preview[:80]}"})
        self.state.invalidate_read_cache(path)
        self.telemetry.log_edit(path, old_preview, new_preview)

        v = verify_edit(self.repo_path, path)
        self.telemetry.log_verify(path, v["passed"], v["summary"])
        if not v["passed"]:
            fake_result = {"tool": "edit_file", "status": "error", "output": v["summary"],
                            "failure_hint": "syntax"}
            rec = self.recovery.handle(fake_result)
            self.telemetry.log_recovery(rec["failure_type"], rec["action"], rec["message"])
            self._apply_recovery(rec, fake_result, edited_path=path)
            return

        tests_before = self.state.tests_passed
        t = self._run_tests()

        if self._is_regression(tests_before, t):
            fake_result = {"tool": "run_tests", "status": "error", "output": t.get("output", ""),
                            "failure_hint": "regression"}
            rec = self.recovery.handle(fake_result)
            self.telemetry.log_recovery(rec["failure_type"], rec["action"], rec["message"])
            self._apply_recovery(rec, fake_result, edited_path=path)
            return

        self.recent_llm_turns.append({
            "role": "user",
            "content": f"Edit to {path} verified OK. {v['summary']} Tests passed={t.get('passed')}.",
        })

    def _apply_recovery(self, rec: dict, result: dict, edited_path: Optional[str] = None) -> None:
        action = rec["action"]
        if action == "retry":
            self.recent_llm_turns.append({"role": "user", "content": rec["prompt_addition"]})
        elif action == "replan":
            self.recent_llm_turns.append({"role": "user", "content": rec["prompt_addition"]})
            self._handle_replan()
        elif action == "rollback":
            path = edited_path or result.get("args", {}).get("path")
            if path:
                self._rollback(path)
            self.recent_llm_turns.append({"role": "user", "content": rec["prompt_addition"]})
        elif action == "compact":
            self._compact_context()
        elif action == "escalate":
            self.state.status = "stuck"
            self.state.failure_reason = "escalated"
            self.recent_llm_turns.append({"role": "user", "content": rec["prompt_addition"]})

    # ------------------------------------------------------------------
    def _run_tests(self) -> dict:
        result = execute_tool("run_tests", {}, self.repo_path, self.budget_tracker)
        self.state.record_tool_call("run_tests", {}, result)
        passed = bool(result.get("passed"))
        no_tests = bool(result.get("no_tests"))
        self.state.tests_passed = passed
        self.state.tests_output = str(result.get("output", ""))[:2000]
        self.state.no_tests = no_tests
        self.telemetry.log_test_run(passed, result.get("exit_code"), result.get("duration_ms", 0),
                                     output_preview=str(result.get("output", "")), no_tests=no_tests)
        return result

    # ------------------------------------------------------------------
    def _mark_resolved(self) -> dict:
        if self.state.tests_passed is not True:
            t = self._run_tests()
            if not t.get("passed"):
                msg = (
                    "Cannot mark resolved: tests are not passing.\n"
                    f"{str(t.get('output', ''))[:800]}\n"
                    "Fix the failing tests, then call done again."
                )
                return {"status": "error", "output": msg}

        try:
            subprocess.run(["git", "add", "-A"], cwd=self.repo_path, capture_output=True, text=True)
            commit = subprocess.run(
                ["git", "commit", "-m", f"fix: {self.state.issue[:50]}"],
                cwd=self.repo_path, capture_output=True, text=True,
            )
            m = re.search(r"[a-f0-9]{7,40}", commit.stdout)
            self.state.final_commit = m.group(0) if m else None
        except (OSError, FileNotFoundError):
            self.state.final_commit = None

        self.state.resolved = True
        self.state.status = "resolved"
        return {"status": "success", "output": f"Committed {self.state.final_commit}"}

    # ------------------------------------------------------------------
    def _compact_context(self) -> None:
        results = self.state.tool_results
        if len(results) <= 10:
            return
        before = self.context_mgr.estimate_tokens(results)
        old, recent = results[:-10], results[-10:]
        digest = self._summarize_results(old)
        self.state.tool_results = [
            {"tool": "summary", "args": {}, "result": {"status": "success", "output": digest},
             "timestamp": now_iso()}
        ] + recent
        after = self.context_mgr.estimate_tokens(self.state.tool_results)
        self.telemetry.log_compaction(before, after)

    def _summarize_results(self, old: list) -> str:
        reads, edits, tests_fail, tests_pass, other = 0, 0, 0, 0, 0
        read_paths, edit_paths = [], []
        for entry in old:
            tool = entry.get("tool")
            result = entry.get("result", {})
            args = entry.get("args", {})
            if tool == "read_file":
                reads += 1
                p = args.get("path")
                if p and p not in read_paths:
                    read_paths.append(p)
            elif tool in ("edit_file", "edit_file_multi"):
                edits += 1
                p = args.get("path")
                if p and p not in edit_paths:
                    edit_paths.append(p)
            elif tool == "run_tests":
                if result.get("passed"):
                    tests_pass += 1
                else:
                    tests_fail += 1
            else:
                other += 1
        parts = []
        if reads:
            parts.append(f"Read {reads} files ({', '.join(read_paths[:5])})")
        if edits:
            parts.append(f"{edits} edits to {', '.join(edit_paths[:5])}")
        if tests_fail:
            parts.append(f"{tests_fail} failed test run(s)")
        if tests_pass:
            parts.append(f"{tests_pass} passed test run(s)")
        if other:
            parts.append(f"{other} other tool calls")
        return "; ".join(parts) if parts else "No significant activity."

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        self.state.save(self._state_path())
        self.telemetry.log_state_save()
