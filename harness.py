"""A thin ReAct loop for the fact-removal study.

Why a small loop instead of an existing agent framework. The experiment needs
three things a shipped agent fights you on: stop at an exact checkpoint, save
the state, and resume from a context we built by hand with a matched token
budget. Those are a few lines here and a patch set in OpenHands. Section 12 of
the plan keeps shipped agents as later work, which is where they belong.

The model is a seam. Anything that takes messages and returns a reply with a
tool call and real token usage will work. `ScriptedModel` drives the offline
self-check. `AnthropicModel` is the live adapter.

Tools the agent gets, all logged through tracelog:
    read_file(path, start, end)
    search(pattern, scope)
    edit_file(path, old, new)
    run_tests(target)
    finish(summary)

Run `python harness.py` for the self-check. It needs no network and no key.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tracelog import EDIT, READ, SEARCH, TEST, Trace

MAX_HITS = 40
MAX_READ_LINES = 400
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".tox", ".idea"}

SYSTEM = """You are fixing a bug in a Python repository.

Reply with one JSON object and nothing else. No prose, no code fences.

  {"tool": "search",    "args": {"pattern": "...", "scope": "."}}
  {"tool": "read_file", "args": {"path": "...", "start": 1, "end": 120}}
  {"tool": "edit_file", "args": {"path": "...", "old": "...", "new": "..."}}
  {"tool": "run_tests", "args": {}}
  {"tool": "finish",    "args": {"summary": "..."}}

edit_file replaces `old` with `new`. `old` must appear exactly once in the file.
You cannot create files. Only edit files that already exist.
run_tests with no arguments runs the tests for this issue. You can also pass a
target to run something narrower.

Work in small steps. Run run_tests after any change, and run it before you
finish. An untested fix is not a fix."""


@dataclass
class Reply:
    """What a model returns for one step. Usage must be the provider's real count."""

    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ToolResult:
    kind: str
    output: str
    fields: dict = field(default_factory=dict)


class Workspace:
    """Filesystem and test access for one repo, with a path guard.

    The guard is not optional. An agent that walks out of the repo would taint
    the run and could touch the rest of the machine.
    """

    def __init__(self, root: str | Path, *, test_cmd: list[str] | None = None,
                 default_target: list[str] | None = None):
        self.root = Path(root).resolve()
        self.test_cmd = test_cmd or [sys.executable, "-m", "pytest", "-x", "-q"]
        # `run_tests` with no target runs the task's own tests. The agent never
        # has to be told their ids, so the ids stay out of the prompt and the
        # F_test fact can actually be scrubbed from a resume context.
        self.default_target = list(default_target or [])

    def _safe(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"{path!r} is outside the repo")
        return target

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> ToolResult:
        f = self._safe(path)
        if not f.is_file():
            return ToolResult(READ, f"no such file: {path}", {"path": path, "lines": None})
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        lo = max(1, start or 1)
        hi = min(len(lines), end or min(len(lines), lo + MAX_READ_LINES - 1))
        body = "\n".join(lines[lo - 1 : hi])
        return ToolResult(READ, body, {"path": path, "lines": (lo, hi)})

    def search(self, pattern: str, scope: str = ".") -> ToolResult:
        """Emits grep-style `path:line:text`, which is the shape facts.py parses."""
        base = self._safe(scope)
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return ToolResult(SEARCH, f"bad pattern: {exc}", {"pattern": pattern, "scope": scope})
        hits = []
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for f in files:
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = f.relative_to(self.root).as_posix()
            for n, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{rel}:{n}:{line.strip()}")
                    if len(hits) >= MAX_HITS:
                        break
            if len(hits) >= MAX_HITS:
                break
        out = "\n".join(hits) if hits else "no matches"
        return ToolResult(SEARCH, out, {"pattern": pattern, "scope": scope})

    def edit_file(self, path: str, old: str, new: str) -> ToolResult:
        f = self._safe(path)
        if not f.is_file():
            return ToolResult(EDIT, f"no such file: {path}", {"failed": True, "path": path})
        before = f.read_text(encoding="utf-8", errors="replace")
        found = before.count(old)
        if found != 1:
            why = "not found" if found == 0 else f"found {found} times, must be unique"
            return ToolResult(EDIT, f"edit rejected: {why}", {"failed": True, "path": path})
        after = before.replace(old, new, 1)
        f.write_text(after, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{path}", tofile=f"b/{path}",
            )
        )
        return ToolResult(EDIT, f"edited {path}", {"path": path, "post_text": after, "diff": diff})

    def run_tests(self, target: str | list[str] = "") -> ToolResult:
        """`target` may be a list, so a regression suite runs in one invocation."""
        targets = [target] if isinstance(target, str) else list(target)
        targets = [t for t in targets if t] or list(self.default_target)
        cmd = [*self.test_cmd, *targets]
        proc = subprocess.run(
            cmd, cwd=self.root, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        out = (proc.stdout + proc.stderr).strip()
        label = " ".join(targets) if targets else " ".join(self.test_cmd)
        return ToolResult(TEST, out, {"test": label, "passed": proc.returncode == 0})


MIN_CHECKPOINT_CALLS = 8
# how often phase 1 refuses an early finish before accepting it
MAX_PUSHBACKS = 3
# A resume arm that finishes without touching anything is not an observation.
# One astropy arm did exactly that: 0 steps, 0 tokens, and it was recorded as
# "did not re-learn the fact", which is true only because it did nothing.
MIN_RESUME_CALLS = 1
# consecutive unparseable replies tolerated before the run is abandoned
MAX_BAD_REPLIES = 6


NO_EDIT_CAP = 18


def checkpoint_reached(
    events: list[dict],
    min_calls: int = MIN_CHECKPOINT_CALLS,
    no_edit_cap: int = NO_EDIT_CAP,
) -> bool:
    """A failing test, enough work behind it, and an edit if one has happened.

    Always required: a failing test and `min_calls` tool calls. Told which test
    to run, one agent went search, read, edit, test and stopped at four steps
    with a 1,200 token history. There is almost nothing there to delete a fact
    from, and no spare history to pad the other arms back up with.

    An edit is preferred, not required. Negative-knowledge facts only exist
    once a fix has been tried, so an edit makes for a richer checkpoint. But
    demanding one is fragile. Across four phase-1 runs on the same task this
    agent did 3 edits with no test, then an edit after a test, then six edits
    with no re-test, then 26 steps of pure investigation with no edit at all.
    Two of those produced no checkpoint and the whole run was wasted.

    So we wait up to `no_edit_cap` calls for an edit and then stop anyway. A
    checkpoint with no edit is still a good one: the agent has investigated and
    reproduced the bug, and the repo is untouched, so the arms differ by
    context alone with no partial fix in the way.

    Order never matters. An earlier version demanded the failure come after the
    edit, which punished the agent for reproducing the bug before changing
    anything.
    """
    if len(events) < min_calls:
        return False
    if not any(e.get("kind") == TEST and e.get("passed") is False for e in events):
        return False
    edited = any(e.get("kind") == EDIT and not e.get("failed") for e in events)
    return edited or len(events) >= no_edit_cap


def _timed(fn, *a, **kw):
    """Run something and report how long it took, in milliseconds.

    Model wait and tool execution are logged apart because the total is not
    interpretable on its own. Each step starts a fresh CLI process and may sit
    behind a rate limit, so wall clock measures our plumbing as much as the
    agent. The tool side is clean and is the part that says what the work cost.
    """
    start = time.perf_counter()
    res = fn(*a, **kw)
    return res, round((time.perf_counter() - start) * 1000, 1)


def _measure(workspace, fn, *a, **kw):
    """Run one tool call and report time plus its kernel cost.

    A workspace that exposes `counters` is read either side of the call, so
    the CPU time, disk bytes and page faults land on the call that caused
    them. A workspace without counters just gets the timing, which is how the
    offline checks keep working.
    """
    counters = getattr(workspace, "counters", None)
    before = counters.read() if counters is not None else {}
    res, tool_ms = _timed(fn, *a, **kw)
    metrics = {"tool_ms": tool_ms}
    if counters is not None:
        from ktrace import Counters

        metrics.update(Counters.delta(before, counters.read()))
    return res, metrics


def _parse(text: str) -> dict | None:
    """Pull the tool call out of a reply. Tolerates a code fence around it."""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        call = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None
    return call if isinstance(call, dict) and "tool" in call else None


def run(
    model,
    workspace: Workspace,
    trace: Trace,
    *,
    history: list[dict],
    max_steps: int = 60,
    stop_when=None,
) -> dict:
    """Drive the loop until it finishes, hits `stop_when`, or runs out of steps.

    `history` is the starting conversation. Phase 1 passes the task text. A
    resume passes the context built by resume.py, which is the only difference
    between the two phases.
    """
    events: list[dict] = []
    bad_replies = 0
    pushbacks = 0
    reason = "max_steps"

    for _ in range(max_steps):
        reply, model_ms = _timed(model, [{"role": "system", "content": SYSTEM}, *history])
        usage = dict(prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                     model_ms=model_ms)
        history.append({"role": "assistant", "content": reply.text})
        call = _parse(reply.text)

        if call is None:
            bad_replies += 1
            if bad_replies >= MAX_BAD_REPLIES:
                reason = "unparseable_replies"
                break
            # Escalate rather than repeat. A bare restatement was not enough:
            # a whole phase-1 run was lost to three prose replies in a row.
            nudge = ("Reply with one JSON object and nothing else. No prose, no "
                     "explanation, no code fences.")
            if bad_replies > 1:
                nudge += (' Example: {"tool": "run_tests", "args": {}}')
            history.append({"role": "user", "content": nudge})
            continue
        bad_replies = 0

        tool, args = call["tool"], call.get("args") or {}
        if tool == "finish":
            # A resume arm with no stop condition still must not finish without
            # doing anything, or the row is a non-observation dressed as data.
            if stop_when is None and len(events) < MIN_RESUME_CALLS and pushbacks < MAX_PUSHBACKS:
                pushbacks += 1
                history.append({"role": "user", "content":
                                "You have not looked at anything yet. Investigate first. "
                                'Reply with one JSON object, for example '
                                '{"tool": "run_tests", "args": {}}'})
                continue
            # Phase 1 needs a checkpoint with real work behind it. Left alone
            # the agent calls finish after a handful of steps: three of four
            # tasks in the first campaign produced no checkpoint at all. So an
            # early finish is pushed back on, a few times, before being taken.
            if stop_when is not None and not stop_when(events) and pushbacks < MAX_PUSHBACKS:
                pushbacks += 1
                # The reminder about the format has to ride along. Phrased as
                # plain conversation, this nudge drew a prose answer, which
                # then counted as an unparseable reply and killed the run.
                history.append({"role": "user", "content":
                                "Not yet. Run run_tests to confirm the current state, "
                                "then keep working until you are sure the fix is right. "
                                "Reply with one JSON object and nothing else, for "
                                'example {"tool": "run_tests", "args": {}}'})
                continue
            reason = "finished"
            break

        try:
            if tool == "read_file":
                res, m = _measure(workspace, workspace.read_file, args.get("path", ""),
                                  args.get("start"), args.get("end"))
                ev = trace.read(res.fields["path"], res.output, lines=res.fields["lines"],
                                **m, **usage)
            elif tool == "search":
                res, m = _measure(workspace, workspace.search,
                                  args.get("pattern", ""), args.get("scope", "."))
                ev = trace.search(res.fields["pattern"], res.output, scope=res.fields["scope"],
                                  **m, **usage)
            elif tool == "edit_file":
                res, m = _measure(workspace, workspace.edit_file, args.get("path", ""),
                                  args.get("old", ""), args.get("new", ""))
                if res.fields.get("failed"):
                    ev = trace.record(EDIT, "edit_file", args, res.output, failed=True,
                                      path=res.fields["path"], **m, **usage)
                else:
                    ev = trace.edit(res.fields["path"], res.output, post_text=res.fields["post_text"],
                                    diff=res.fields["diff"], **m, **usage)
            elif tool == "run_tests":
                res, m = _measure(workspace, workspace.run_tests, args.get("target", ""))
                ev = trace.test(res.fields["test"], res.output, passed=res.fields["passed"],
                                **m, **usage)
            else:
                res = ToolResult("other", f"unknown tool: {tool}")
                ev = trace.record("other", str(tool), args, res.output, tool_ms=0.0, **usage)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            res = ToolResult("other", f"tool error: {exc}")
            ev = trace.record("other", str(tool), args, res.output, tool_ms=0.0, **usage)

        events.append(ev)
        history.append({"role": "user", "content": f"{tool} result:\n{res.output}"})

        if stop_when and stop_when(events):
            reason = "checkpoint"
            break

    return {
        "reason": reason,
        "steps": len(events),
        "events": events,
        "checkpoint": reason == "checkpoint",
        # C0 continues from here with the context untouched, so it needs the
        # conversation back rather than a rebuilt one.
        "history": history,
    }


def phase1(model, workspace: Workspace, trace: Trace, task_text: str, **kw) -> dict:
    """Run until the checkpoint: the first failing test after an edit."""
    return run(model, workspace, trace, history=[{"role": "user", "content": task_text}],
               stop_when=checkpoint_reached, **kw)


def resume(model, workspace: Workspace, trace: Trace, resume_text: str, **kw) -> dict:
    """Carry on from a context built by resume.py. No stop predicate: run to the end."""
    return run(model, workspace, trace, history=[{"role": "user", "content": resume_text}], **kw)


class ScriptedModel:
    """Replays fixed replies. Used by the self-check so no key or network is needed."""

    def __init__(self, replies: list[str], *, prompt_tokens: int = 500, completion_tokens: int = 40):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self._usage = (prompt_tokens, completion_tokens)

    def __call__(self, messages: list[dict]) -> Reply:
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else '{"tool": "finish", "args": {"summary": "done"}}'
        return Reply(text=text, prompt_tokens=self._usage[0], completion_tokens=self._usage[1])


class AnthropicModel:
    """Live adapter. Needs `pip install anthropic` and ANTHROPIC_API_KEY.

    Untested here because this machine has neither. Verify it with one cheap
    call before starting the pilot: see the note printed by the self-check.
    """

    def __init__(self, model: str, *, max_tokens: int = 1024, temperature: float = 0.0):
        try:
            import anthropic
        except ModuleNotFoundError as exc:
            raise RuntimeError("pip install anthropic, or pass a different model object") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def __call__(self, messages: list[dict]) -> Reply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        res = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=system or None, messages=turns,
        )
        text = "".join(b.text for b in res.content if getattr(b, "type", "") == "text")
        return Reply(text=text, prompt_tokens=res.usage.input_tokens, completion_tokens=res.usage.output_tokens)

    def count_tokens(self, text: str) -> int:
        """Real tokenizer for budget matching. Never estimate from characters."""
        res = self._client.messages.count_tokens(
            model=self.model, messages=[{"role": "user", "content": text}]
        )
        return res.input_tokens


BUGGY = '''def total(items):
    """Sum the prices. Discount applies once, not per item."""
    out = 0
    for price in items:
        out += price * 0.9
    return out
'''

TESTS = '''import unittest

from calc import total


class TestTotal(unittest.TestCase):
    def test_single_discount(self):
        self.assertEqual(total([100, 100]), 180)
'''


def _selfcheck() -> None:
    """Runs the whole pipeline offline: phase 1 to the checkpoint, then a resume."""
    import tempfile

    from facts import extract_facts, relearn_events
    from resume import build_resume
    from tracelog import load, summarise

    def count(text: str) -> int:
        return len(text.split())  # stand-in; live runs pass AnthropicModel.count_tokens

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        (repo / "calc.py").write_text(BUGGY, encoding="utf-8")
        (repo / "test_calc.py").write_text(TESTS, encoding="utf-8")
        for n in range(3):
            (repo / f"mod{n}.py").write_text(f"def helper{n}(x):\n    return x\n", encoding="utf-8")

        ws = Workspace(repo, test_cmd=[sys.executable, "-m", "unittest", "-q"])
        run_meta = dict(model="scripted", tokenizer="words", context_limit=32768,
                        task="calc-discount", arm="C0", commit="abc123")

        # the path guard holds before anything else runs
        try:
            ws.read_file("../outside.txt")
        except ValueError:
            pass
        else:
            raise AssertionError("the agent escaped the repo")

        # The failing test lands at step 4 but the checkpoint waits for 8 calls,
        # so there is enough history behind it to delete a fact from.
        script = [
            '{"tool": "search", "args": {"pattern": "def total", "scope": "."}}',
            '{"tool": "read_file", "args": {"path": "calc.py", "start": 1, "end": 6}}',
            '```json\n{"tool": "edit_file", "args": {"path": "calc.py", '
            '"old": "out += price * 0.9", "new": "out += price * 0.95"}}\n```',
            '{"tool": "run_tests", "args": {"target": "test_calc"}}',
            '{"tool": "read_file", "args": {"path": "mod0.py", "start": 1, "end": 5}}',
            '{"tool": "read_file", "args": {"path": "mod1.py", "start": 1, "end": 5}}',
            '{"tool": "search", "args": {"pattern": "def helper", "scope": "."}}',
            '{"tool": "read_file", "args": {"path": "mod2.py", "start": 1, "end": 5}}',
            '{"tool": "finish", "args": {"summary": "should not get here"}}',
        ]
        trace_path = Path(tmp) / "phase1.jsonl"
        with Trace(trace_path, repo, run=run_meta) as tr:
            out = phase1(ScriptedModel(script), ws, tr, "Fix the discount bug in calc.py.")

        assert out["checkpoint"], out
        assert out["reason"] == "checkpoint", out["reason"]
        assert out["steps"] == 8, out["steps"]  # stopped at the floor, never reached finish

        _, events = load(trace_path)
        kinds = [e["kind"] for e in events]
        assert kinds[:4] == [SEARCH, READ, EDIT, TEST], kinds
        assert events[2]["diff"], "the edit logged no diff"
        assert events[3]["passed"] is False, "the test should have failed"
        assert events[0]["out"].startswith("calc.py:1:def total"), events[0]["out"][:60]

        # a four-step trace must not qualify, however it ends
        assert not checkpoint_reached(events[:4]), "the 8-call floor is not being enforced"

        facts = extract_facts(events)
        kinds_found = {f["kind"] for f in facts}
        assert {"F_loc", "F_dep", "F_neg", "F_test"} <= kinds_found, sorted(kinds_found)

        # a second scripted run that re-reads and re-runs, so the rules have something to catch
        target = next(f for f in facts if f["kind"] == "F_test")
        built = build_resume(events, target, budget=600, count_tokens=count,
                             task_text="Fix the discount bug in calc.py.")
        assert "C1" in built and "C2" in built, built["check"]

        resume_script = [
            '{"tool": "read_file", "args": {"path": "calc.py", "start": 1, "end": 6}}',
            '{"tool": "read_file", "args": {"path": "calc.py", "start": 1, "end": 6}}',
            '{"tool": "run_tests", "args": {"target": "test_calc"}}',
            '{"tool": "edit_file", "args": {"path": "calc.py", '
            '"old": "out += price * 0.95", "new": "out += price"}}',
            '{"tool": "edit_file", "args": {"path": "calc.py", '
            '"old": "    return out", "new": "    return out * 0.9"}}',
            '{"tool": "run_tests", "args": {"target": "test_calc"}}',
            '{"tool": "finish", "args": {"summary": "discount applied once"}}',
        ]
        r_path = Path(tmp) / "c2.jsonl"
        with Trace(r_path, repo, run={**run_meta, "arm": "C2"}) as tr:
            r_out = resume(ScriptedModel(resume_script), ws, tr, built["C2"].text)

        assert r_out["reason"] == "finished", r_out["reason"]
        _, r_events = load(r_path)
        got = summarise(r_events)
        assert got["redundant"].get("redundant_read") == 1, got["redundant"]
        assert 0 < got["prompt_tokens"] < got["total_tokens"], got

        # the agent went back for the deleted fact, which is the signal the study needs
        assert relearn_events(target, r_events, 0), "the re-learn signal never fired"

        # the fix actually landed, so a resume can reach a passing state
        final = ws.run_tests("test_calc")
        assert final.fields["passed"], final.output[-300:]

        # A few prose replies must not end a run. A whole phase-1 run was lost
        # when the tolerance was three and the agent answered in prose.
        recover = Path(tmp) / "recover.jsonl"
        with Trace(recover, repo, run=run_meta) as tr:
            ok = run(ScriptedModel(["let me think about this", "here is my plan",
                                    "first I will look at the file",
                                    '{"tool": "read_file", "args": {"path": "calc.py"}}',
                                    '{"tool": "finish", "args": {"summary": "done"}}']),
                     ws, tr, history=[{"role": "user", "content": "go"}])
        assert ok["reason"] == "finished", f"prose replies ended the run: {ok['reason']}"
        assert ok["steps"] == 1, ok["steps"]

        # but a model that never complies is reported, not looped on for ever
        junk = Path(tmp) / "junk.jsonl"
        with Trace(junk, repo, run=run_meta) as tr:
            bad = run(ScriptedModel(["nope"] * (MAX_BAD_REPLIES + 2)), ws, tr,
                      history=[{"role": "user", "content": "go"}])
        assert bad["reason"] == "unparseable_replies", bad["reason"]

    print(
        f"self-check ok: phase 1 stopped at the checkpoint in {out['steps']} steps, "
        f"{len(facts)} facts, resume re-learned the deleted fact and the tests pass"
    )
    print("to go live: pip install anthropic, set ANTHROPIC_API_KEY, "
          "then pass AnthropicModel(<model-id>) and its count_tokens to build_resume")


if __name__ == "__main__":
    _selfcheck()
