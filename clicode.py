"""Model adapter that runs through the Claude Code CLI on a subscription.

Why this shape. Claude Code is a whole agent, not a model. If we let it use its
own tools it would do the work its own way and our trace, our tool set and our
context budget would all be out of our hands. So we switch its own tools off,
replace its system prompt with ours, and use it for one decision at a time. Our
loop in harness.py still executes every tool and writes the trace.

That keeps the controlled harness the plan asks for. Letting Claude Code drive
with its own tools is a different experiment, and it is Future Plans item 4.

Each call is stateless, so the whole conversation goes into the prompt. The
system prompt is replaced rather than appended, which also cuts the preamble
Claude Code would otherwise carry.

Run `python clicode.py` for a live check. It makes two real calls.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from harness import SYSTEM, Reply

# Claude Code's own tools, switched off so it answers instead of acting.
OWN_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit",
    "WebFetch", "WebSearch", "Task", "TodoWrite", "Artifact", "Skill",
]

ROLE_LABEL = {"user": "USER", "assistant": "ASSISTANT"}


class ModelUnavailable(RuntimeError):
    """The CLI could not answer. Usually the five-hour cap, sometimes a crash.

    A separate type so the runner can record the arm as failed and carry on
    instead of losing the arms that already finished.
    """


def _rows(stdout: str) -> list[dict]:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def _limit_note(rows: list[dict]) -> str:
    """Says whether a usage cap was in play, and when it lifts.

    The subscription has a rolling five-hour cap and this account has overage
    turned off at org level, so a call past the cap fails outright. Knowing the
    reset time is the difference between waiting and giving up.
    """
    for r in rows:
        if r.get("type") != "rate_limit_event":
            continue
        info = r.get("rate_limit_info") or {}
        if info.get("status") == "allowed":
            continue
        resets = info.get("resetsAt")
        when = (
            time.strftime("%H:%M", time.localtime(resets)) if isinstance(resets, (int, float)) else "?"
        )
        return f" [{info.get('rateLimitType', 'rate')} limit {info.get('status')}, resets {when}]"
    return ""


def _system_file(text: str) -> Path:
    """The system prompt as a file, named by its content so instances share one."""
    d = Path(tempfile.gettempdir()) / "cfa-sysprompt"
    d.mkdir(exist_ok=True)
    f = d / (hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] + ".txt")
    if not f.exists():
        f.write_text(text, encoding="utf-8")
    return f


def render(messages: list[dict]) -> str:
    """Flatten the conversation into one prompt. The system turn goes to --system-prompt."""
    parts = []
    for m in messages:
        if m["role"] == "system":
            continue
        parts.append(f"{ROLE_LABEL.get(m['role'], m['role'].upper())}:\n{m['content']}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


class ClaudeCliModel:
    """Runs `claude -p` once per step. Uses whatever auth the CLI already has."""

    def __init__(
        self,
        model: str = "sonnet",
        *,
        system: str = SYSTEM,
        timeout: int = 300,
        max_budget_usd: float | None = None,
        cwd: str | None = None,
    ):
        if not shutil.which("claude"):
            raise RuntimeError("the claude CLI is not on PATH")
        self.model, self.system, self.timeout = model, system, timeout
        self.max_budget_usd, self.cwd = max_budget_usd, cwd
        self.cost_usd = 0.0
        self.calls = 0

    def _cmd(self) -> list[str]:
        """No prompt on the command line. Windows caps a command line at about
        32k characters, and a resumed conversation blows past that, which fails
        as `WinError 206: filename or extension is too long`. The conversation
        goes in on stdin and the system prompt goes in as a file."""
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "json",
            "--system-prompt-file", str(_system_file(self.system)),
            "--no-session-persistence",
            "--disallowed-tools", *OWN_TOOLS,
        ]
        if self.max_budget_usd:
            cmd += ["--max-budget-usd", str(self.max_budget_usd)]
        return cmd

    def __call__(self, messages: list[dict]) -> Reply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        self.system = system or self.system
        proc = subprocess.run(
            self._cmd(), capture_output=True,
            input=render(messages).encode("utf-8"),
            timeout=self.timeout, cwd=self.cwd,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        rows = _rows(stdout)
        result = next((r for r in rows if r.get("type") == "result"), None)
        limit = _limit_note(rows)

        if proc.returncode != 0 or result is None:
            # The useful message is in the result row, not the exit code. Quote
            # it, and say so when the five-hour cap is what stopped us.
            detail = str(result.get("result")) if result else (stderr or stdout[-600:])
            raise ModelUnavailable(f"claude exited {proc.returncode}{limit}: {detail[:600]}")
        if result.get("is_error") or result.get("subtype") != "success":
            raise ModelUnavailable(
                f"claude reported {result.get('subtype')}{limit}: {str(result.get('result'))[:600]}"
            )

        usage = result.get("usage") or {}
        self.cost_usd += result.get("total_cost_usd") or 0.0
        self.calls += 1
        # Cache reads and cache writes are still prompt tokens we paid for, so
        # they belong in the prompt count. Leaving them out would understate it.
        prompt_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        return Reply(
            text=result.get("result") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.get("output_tokens", 0),
        )


def _livecheck() -> None:
    """Two real calls. Confirms the tool protocol survives and usage comes back."""
    model = ClaudeCliModel("sonnet")

    plain = model([
        {"role": "system", "content": "Reply with exactly the word: ok"},
        {"role": "user", "content": "go"},
    ])
    assert plain.prompt_tokens > 0, plain
    assert plain.completion_tokens > 0, plain
    print(f"call 1  reply={plain.text.strip()[:40]!r}  "
          f"prompt={plain.prompt_tokens}  out={plain.completion_tokens}")

    step = model([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Fix the discount bug in calc.py. Start by finding the function."},
    ])
    print(f"call 2  reply={step.text.strip()[:90]!r}  "
          f"prompt={step.prompt_tokens}  out={step.completion_tokens}")

    from harness import _parse

    call = _parse(step.text)
    assert call and "tool" in call, f"the reply was not a tool call: {step.text[:200]}"
    print(f"parsed  tool={call['tool']}  args={call.get('args')}")
    print(f"total   {model.calls} calls, ${model.cost_usd:.4f}")
    print("livecheck ok: the CLI answers in our tool protocol and reports real usage")


if __name__ == "__main__":
    _livecheck()
