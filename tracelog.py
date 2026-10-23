"""Trace log and redundancy rules for the fact-removal study.

Writes one JSONL file per agent run. Every claim in the paper has to trace back
to a logged run, so `record` refuses events that arrive without real token
counts from the API.

The four `redundant_*` functions are the paper's credibility core. They decide
what counts as repeated work using content hashes only, never a judgement call.

Run `python tracelog.py` for the self-check.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

READ, SEARCH, TEST, EDIT = "read", "search", "test", "edit"


def _h(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    # 16 hex chars is 64 bits. Collisions are not a concern at trace scale.
    return hashlib.sha256(data).hexdigest()[:16]


def _norm(text: str) -> str:
    """Normalise source text before hashing, so cosmetic churn is not a new fix.

    ponytail: whitespace only. Add comment stripping if the repeated-fix rule
    starts missing duplicates that differ by formatting.
    """
    lines = (ln.rstrip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)


class Trace:
    """Append-only JSONL writer. One instance per agent run.

    `run` must carry the facts the paper needs to reproduce the row: model id,
    tokenizer, temperature, context limit, task id, arm (C0..C3) and git commit.
    """

    def __init__(
        self,
        out_path: str | Path,
        repo_root: str | Path,
        *,
        run: dict,
        max_output: int = 200_000,
        hash_file=None,
    ):
        for key in ("model", "tokenizer", "context_limit", "task", "arm", "commit"):
            if not run.get(key):
                raise ValueError(f"run metadata is missing {key!r}; the run would be unusable")
        # Hardware goes in the header, captured rather than passed, because a
        # run collected on a different machine is not comparable and nothing in
        # the old header said which machine it came from. Core count changes
        # how a parallel test suite spends CPU, so a kernel measurement from a
        # 4-core box cannot sit in the same table as one from a 16-core box.
        run = {
            **run,
            "host": socket.gethostname(),
            "nproc": os.cpu_count(),
            "kernel": platform.release(),
        }
        self.repo_root = Path(repo_root)
        self.max_output = max_output
        # SWE-bench repos live inside a container, so the hasher is swappable.
        # It must return None for a path that does not exist.
        self._hash_file = hash_file
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._n = 0
        self._emit({"kind": "run", **run})

    def _emit(self, event: dict) -> dict:
        event["i"] = self._n
        event["t"] = datetime.now(timezone.utc).isoformat()
        self._n += 1
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()  # a killed run must still leave a readable trace
        return event

    def _file_hash(self, path: str | None) -> str | None:
        if not path:
            return None
        if self._hash_file is not None:
            return self._hash_file(path)
        f = self.repo_root / path
        return _h(f.read_bytes()) if f.is_file() else None

    def record(
        self,
        kind: str,
        tool: str,
        args: dict,
        output: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        stored_tokens: int = 0,
        **fields,
    ) -> dict:
        """Log one tool call. Token counts come from the API usage fields.

        Estimated counts are not accepted: the paper reports token costs, so a
        guessed number would silently corrupt the result.
        """
        for name, value in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a real token count from the API, got {value!r}")
        return self._emit(
            {
                "kind": kind,
                "tool": tool,
                "args": args,
                # the hash always covers the full output, so truncation is detectable
                "out_hash": _h(output),
                "out_bytes": len(output.encode("utf-8", "replace")),
                "out": output[: self.max_output],
                "out_truncated": len(output) > self.max_output,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "stored": stored_tokens,
                },
                **fields,
            }
        )

    # Thin helpers so callers cannot misspell the fields the rules read.

    def read(self, path: str, output: str, *, lines: tuple[int, int] | None = None, **kw) -> dict:
        return self.record(
            READ, kw.pop("tool", "read_file"), {"path": path, "lines": lines}, output,
            path=path, lines=list(lines) if lines else None, file_hash=self._file_hash(path), **kw,
        )

    def search(self, pattern: str, output: str, *, scope: str = ".", **kw) -> dict:
        return self.record(
            SEARCH, kw.pop("tool", "grep"), {"pattern": pattern, "scope": scope}, output,
            pattern=pattern, scope=scope, **kw,
        )

    def test(self, target: str, output: str, *, passed: bool, **kw) -> dict:
        return self.record(
            TEST, kw.pop("tool", "run_tests"), {"target": target}, output,
            test=target, passed=bool(passed), **kw,
        )

    def edit(self, path: str, output: str, *, post_text: str, diff: str | None = None, **kw) -> dict:
        """`post_text` is the full file content after the edit. `diff` is optional
        but the negative-knowledge facts read it, so pass it when the tool has one."""
        return self.record(
            EDIT, kw.pop("tool", "edit_file"), {"path": path}, output,
            path=path, post_hash=_h(_norm(post_text)), diff=diff,
            file_hash=self._file_hash(path), **kw,
        )

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load(path: str | Path) -> tuple[dict, list[dict]]:
    """Return (run header, tool-call events). Event order is list position."""
    rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows or rows[0].get("kind") != "run":
        raise ValueError(f"{path} does not start with a run header")
    events = rows[1:]
    for pos, e in enumerate(events, start=1):
        if e["i"] != pos:
            raise ValueError(f"trace is out of order at position {pos}: event index {e['i']}")
    return rows[0], events


def _edits_between(events: list[dict], lo: int, hi: int, path: str | None = None) -> list[dict]:
    """Edits that actually wrote bytes, strictly between two events.

    A rejected edit is skipped. It never touched the file, so it cannot justify
    re-reading, re-searching or re-running anything. Counting it as an edit
    would excuse genuinely redundant work and under-report the waste.
    """
    return [
        e for e in events[lo + 1 : hi]
        if e.get("kind") == EDIT and not e.get("failed")
        and (path is None or e.get("path") == path)
    ]


def overlap(a: list[int] | None, b: list[int] | None) -> bool:
    if not a or not b:
        return True  # a missing range means the whole file was read
    return a[0] <= b[1] and b[0] <= a[1]


def redundant_reads(events: list[dict]) -> list[dict]:
    """A re-read is redundant only if the bytes could not have changed.

    Same file, overlapping lines, identical content hash, and no edit by the
    agent in between. An edit-then-revert restores the hash but justifies the
    re-read, so the edit check is not covered by the hash check.
    """
    out = []
    for n, e in enumerate(events):
        if e.get("kind") != READ:
            continue
        for m in range(n - 1, -1, -1):
            p = events[m]
            if p.get("kind") != READ or p.get("path") != e.get("path"):
                continue
            if not overlap(p.get("lines"), e.get("lines")):
                continue
            if p.get("file_hash") == e.get("file_hash") and not _edits_between(events, m, n, e.get("path")):
                out.append({"rule": "redundant_read", "i": e["i"], "prior_i": p["i"], "path": e.get("path")})
            break  # only the most recent overlapping read matters
    return out


def _subsumes(prior: dict, cur: dict) -> bool:
    pp, cp = prior.get("pattern", ""), cur.get("pattern", "")
    if pp != cp and pp not in cp:
        return False  # a wider pattern covers a narrower one, not the reverse
    ps, cs = (prior.get("scope") or ".").rstrip("/"), (cur.get("scope") or ".").rstrip("/")
    return ps in ("", ".") or cs == ps or cs.startswith(ps + "/")


def redundant_searches(events: list[dict]) -> list[dict]:
    """Repeating a search whose answer cannot have moved since last time."""
    out = []
    for n, e in enumerate(events):
        if e.get("kind") != SEARCH:
            continue
        for m in range(n - 1, -1, -1):
            p = events[m]
            if p.get("kind") != SEARCH or not _subsumes(p, e):
                continue
            if not _edits_between(events, m, n):
                out.append({"rule": "redundant_search", "i": e["i"], "prior_i": p["i"], "pattern": e.get("pattern")})
            break
    return out


def redundant_tests(events: list[dict]) -> list[dict]:
    """Re-running a test with nothing edited in between cannot tell it anything new."""
    out = []
    for n, e in enumerate(events):
        if e.get("kind") != TEST:
            continue
        for m in range(n - 1, -1, -1):
            p = events[m]
            if p.get("kind") != TEST or p.get("test") != e.get("test"):
                continue
            if not _edits_between(events, m, n):
                out.append({"rule": "redundant_test", "i": e["i"], "prior_i": p["i"], "test": e.get("test")})
            break
    return out


def _failed_before_next_edit(events: list[dict], m: int) -> bool:
    for e in events[m + 1 :]:
        if e.get("kind") == EDIT:
            return False
        if e.get("kind") == TEST and e.get("passed") is False:
            return True
    return False


def repeated_failed_fixes(events: list[dict]) -> list[dict]:
    """Re-applying an edit that already failed its test once."""
    out = []
    for n, e in enumerate(events):
        if e.get("kind") != EDIT:
            continue
        for m in range(n - 1, -1, -1):
            p = events[m]
            if p.get("kind") != EDIT or p.get("path") != e.get("path") or p.get("post_hash") != e.get("post_hash"):
                continue
            if _failed_before_next_edit(events, m):
                out.append({"rule": "repeated_failed_fix", "i": e["i"], "prior_i": p["i"], "path": e.get("path")})
            break
    return out


def summarise(events: list[dict]) -> dict:
    """Counts per rule, plus the three token figures kept apart."""
    findings = (
        redundant_reads(events)
        + redundant_searches(events)
        + redundant_tests(events)
        + repeated_failed_fixes(events)
    )
    tok = [e["tokens"] for e in events if "tokens" in e]
    prompt = sum(t["prompt"] for t in tok)
    completion = sum(t["completion"] for t in tok)
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["rule"]] = counts.get(f["rule"], 0) + 1
    return {
        "calls": len(events),
        "redundant": counts,
        "redundant_total": len(findings),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "stored_tokens_peak": max((t["stored"] for t in tok), default=0),
        # kept apart on purpose: the model number is polluted by process spawn
        # and rate-limit waits, the tool number is clean
        "model_ms": round(sum(e.get("model_ms") or 0 for e in events), 1),
        "tool_ms": round(sum(e.get("tool_ms") or 0 for e in events), 1),
        "findings": findings,
    }


def _selfcheck() -> None:
    """Each rule fires once on repeated work and stays silent when the repeat is justified."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src, other = root / "a.py", root / "b.py"
        src.write_text("x = 1\n", encoding="utf-8")
        other.write_text("y = 2\n", encoding="utf-8")
        run = dict(model="m-1", tokenizer="tk", context_limit=32768, task="t-1", arm="C1", commit="abc123")

        with Trace(root / "t.jsonl", root, run=run) as tr:
            kw = dict(prompt_tokens=10, completion_tokens=2)

            # reads
            tr.read("a.py", "x = 1", lines=(1, 10), **kw)
            tr.read("a.py", "x = 1", lines=(5, 8), **kw)          # redundant, overlaps, unchanged
            tr.read("b.py", "y = 2", lines=(1, 10), **kw)         # different file, fine
            src.write_text("x = 99\n", encoding="utf-8")          # changed outside the agent
            tr.read("a.py", "x = 99", lines=(1, 10), **kw)        # justified, bytes moved

            # searches
            tr.search("def foo", "hit", scope="src", **kw)
            tr.search("def foo_bar", "hit", scope="src/pkg", **kw)  # redundant, narrower and inside
            tr.search("def zzz", "", scope="src", **kw)             # unrelated pattern, fine

            # a failing fix, then the same fix again
            tr.edit("a.py", "ok", post_text="x = 2\n", **kw)
            tr.test("tests/test_a.py::test_x", "FAIL", passed=False, **kw)
            tr.test("tests/test_a.py::test_x", "FAIL", passed=False, **kw)  # redundant, nothing edited
            tr.edit("a.py", "ok", post_text="x = 2\n", **kw)                # redundant, already failed
            tr.test("tests/test_a.py::test_x", "FAIL", passed=False, **kw)  # justified, edit in between

            # a fix that never failed, repeated: not a repeated failed fix
            tr.edit("b.py", "ok", post_text="y = 3\n", **kw)
            tr.test("tests/test_b.py::test_y", "PASS", passed=True, **kw)
            tr.edit("b.py", "ok", post_text="y = 3\n", **kw)

            # search after an edit is justified
            tr.search("def foo", "hit", scope="src", **kw)

            # edit then revert: hash matches the first read again, but re-reading is fair
            tr.edit("a.py", "ok", post_text="x = 1\n", **kw)
            src.write_text("x = 99\n", encoding="utf-8")
            tr.read("a.py", "x = 99", lines=(1, 10), **kw)

        header, events = load(root / "t.jsonl")
        assert header["task"] == "t-1" and header["arm"] == "C1"
        # a run with no record of its machine cannot be pooled with another
        assert header["nproc"] and header["host"] and header["kernel"], header
        got = summarise(events)

        assert [f["i"] for f in redundant_reads(events)] == [2], redundant_reads(events)
        assert [f["i"] for f in redundant_searches(events)] == [6], redundant_searches(events)
        assert [f["i"] for f in redundant_tests(events)] == [10], redundant_tests(events)
        assert [f["i"] for f in repeated_failed_fixes(events)] == [11], repeated_failed_fixes(events)
        assert got["redundant_total"] == 4, got["redundant"]
        assert got["calls"] == 18, got["calls"]
        assert got["prompt_tokens"] == 180 and got["completion_tokens"] == 36, got

        # token counts must be real
        with Trace(root / "t2.jsonl", root, run=run) as tr:
            for bad in (None, 12.5, "100", True, -1):
                try:
                    tr.read("a.py", "x", prompt_tokens=bad, completion_tokens=1)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"accepted a fake token count: {bad!r}")

        # run metadata must be complete
        try:
            Trace(root / "t3.jsonl", root, run={"model": "m-1"})
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a run with no task or commit")

    print("self-check ok: 4 rules fired, justified repeats stayed silent")


if __name__ == "__main__":
    _selfcheck()
