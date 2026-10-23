"""Fact extraction, re-learn signals and the leak check for the fact-removal study.

A fact is something the agent clearly learned, tied to the tool call that taught
it. The re-learn signal is a fixed rule that fires when the agent goes and gets
the same fact again after the restart. The leak check makes sure a deleted fact
is really gone from the resume context.

Four kinds, matching section 3 of RESEARCH-IDEA.md:
  F_loc   where something lives, from a search that found it
  F_test  a test result, from a failing test run
  F_dep   an interface, from a read that showed the signature
  F_neg   what did not work, from an edit whose test then failed

Run `python facts.py` for the self-check.
"""

from __future__ import annotations

import re

from tracelog import EDIT, READ, SEARCH, TEST, overlap

# Our harness emits grep-style hits, "path:line:text". Keep the tool that way.
_HIT = re.compile(r"^(?P<path>[\w./\\-]+\.\w+):(?P<line>\d+):", re.M)
_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+(?P<name>\w+)[^\n]*", re.M)
# Failure locations come in two shapes. pytest prints `path:line:`, while
# unittest and Django's runner print a traceback frame instead.
_PYLOC = re.compile(r"(?P<path>[\w./\\-]+\.py):(?P<line>\d+)")
_TBLOC = re.compile(r'File "(?P<path>[^"\n]+\.py)", line (?P<line>\d+)')
# Likewise for the message: pytest prefixes `E `, unittest prints the bare
# exception. The asserting source line is kept too, since it names the symbols
# under test and so makes scrubbing much stronger.
_ASSERT = re.compile(r"^E\s+(?P<msg>.+?)\s*$", re.M)
_EXC = re.compile(r"^(?P<msg>\w*(?:Error|Exception|Failure)\b.*?)\s*$", re.M)
_ASSERT_CALL = re.compile(r"^\s*(?P<msg>(?:self\.)?assert\w*\(.+?\))\s*$", re.M)
# Container paths should match the repo-relative mentions in a resume context.
_CONTAINER_PREFIX = re.compile(r"^/(?:testbed|workspace|repo|app)/")


def failure_ids(out: str) -> list[str]:
    """Identifiers of a test failure, across pytest and unittest output.

    More identifiers means stronger scrubbing, so this is deliberately greedy.
    `usable_ids` drops anything too short or purely numeric later.
    """
    ids = []
    for rx in (_PYLOC, _TBLOC):
        for m in rx.finditer(out):
            path = _CONTAINER_PREFIX.sub("", m["path"].replace("\\", "/"))
            ids.append(f"{path}:{m['line']}")
            ids.append(path)
    for rx in (_ASSERT, _EXC, _ASSERT_CALL):
        if m := rx.search(out):
            ids.append(m["msg"])
    return ids

# Caps stop one noisy call from flooding the sample.
# ponytail: raise them if the pilot comes up short of facts.
MAX_LOC_PER_SEARCH = 3
MAX_DEP_PER_READ = 3


def _fact(kind: str, fid: str, source: dict, fields: dict, ids: list[str],
          weak: list[str] | None = None) -> dict:
    """`ids` are scrubbed from a resume context. `weak` are scrubbed too, but
    do not disqualify the fact if they survive.

    The split exists because issue text names files. The Django issue lists
    `django/db/backends/mysql/operations.py`, the task text is never scrubbed,
    so every fact about that file counted as leaked and all of them were
    dropped. A bare path is not the fact though: the fact is which symbol sits
    at which line, or which test failed with which assertion. Those must be
    gone. The path merely being mentioned does not give them away.
    """
    weak_set = {s.strip() for s in (weak or []) if s and s.strip()}
    strong = {s.strip() for s in ids if s and s.strip()} - weak_set
    return {
        "id": fid,
        "kind": kind,
        "source_i": source["i"],
        "fields": fields,
        "ids": sorted(strong | weak_set),
        "key_ids": sorted(strong),
    }


def _failing_test_after(events: list[dict], n: int) -> str | None:
    """The test that failed after event n, before the agent edited anything else."""
    for e in events[n + 1 :]:
        if e.get("kind") == EDIT:
            return None
        if e.get("kind") == TEST and e.get("passed") is False:
            return e.get("test")
    return None


def _added_lines(diff: str | None) -> list[str]:
    if not diff:
        return []
    out = []
    for ln in diff.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            body = ln[1:].strip()
            if len(body) >= 4:
                out.append(body)
    return out[:3]


def extract_facts(events: list[dict]) -> list[dict]:
    """Pull every fact the trace shows the agent learned, earliest source wins."""
    found: dict[str, dict] = {}

    def keep(f: dict) -> None:
        if f["id"] not in found:
            found[f["id"]] = f

    for n, e in enumerate(events):
        kind, out = e.get("kind"), (e.get("out") or "")

        if kind == TEST and e.get("passed") is False:
            target = e.get("test") or ""
            ids = [target, *failure_ids(out)]
            keep(_fact("F_test", f"F_test:{target}", e, {"test": target}, ids,
                       weak=[i for i in ids if i.endswith(".py")]))

        elif kind == EDIT:
            failed = _failing_test_after(events, n)
            if failed:
                path, post = e.get("path") or "", e.get("post_hash") or ""
                ids = [path, post, failed, *_added_lines(e.get("diff"))]
                keep(
                    _fact(
                        "F_neg",
                        f"F_neg:{path}:{post}",
                        e,
                        {"path": path, "post_hash": post, "test": failed},
                        ids,
                        weak=[path],
                    )
                )

        elif kind == SEARCH:
            symbol = e.get("pattern") or ""
            seen_paths: set[str] = set()
            for m in _HIT.finditer(out):
                path, line = m["path"], int(m["line"])
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                keep(
                    _fact(
                        "F_loc",
                        f"F_loc:{symbol}@{path}:{line}",
                        e,
                        {"symbol": symbol, "path": path, "line": line},
                        [symbol, path, f"{path}:{line}"],
                        weak=[path],
                    )
                )
                if len(seen_paths) >= MAX_LOC_PER_SEARCH:
                    break

        elif kind == READ:
            path = e.get("path") or ""
            start = (e.get("lines") or [1])[0]
            for i, m in enumerate(_DEF.finditer(out)):
                if i >= MAX_DEP_PER_READ:
                    break
                line = start + out.count("\n", 0, m.start())
                name, sig = m["name"], m.group(0).strip()
                keep(
                    _fact(
                        "F_dep",
                        f"F_dep:{name}@{path}:{line}",
                        e,
                        {"symbol": name, "path": path, "line": line, "sig": sig},
                        [name, path, f"{path}:{line}", sig],
                        weak=[path],
                    )
                )

    return list(found.values())


def relearn_events(fact: dict, events: list[dict], after_i: int) -> list[int]:
    """Event indices after `after_i` where the agent gets this fact again.

    One fixed rule per kind. A failing test is re-learned by running it again.
    A failed fix is re-learned by applying the same edit again. A location is
    re-learned by reading those lines again or searching for the symbol again.
    """
    kind, f = fact["kind"], fact["fields"]
    hits = []
    for e in events:
        if e["i"] <= after_i:
            continue
        k = e.get("kind")
        if kind == "F_test":
            if k == TEST and e.get("test") == f["test"]:
                hits.append(e["i"])
        elif kind == "F_neg":
            if k == EDIT and e.get("path") == f["path"] and e.get("post_hash") == f["post_hash"]:
                hits.append(e["i"])
        else:  # F_loc and F_dep are both locations
            line = f["line"]
            if k == READ and e.get("path") == f["path"] and overlap(e.get("lines"), [line, line]):
                hits.append(e["i"])
            elif k == SEARCH and f["symbol"] and f["symbol"] in (e.get("pattern") or ""):
                hits.append(e["i"])
    return hits


def relearned(fact: dict, events: list[dict], after_i: int) -> bool:
    return bool(relearn_events(fact, events, after_i))


def relearn_cost(fact: dict, events: list[dict], after_i: int) -> dict | None:
    """R(f): calls and tokens from the restart until the agent gets the fact back.

    None means it never got it back, which is its own result and must be
    reported separately rather than counted as zero cost.
    """
    hits = relearn_events(fact, events, after_i)
    if not hits:
        return None
    first = hits[0]
    window = [e for e in events if after_i < e["i"] <= first and "tokens" in e]
    return {
        "at_i": first,
        "calls": len(window),
        "tokens": sum(e["tokens"]["prompt"] + e["tokens"]["completion"] for e in window),
    }


def usable_ids(ids: list[str]) -> list[str]:
    """Identifiers worth searching for.

    Bare numbers are dropped because a line number matches almost any text. The
    composite "path:line" carries that information instead. The 3-character
    floor is deliberately low: over-detecting a leak costs sample size, while
    missing one leaves the fact visible in the deleted arm and biases the
    result toward no effect.
    """
    return sorted({s for s in ids if len(s.strip()) >= 3 and not s.strip().isdigit()})


def edge_bounded(s: str) -> re.Pattern[str]:
    """Match `s` as a whole token, so a symbol does not match inside a longer name.

    Plain substring search is too eager. The symbol `charge` would match inside
    `test_charge`, and `pay.py` inside `test_pay.py`, which would disqualify
    almost every fact. Boundaries are only added at edges that are word
    characters, so an identifier ending in punctuation still matches.
    """
    word = r"[^\W]"  # letters, digits, underscore
    left = r"(?<!" + word + r")" if re.match(r"\w", s[0]) else ""
    right = r"(?!" + word + r")" if re.match(r"\w", s[-1]) else ""
    return re.compile(left + re.escape(s) + right, re.I)


def leaks(fact: dict, context_text: str) -> list[str]:
    """Identifiers of this fact that are still visible in the resume context."""
    return [s for s in usable_ids(fact["ids"]) if edge_bounded(s).search(context_text)]


def key_leaks(fact: dict, context_text: str) -> list[str]:
    """Only the identifiers that constitute the fact. Used for the pass/fail
    decision, while `leaks` reports everything for the record."""
    keys = set(fact.get("key_ids") or fact["ids"])
    return [s for s in leaks(fact, context_text) if s in keys]


def leak_report(facts: list[dict], context_text: str) -> dict:
    """Which facts survived scrubbing. Log this rate in the paper."""
    leaked = {f["id"]: found for f in facts if (found := leaks(f, context_text))}
    return {
        "checked": len(facts),
        "leaked": len(leaked),
        "clean": len(facts) - len(leaked),
        "rate": round(len(leaked) / len(facts), 3) if facts else 0.0,
        "detail": leaked,
    }


def _selfcheck() -> None:
    """All four kinds extract, the signals fire only when they should, leaks are caught."""
    import tempfile
    from pathlib import Path

    from tracelog import Trace, load

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pay.py").write_text("def charge(amount):\n    return amount\n", encoding="utf-8")
        run = dict(model="m-1", tokenizer="tk", context_limit=32768, task="t-1", arm="C0", commit="abc")
        kw = dict(prompt_tokens=100, completion_tokens=10)

        with Trace(root / "t.jsonl", root, run=run) as tr:
            # i=1 search finds a location
            tr.search("charge", "pay.py:1:def charge(amount):\nutil.py:9:charge(x)\n", scope="src", **kw)
            # i=2 read shows the signature
            tr.read("pay.py", "def charge(amount):\n    return amount\n", lines=(1, 2), **kw)
            # i=3 edit, then i=4 the test fails -> F_neg plus F_test
            tr.edit("pay.py", "ok", post_text="def charge(a):\n    return a * 2\n",
                    diff="--- a\n+++ b\n+    return a * 2\n", **kw)
            tr.test("tests/test_pay.py::test_charge",
                    "tests/test_pay.py:12: in test_charge\nE  assert 4 == 2\n", passed=False, **kw)
            # i=5..7 the agent repeats itself after the checkpoint
            tr.test("tests/test_pay.py::test_charge", "E  assert 4 == 2\n", passed=False, **kw)
            tr.read("pay.py", "def charge(a):\n    return a * 2\n", lines=(1, 2), **kw)
            tr.edit("pay.py", "ok", post_text="def charge(a):\n    return a * 2\n", **kw)

        _, events = load(root / "t.jsonl")
        facts = extract_facts(events)
        by_kind: dict[str, list[dict]] = {}
        for f in facts:
            by_kind.setdefault(f["kind"], []).append(f)

        assert set(by_kind) == {"F_loc", "F_dep", "F_neg", "F_test"}, sorted(by_kind)
        assert len(by_kind["F_loc"]) == 2, by_kind["F_loc"]  # pay.py and util.py
        assert by_kind["F_test"][0]["fields"]["test"] == "tests/test_pay.py::test_charge"
        assert "assert 4 == 2" in by_kind["F_test"][0]["ids"], by_kind["F_test"][0]["ids"]
        assert by_kind["F_neg"][0]["fields"]["test"] == "tests/test_pay.py::test_charge"
        assert "return a * 2" in by_kind["F_neg"][0]["ids"], by_kind["F_neg"][0]["ids"]

        # the checkpoint sits after event 4; everything before it is learned, not re-learned
        after = 4
        test_fact = by_kind["F_test"][0]
        neg_fact = by_kind["F_neg"][0]
        dep_fact = next(f for f in by_kind["F_dep"] if f["fields"]["symbol"] == "charge")

        assert relearn_events(test_fact, events, after) == [5], relearn_events(test_fact, events, after)
        assert relearn_events(dep_fact, events, after) == [6], relearn_events(dep_fact, events, after)
        assert relearn_events(neg_fact, events, after) == [7], relearn_events(neg_fact, events, after)

        cost = relearn_cost(test_fact, events, after)
        assert cost == {"at_i": 5, "calls": 1, "tokens": 110}, cost

        # nothing happens after the last event, so nothing is re-learned
        assert relearn_cost(test_fact, events, 7) is None
        assert not relearned(neg_fact, events, 7)

        # a location nobody touches again stays silent
        util_fact = next(f for f in by_kind["F_loc"] if f["fields"]["path"] == "util.py")
        assert not relearned(util_fact, events, after), "util.py was never revisited"

        # leak check
        dirty = "we already know tests/test_pay.py::test_charge is failing"
        # both the bare path and the full test id are identifiers, so both hit
        assert leaks(test_fact, dirty) == [
            "tests/test_pay.py", "tests/test_pay.py::test_charge"
        ], leaks(test_fact, dirty)
        assert leaks(test_fact, "resume and keep going") == []

        rep = leak_report([test_fact, dep_fact], dirty)
        assert (rep["checked"], rep["leaked"], rep["clean"], rep["rate"]) == (2, 1, 1, 0.5), rep
        assert test_fact["id"] in rep["detail"], rep["detail"]

        # a bare line number must not count as a leak
        assert leaks(test_fact, "step 12 of 20") == [], leaks(test_fact, "step 12 of 20")

        # a symbol must not leak through a longer name that merely contains it
        assert leaks(dep_fact, "we already know test_charge in test_pay.py fails") == [], \
            leaks(dep_fact, "we already know test_charge in test_pay.py fails")
        # but the symbol on its own does leak
        assert leaks(dep_fact, "look at charge() in pay.py") == ["charge", "pay.py"], \
            leaks(dep_fact, "look at charge() in pay.py")

    print(f"self-check ok: {len(facts)} facts across 4 kinds, signals and leak check behave")


if __name__ == "__main__":
    _selfcheck()
