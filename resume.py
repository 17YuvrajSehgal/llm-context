"""Builds the resume contexts for the fact-removal study.

Three contexts come out of here, all from the same checkpoint and all filled to
the same token budget:

  C1 Keep     the target fact is in the context
  C2 Delete   the target fact is removed and scrubbed everywhere
  C3 Placebo  a fact the agent never needed again is removed instead

C0 has no resume context. The agent simply keeps going without interruption.

Two rules this module exists to enforce. The arms must match on token budget,
or a cost difference could just be a shorter prompt. And the deleted fact must
be gone from C2, because a fact that survives makes C2 behave like C1 and
pushes the result toward no effect.

Token counts come from a real tokenizer passed in as `count_tokens`. Nothing
here estimates from character counts.

Run `python resume.py` for the self-check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from facts import edge_bounded, key_leaks, relearned, usable_ids

# Each block is one logged tool call. Deleting is line-level inside a block,
# so we never need to drop a whole call unless fallback B in the plan kicks in.
BLOCK = "[{i}] {tool}({args})\n{out}"


@dataclass(frozen=True)
class Block:
    i: int
    text: str
    tokens: int


@dataclass(frozen=True)
class Resume:
    arm: str
    text: str
    tokens: int
    blocks: tuple[int, ...]
    removed_lines: int
    removed_tokens: int
    visible: tuple[str, ...]  # identifiers of the target fact still in the text

    @property
    def clean(self) -> bool:
        return not self.visible


def tiktoken_counter(encoding: str = "cl100k_base"):
    """A real tokenizer for budget matching.

    This is not Claude's own tokenizer. What validity needs is that every arm is
    measured with the same function, so the arms are matched under one
    consistent measure. The runs then report the provider's real counts from
    usage as well, so both numbers are on the table and the paper can say so.

    `disallowed_special` is cleared because repo files can contain strings that
    tiktoken otherwise refuses to encode.
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


def render_blocks(events: list[dict], count_tokens) -> list[Block]:
    """One block per logged tool call, in the order the agent saw them."""
    out = []
    for e in events:
        args = ", ".join(f"{k}={v!r}" for k, v in (e.get("args") or {}).items())
        text = BLOCK.format(i=e["i"], tool=e.get("tool", "?"), args=args, out=e.get("out") or "")
        out.append(Block(i=e["i"], text=text, tokens=count_tokens(text)))
    return out


def scrub(blocks: list[Block], fact: dict, count_tokens) -> tuple[list[Block], int, int]:
    """Drop every line that carries an identifier of `fact`.

    Lines go away rather than being replaced with a marker. A marker would tell
    the agent something was taken, which is a cue we do not want to hand it.
    """
    pats = [edge_bounded(s) for s in usable_ids(fact["ids"])]
    out, removed_lines, removed_tokens = [], 0, 0
    for b in blocks:
        lines = b.text.splitlines()
        kept = [ln for ln in lines if not any(p.search(ln) for p in pats)]
        dropped = len(lines) - len(kept)
        removed_lines += dropped
        if not dropped:
            # Unchanged, so keep the block as it is. The guard is per block
            # on purpose: it used to test the running total, so once any
            # earlier block lost a line every later block took the slow
            # path and was rebuilt needlessly.
            out.append(b)
            continue
        text = "\n".join(kept)
        tokens = count_tokens(text)
        removed_tokens += b.tokens - tokens
        out.append(replace(b, text=text, tokens=tokens))
    return out, removed_lines, removed_tokens


def _fill(blocks: list[Block], budget: int, pinned: set[int]) -> tuple[list[Block], int]:
    """Pinned blocks first, then the newest that fit.

    One pass newest-first also picks up small old blocks after a big one is
    skipped, so the context lands close to the budget instead of leaving a hole.
    """
    chosen = [b for b in blocks if b.i in pinned]
    used = sum(b.tokens for b in chosen)
    if used > budget:
        raise ValueError(
            f"the pinned blocks need {used} tokens but the budget is {budget}; raise the budget"
        )
    for b in sorted((b for b in blocks if b.i not in pinned), key=lambda b: -b.i):
        if b.tokens and used + b.tokens <= budget:
            chosen.append(b)
            used += b.tokens
    return sorted(chosen, key=lambda b: b.i), used


def _assemble(task_text: str, blocks: list[Block]) -> str:
    body = "\n\n".join(b.text for b in blocks if b.text.strip())
    return f"{task_text}\n\nWork so far:\n\n{body}\n"


def task_reserve(task_text: str, count_tokens) -> int:
    """Tokens the task text costs before any history is added.

    The runner needs this to size a budget. A budget below the reserve leaves
    no room for history at all, which is how a first version rejected every
    fact with "the task text alone needs 935 tokens of the 721 budget".
    """
    return count_tokens(_assemble(task_text, []))


def pick_placebo(
    fact: dict,
    all_facts: list[dict],
    c0_events: list[dict],
    checkpoint: int,
    blocks: list[Block],
    count_tokens,
) -> dict | None:
    """A fact the agent never went back for, costing about the same to remove.

    "Never needed again" is read off the uninterrupted C0 continuation, so the
    choice does not depend on the arm being measured.
    """
    _, _, target_cost = scrub(blocks, fact, count_tokens)
    scored = []
    for cand in all_facts:
        if cand["id"] == fact["id"]:
            continue
        if relearned(cand, c0_events, checkpoint):
            continue
        _, _, cost = scrub(blocks, cand, count_tokens)
        if cost == 0:
            continue
        # same kind is nicer but size is what the placebo has to match. The id
        # is the last key so ties resolve the same way on every run.
        scored.append(((cand["kind"] != fact["kind"], abs(cost - target_cost), cand["id"]), cand))
    return min(scored, key=lambda pair: pair[0])[1] if scored else None


def build_resume(
    events: list[dict],
    fact: dict,
    *,
    budget: int,
    count_tokens,
    task_text: str,
    all_facts: list[dict] | None = None,
    c0_events: list[dict] | None = None,
    # The largest share of the context the removed fact may be. Set from real
    # traces, which came in at 6% to 22%: a 5% limit rejected almost every
    # fact. This only excludes the absurd cases. The actual share travels with
    # the pair so the analysis can test whether it explains anything, rather
    # than the gate pretending to settle it.
    tolerance: float = 0.25,
) -> dict:
    """Build C1, C2 and C3 from one checkpoint.

    Returns the arms plus a `check` block. Read `check["ok"]` before using the
    sample: a failed check means this fact must be dropped and the drop logged.
    """
    if not events:
        raise ValueError("no phase-1 events, so there is nothing to resume from")
    blocks = render_blocks(events, count_tokens)
    checkpoint = max(e["i"] for e in events)
    reserve = count_tokens(_assemble(task_text, []))
    block_budget = budget - reserve
    if block_budget <= 0:
        raise ValueError(f"the task text alone needs {reserve} tokens of the {budget} budget")

    # One block set for every arm, chosen from the unscrubbed blocks, then
    # scrubbed in place. See the module docstring: matching token counts
    # exactly meant an arm gained history the reference arm never had.
    source = {fact["source_i"]}
    chosen, _ = _fill(blocks, block_budget, source)
    target = count_tokens(_assemble(task_text, chosen))

    def build(name: str, to_remove: dict) -> Resume:
        arm_blocks, lines, toks = scrub(chosen, to_remove, count_tokens)
        text = _assemble(task_text, arm_blocks)
        return Resume(
            arm=name,
            text=text,
            tokens=count_tokens(text),
            blocks=tuple(b.i for b in arm_blocks),
            removed_lines=lines,
            removed_tokens=toks,
            visible=tuple(key_leaks(fact, text)),
        )

    nothing = {"ids": [], "key_ids": []}  # removes nothing, same code path
    arms = {"C1": build("C1", nothing), "C2": build("C2", fact)}

    placebo = None
    if all_facts and c0_events:
        placebo = pick_placebo(fact, all_facts, c0_events, checkpoint, chosen, count_tokens)
    if placebo:
        arms["C3"] = build("C3", placebo)

    sizes = {a: r.tokens for a, r in arms.items()}
    spread = (max(sizes.values()) - min(sizes.values())) / max(sizes.values())
    problems = []
    if not arms["C2"].clean:
        problems.append(f"the fact is still visible in C2: {list(arms['C2'].visible)}")
    if not arms["C1"].visible:
        problems.append("the fact is not visible in C1, so the kept arm does not keep it")
    # The arms hold the same blocks, so any difference is exactly the lines
    # that were scrubbed. A small one is expected and gets reported; a large
    # one means the removal was a big share of the context and the arm is no
    # longer comparable.
    if spread > tolerance:
        problems.append(
            f"removing the fact changed the context by {spread:.1%}, over the "
            f"{tolerance:.0%} limit, so the arms are not comparable: {sizes}"
        )
    # A missing placebo is recorded, not fatal. C3 separates "this fact" from
    # "any deletion", but a pair without it still measures C1 against C2, and
    # throwing the pair away cost 15 of 40 tasks.

    return {
        **arms,
        "check": {
            "ok": not problems,
            "problems": problems,
            "tokens": sizes,
            "spread": round(spread, 4),
            "placebo_id": placebo["id"] if placebo else None,
            "checkpoint": checkpoint,
        },
    }


def _selfcheck() -> None:
    """The arms match on budget, C2 loses the fact, C1 keeps it, C3 removes something else."""
    import tempfile
    from pathlib import Path

    from facts import extract_facts
    from tracelog import Trace, load

    # A stand-in tokenizer. The real runs pass the provider's counter.
    def count(text: str) -> int:
        return len(text.split())

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pay.py").write_text("def charge(amount):\n    return amount\n", encoding="utf-8")
        run = dict(model="m-1", tokenizer="words", context_limit=32768, task="t-1", arm="C0", commit="abc")
        kw = dict(prompt_tokens=100, completion_tokens=10)

        with Trace(root / "t.jsonl", root, run=run) as tr:
            # filler history of varied size, so the matcher has something to trim
            for n in range(12):
                hits = "\n".join(f"mod{n}.py:{j}:def helper{n}_{j}(x): pass" for j in range(1 + n % 6))
                tr.search(f"helper{n}", hits, scope="src", **kw)
            # the work that matters
            tr.search("charge", "pay.py:1:def charge(amount):\n", scope="src", **kw)
            tr.read("pay.py", "def charge(amount):\n    return amount\n", lines=(1, 2), **kw)
            tr.edit("pay.py", "ok", post_text="def charge(a):\n    return a * 2\n",
                    diff="--- a\n+++ b\n+    return a * 2\n", **kw)
            tr.test("tests/test_pay.py::test_charge",
                    "tests/test_pay.py:12: in test_charge\nE  assert 4 == 2\n", passed=False, **kw)
            # the continuation: the agent comes back for the test, never for the helpers
            tr.test("tests/test_pay.py::test_charge", "E  assert 4 == 2\n", passed=False, **kw)

        _, events = load(root / "t.jsonl")
        checkpoint = 16
        phase1 = [e for e in events if e["i"] <= checkpoint]
        all_facts = extract_facts(phase1)
        target = next(f for f in all_facts if f["kind"] == "F_test")

        # scrubbing must take the fact and leave unrelated work alone
        blocks = render_blocks(phase1, count)
        scrubbed, lines, toks = scrub(blocks, target, count)
        joined = "\n".join(b.text for b in scrubbed)
        assert lines > 0 and toks > 0, (lines, toks)
        assert "test_charge" not in joined, "the fact survived scrubbing"
        assert "helper3" in joined, "scrubbing took unrelated work with it"

        out = build_resume(
            phase1, target, budget=400, count_tokens=count,
            task_text="Fix the double charge bug.", all_facts=all_facts, c0_events=events,
            # The toy context is ~200 words, so three removed lines are 5% of
            # it by arithmetic. Real traces are ten times bigger and the same
            # removal is well under the default limit.
            tolerance=0.15,
        )

        assert out["check"]["ok"], out["check"]["problems"]
        assert out["C2"].clean, out["C2"].visible
        assert out["C1"].visible, "C1 must still show the fact"
        assert out["C2"].removed_lines > 0, "C2 removed nothing"
        assert "C3" in out, out["check"]
        # the spread is now exactly the share of the context the fact was,
        # so it is reported rather than driven to zero
        assert out["check"]["spread"] <= 0.15, out["check"]
        assert out["C2"].removed_tokens > 0, "C2 removed no tokens"
        assert "test_charge" not in out["C2"].text
        # the budget is the cap for every arm
        assert all(r.tokens <= 400 for r in (out["C1"], out["C2"], out["C3"])), out["check"]["tokens"]
        # every arm must hold the same blocks; only the scrubbed lines differ
        assert out["C1"].blocks == out["C2"].blocks == out["C3"].blocks, (
            out["C1"].blocks, out["C2"].blocks, out["C3"].blocks)
        assert out["C1"].removed_lines == 0, out["C1"].removed_lines
        assert out["C2"].removed_lines > 0, "C2 removed nothing"

        # the placebo must be a different fact the agent never went back for
        placebo = next(f for f in all_facts if f["id"] == out["check"]["placebo_id"])
        assert placebo["id"] != target["id"]
        assert not relearned(placebo, events, checkpoint), "the placebo fact was revisited in C0"

        # a budget too small to hold the pinned block is an error, not a silent trim
        try:
            build_resume(phase1, target, budget=12, count_tokens=count, task_text="Fix it.")
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a budget smaller than the pinned block")

        # a fact whose identifiers appear nowhere fails the check instead of being used
        ghost = dict(target, ids=["nowhere-in-this-text"])
        bad = build_resume(phase1, ghost, budget=400, count_tokens=count, task_text="Fix it.")
        assert not bad["check"]["ok"], "a fact that cannot be located should fail the check"

    print(
        f"self-check ok: same blocks in every arm, the fact was "
        f"{out['check']['spread']:.1%} of the context, "
        f"at {out['check']['tokens']}, C2 dropped {out['C2'].removed_lines} lines, placebo picked"
    )


if __name__ == "__main__":
    _selfcheck()
