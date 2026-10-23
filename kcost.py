"""Per-tool-call cost, in tokens and in kernel resources.

This is the one analysis the broken arm construction does not touch. It asks a
different question: when an agent does a unit of work, what does that unit cost
in tokens, and what does it cost the machine? If the two disagree, then every
paper that counts wasted tokens is mis-weighting the waste.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

KINDS = ("read", "search", "edit", "test")


def load_calls(root: Path) -> list[dict]:
    """Every logged tool call across every run, with its arm and task."""
    calls = []
    for f in sorted(root.rglob("*.jsonl")):
        rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows or rows[0].get("kind") != "run":
            continue
        head = rows[0]
        for e in rows[1:]:
            if e.get("kind") not in KINDS:
                continue
            tok = e.get("tokens") or {}
            calls.append({
                "task": head["task"], "arm": head["arm"], "kind": e["kind"],
                "cpu_ms": (e.get("k_usage_usec") or 0) / 1000,
                "rd_mb": (e.get("k_rbytes") or 0) / 1e6,
                "majflt": e.get("k_pgmajfault") or 0,
                "tool_ms": e.get("tool_ms") or 0,
                "model_ms": e.get("model_ms") or 0,
                "tokens": (tok.get("prompt") or 0) + (tok.get("completion") or 0),
                "has_kernel": e.get("k_usage_usec") is not None,
            })
    return calls


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else 0.0


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "vm-results/runs-vm")
    calls = [c for c in load_calls(root) if c["has_kernel"]]
    if not calls:
        print("no calls with kernel data found")
        return
    print(f"{len(calls)} tool calls with kernel data, "
          f"{len({c['task'] for c in calls})} tasks\n")

    by = defaultdict(list)
    for c in calls:
        by[c["kind"]].append(c)

    tot_tok = sum(c["tokens"] for c in calls)
    tot_cpu = sum(c["cpu_ms"] for c in calls)
    tot_rd = sum(c["rd_mb"] for c in calls)

    print("median cost of one tool call")
    print(f"  {'tool':8} {'n':>4} {'tokens':>9} {'cpu_ms':>8} {'read_MB':>8} "
          f"{'majflt':>7} {'tool_ms':>8}")
    for k in KINDS:
        g = by.get(k) or []
        if not g:
            continue
        print(f"  {k:8} {len(g):4} {med([x['tokens'] for x in g]):9.0f} "
              f"{med([x['cpu_ms'] for x in g]):8.1f} {med([x['rd_mb'] for x in g]):8.2f} "
              f"{med([x['majflt'] for x in g]):7.0f} {med([x['tool_ms'] for x in g]):8.1f}")

    print("\nshare of the total, by tool")
    print(f"  {'tool':8} {'calls':>6} {'of tokens':>10} {'of CPU':>8} {'of disk':>8}")
    for k in KINDS:
        g = by.get(k) or []
        if not g:
            continue
        print(f"  {k:8} {len(g) / len(calls):6.0%} "
              f"{sum(x['tokens'] for x in g) / tot_tok:10.0%} "
              f"{sum(x['cpu_ms'] for x in g) / tot_cpu:8.0%} "
              f"{(sum(x['rd_mb'] for x in g) / tot_rd if tot_rd else 0):8.0%}")

    # the disagreement, as a single number per tool
    print("\nmachine cost per 1000 tokens spent (CPU ms)")
    for k in KINDS:
        g = by.get(k) or []
        t = sum(x["tokens"] for x in g)
        if not g or not t:
            continue
        print(f"  {k:8} {sum(x['cpu_ms'] for x in g) / t * 1000:8.2f} ms/1k tokens")

    reads, tests = by.get("read") or [], by.get("test") or []
    if reads and tests:
        rt, tt = med([x["tokens"] for x in reads]), med([x["tokens"] for x in tests])
        rc, tc = med([x["cpu_ms"] for x in reads]), med([x["cpu_ms"] for x in tests])
        print(f"\na test run against a file read, median ratios")
        print(f"  tokens: {tt / rt:5.2f}x    CPU: {tc / rc:5.2f}x")


if __name__ == "__main__":
    main()
