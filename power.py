"""How many pairs we need, from the variance we actually measured.

A paired design: one pair per (task, fact), each side averaged over its runs.
The relevant spread is the spread of those paired differences across tasks,
because tasks are the resampling unit.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import analyse

Z_A, Z_B = 1.959964, 0.841621  # two-sided 0.05, 80% power


def needed(effect: float, sd: float) -> int:
    if not effect or not sd:
        return 0
    return math.ceil((Z_A + Z_B) ** 2 * sd**2 / effect**2)


def main() -> None:
    rows = analyse.load_rows(Path(sys.argv[1]))
    pairs = analyse.pair_facts(analyse.aggregate(rows))
    done_arms = [r for r in rows if not r.get("failed")]
    print(f"rows: {len(rows)}   usable pairs: {len(pairs)}   "
          f"tasks with a pair: {len({p['task'] for p in pairs})}\n")
    if len(pairs) < 2:
        print("need at least two pairs before any spread can be estimated")
        return

    for field, name in (("d_total_tokens", "tokens"), ("d_calls", "calls"),
                        ("d_tool_ms", "tool ms"), ("d_redundant_total", "redundant")):
        vals = [p[field] for p in pairs if p.get(field) is not None]
        plac = [p[f"p_{field[2:]}"] for p in pairs if p.get(f"p_{field[2:]}") is not None]
        if len(vals) < 2:
            continue
        mean, sd = st.mean(vals), st.stdev(vals)
        # the fact-specific effect is how much more the real cut costs than an
        # equally sized irrelevant cut, so the placebo is subtracted
        spec = [v - p for v, p in zip(vals, plac)] if len(plac) == len(vals) else []
        print(f"{name}")
        print(f"  C2-C1 across pairs:  mean {mean:+12.1f}   SD {sd:12.1f}  -> "
              f"{needed(mean, sd):4} pairs for 80% power")
        if len(spec) >= 2:
            sm, ss = st.mean(spec), st.stdev(spec)
            print(f"  minus the placebo:   mean {sm:+12.1f}   SD {ss:12.1f}  -> "
                  f"{needed(sm, ss):4} pairs")
        print()

    cost = st.median([r.get("cost_usd", 0) for r in done_arms if r.get("cost_usd")])
    mins = st.median([r.get("seconds", 0) for r in done_arms if r.get("seconds")]) / 60
    print(f"measured per run: ${cost:.2f}, {mins:.1f} min, "
          f"{st.median([(r.get('ktrace') or {}).get('bytes_after', 0) for r in done_arms]) / 1e6:.0f} MB trace\n")
    print(f"  {'pairs':>6} {'runs':>6} {'cost':>8} {'hours':>7} {'GB':>6}   (11 runs per task:"
          f" 1 phase1 + 1 C0 + 3 trials x 3 arms)")
    for n in (10, 20, 36, 60):
        runs = n * 11
        print(f"  {n:6} {runs:6} ${runs * cost:7.0f} {runs * mins / 60:7.1f} "
              f"{runs * 32 / 1000:6.1f}")


if __name__ == "__main__":
    main()
