"""Turns results.jsonl into the paper's numbers and its main figure.

Agents are stochastic, so every (task, fact, arm) is run several times and the
trials are averaged before anything is compared. Spread across trials is
reported next to every mean, because an effect smaller than the run-to-run
variation is not an effect.

The hierarchy matters. Average over trials first, then difference the arms,
then bootstrap over tasks. Tasks are the resampling unit because two facts from
one task share a repo, a checkpoint and an agent run, so treating them as
independent would make the interval look tighter than the evidence supports.

The headline test is McNemar's exact test on whether the agent went back for
the fact. Only pairs that disagree carry information, so with a small sample
the exact binomial is the honest choice rather than the chi-squared
approximation.

Model wait and tool execution are kept apart. Wall clock includes process spawn
and rate-limit waits, so only the tool number is clean.

Nothing here invents data. A cell with no runs prints empty, and the sample
size sits next to every number.

Run `python analyse.py --selfcheck` for the offline checks.
Run `python analyse.py` to read runs/results.jsonl and write the figure.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

RUNS = Path("runs")
KINDS = ["F_neg", "F_test", "F_loc", "F_dep"]
LABEL = {
    "F_neg": "failed fixes",
    "F_test": "test results",
    "F_loc": "where code lives",
    "F_dep": "interfaces",
}
# averaged per arm, then differenced
COSTS = ("total_tokens", "calls", "redundant_total", "tool_ms", "model_ms")


def load_rows(path: Path = RUNS / "results.jsonl") -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _sd(xs) -> float | None:
    """Sample standard deviation. None for a single trial, which is honest:
    one run tells you nothing about run-to-run variation."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def aggregate(rows: list[dict]) -> dict:
    """One record per (task, fact, arm), averaged over its trials."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    degenerate = 0
    for r in rows:
        if r.get("failed") or not r.get("fact_id") or r["arm"] not in ("C1", "C2", "C3"):
            continue
        # A run that made no tool call carries no information. Counting it as
        # "did not re-learn" would be true only because it did nothing, and it
        # would drag an arm's average toward zero for the wrong reason.
        if not r.get("steps"):
            degenerate += 1
            continue
        by[(r["task"], r["fact_id"], r["arm"])].append(r)
    if degenerate:
        print(f"  note: dropped {degenerate} run(s) that made no tool call")

    out = {}
    for key, rs in by.items():
        rec = {
            "trials": len(rs),
            "kind": rs[0].get("fact_kind"),
            "needed": rs[0].get("fact_needed"),
            "relearn_rate": _mean([bool(r.get("relearned")) for r in rs]),
            "resolved_rate": _mean([bool(r.get("resolved")) for r in rs]),
        }
        for field in COSTS:
            rec[field] = _mean([r.get(field) for r in rs])
            rec[f"{field}_sd"] = _sd([r.get(field) for r in rs])
        out[key] = rec
    return out


def pair_facts(agg: dict) -> list[dict]:
    """Paired records per (task, fact), built from the trial averages."""
    facts = sorted({(t, f) for (t, f, _) in agg})
    out = []
    for task, fid in facts:
        c1, c2 = agg.get((task, fid, "C1")), agg.get((task, fid, "C2"))
        if not c1 or not c2:
            continue
        c3 = agg.get((task, fid, "C3"))
        rec = {
            "task": task, "fact_id": fid, "kind": c1["kind"], "needed": c1["needed"],
            "trials": min(c1["trials"], c2["trials"]),
            "relearn_c1": c1["relearn_rate"], "relearn_c2": c2["relearn_rate"],
            "relearn_c3": c3["relearn_rate"] if c3 else None,
            "resolved_c1": c1["resolved_rate"], "resolved_c2": c2["resolved_rate"],
            "d_relearn": c2["relearn_rate"] - c1["relearn_rate"],
        }
        for field in COSTS:
            if c1.get(field) is None or c2.get(field) is None:
                rec[f"d_{field}"] = None
                continue
            rec[f"d_{field}"] = c2[field] - c1[field]
            rec[f"p_{field}"] = c3[field] - c1[field] if c3 and c3.get(field) is not None else None
        return_sd = c2.get("total_tokens_sd")
        rec["tokens_sd_c2"] = return_sd
        out.append(rec)
    return out


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value, summing every outcome at most as likely."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    observed = probs[k]
    # a tiny slack absorbs float error on the symmetric tail
    return min(1.0, sum(x for x in probs if x <= observed * (1 + 1e-9)))


def mcnemar(pairs: list[dict]) -> dict:
    """Did deleting the fact change whether the agent went back for it.

    A rate above a half counts as yes, so a fact is called re-learned when it
    happened in most of its trials. Only discordant pairs count: b is
    deleted-yes with kept-no, the direction H_forget predicts.
    """
    yes1 = [p["relearn_c1"] > 0.5 for p in pairs]
    yes2 = [p["relearn_c2"] > 0.5 for p in pairs]
    b = sum(1 for a, c in zip(yes1, yes2) if c and not a)
    c = sum(1 for a, cc in zip(yes1, yes2) if a and not cc)
    return {
        "n": len(pairs),
        "deleted_only": b,
        "kept_only": c,
        "both": sum(1 for a, cc in zip(yes1, yes2) if a and cc),
        "neither": sum(1 for a, cc in zip(yes1, yes2) if not a and not cc),
        "discordant": b + c,
        "p_value": _binom_two_sided(min(b, c), b + c) if (b + c) else 1.0,
        "rate_c1": _mean([p["relearn_c1"] for p in pairs]),
        "rate_c2": _mean([p["relearn_c2"] for p in pairs]),
    }


def bootstrap_ci(
    pairs: list[dict], field: str, *, reps: int = 10_000, seed: int = 20261021, level: float = 0.95
) -> dict:
    """Mean paired difference with a confidence interval, resampling whole tasks."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        if p.get(field) is not None:
            by_task[p["task"]].append(p[field])
    vals = [v for group in by_task.values() for v in group]
    if not vals:
        return {"n": 0, "tasks": 0, "mean": None, "lo": None, "hi": None}
    tasks = list(by_task)
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        picked = [by_task[rng.choice(tasks)] for _ in tasks]
        flat = [v for group in picked for v in group]
        means.append(sum(flat) / len(flat))
    means.sort()
    return {
        "n": len(vals), "tasks": len(tasks),
        "mean": sum(vals) / len(vals),
        "lo": means[int((1 - level) / 2 * reps)],
        "hi": means[min(reps - 1, int((1 + level) / 2 * reps))],
    }


def by_kind(pairs: list[dict]) -> dict:
    """Re-learn rates per fact kind. The table engineers will actually use."""
    out = {}
    for kind in KINDS:
        rows = [p for p in pairs if p["kind"] == kind]
        if not rows:
            out[kind] = {"n": 0}
            continue
        c3 = [p for p in rows if p["relearn_c3"] is not None]
        out[kind] = {
            "n": len(rows),
            "trials": _mean([p["trials"] for p in rows]),
            "kept": _mean([p["relearn_c1"] for p in rows]),
            "deleted": _mean([p["relearn_c2"] for p in rows]),
            "placebo": _mean([p["relearn_c3"] for p in c3]) if c3 else None,
            "d_tokens": _mean([p["d_total_tokens"] for p in rows]),
        }
    return out


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Interval for a proportion that stays sensible at n=1, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(max(0.0, p * (1 - p)) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def figure(pairs: list[dict], out: Path = RUNS / "figure1.png") -> Path | None:
    """Fact kind against how often it was re-learned, kept vs deleted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [k for k in KINDS if any(p["kind"] == k for p in pairs)]
    if not present:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.36
    for offset, field, colour, name in (
        (-width / 2, "relearn_c1", "#4a7fb5", "fact kept (C1)"),
        (width / 2, "relearn_c2", "#c4564a", "fact deleted (C2)"),
    ):
        heights, errs = [], [[], []]
        for kind in present:
            rows = [p for p in pairs if p["kind"] == kind]
            rate = _mean([p[field] for p in rows]) or 0.0
            lo, hi = wilson(rate * len(rows), len(rows))
            heights.append(rate)
            # a Wilson interval is a score interval, not centred on the observed
            # rate, so it can sit a hair below it at p=1. Clamp or matplotlib
            # refuses a negative error bar.
            errs[0].append(max(0.0, rate - lo))
            errs[1].append(max(0.0, hi - rate))
        xs = [i + offset for i in range(len(present))]
        ax.bar(xs, heights, width, label=name, color=colour, zorder=3)
        ax.errorbar(xs, heights, yerr=errs, fmt="none", ecolor="#333", capsize=4, lw=1.2, zorder=4)

    # the placebo is a check, not a condition, so it gets a marker
    px, py = [], []
    for i, kind in enumerate(present):
        rows = [p for p in pairs if p["kind"] == kind and p["relearn_c3"] is not None]
        if rows:
            px.append(i)
            py.append(_mean([p["relearn_c3"] for p in rows]))
    if px:
        ax.scatter(px, py, marker="D", s=46, color="#2f7d4f", zorder=5,
                   label="placebo (C3), irrelevant fact deleted")

    counts = [sum(1 for p in pairs if p["kind"] == k) for k in present]
    trials = [_mean([p["trials"] for p in pairs if p["kind"] == k]) or 0 for k in present]
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{LABEL[k]}\n(n={n}, {t:.0f} runs each)"
                        for k, n, t in zip(present, counts, trials)])
    ax.set_ylabel("how often the agent went back for the fact")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Does losing a fact make the agent re-acquire it?", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def report(rows: list[dict]) -> dict:
    agg = aggregate(rows)
    pairs = pair_facts(agg)
    failed = [r for r in rows if r.get("failed")]
    single = [k for k, v in agg.items() if v["trials"] < 2]
    print(f"rows: {len(rows)}  arms: {len(agg)}  usable pairs: {len(pairs)}  "
          f"failed arms: {len(failed)}")
    if single:
        print(f"  warning: {len(single)} of {len(agg)} arms have only one run, "
              f"so their spread is unknown")
    if not pairs:
        print("\nno complete C1/C2 pair yet, so there is nothing to test.")
        return {"pairs": 0}

    m = mcnemar(pairs)
    print(f"\nwent back for the fact, averaged over runs:")
    print(f"  kept (C1)    {m['rate_c1']:.0%}")
    print(f"  deleted (C2) {m['rate_c2']:.0%}")
    print(f"  pairs where they differ: {m['discordant']} of {m['n']} "
          f"(deleted only {m['deleted_only']}, kept only {m['kept_only']})")
    print(f"  McNemar exact p = {m['p_value']:.4f}")
    if m["discordant"] == 0:
        print("  every pair agrees, so this cannot separate the two ideas yet")

    print("\npaired difference, C2 minus C1 (95% interval over tasks):")
    names = {"total_tokens": "tokens", "calls": "calls", "redundant_total": "redundant",
             "tool_ms": "tool ms", "model_ms": "model ms"}
    for field, name in names.items():
        ci = bootstrap_ci(pairs, f"d_{field}")
        if ci["mean"] is None:
            continue
        print(f"  {name:10} {ci['mean']:+12.1f}  [{ci['lo']:+.1f}, {ci['hi']:+.1f}]  "
              f"n={ci['n']} over {ci['tasks']} task(s)")
    print("  (model ms carries process spawn and rate-limit waits; tool ms is clean)")

    print("\nsame difference for the placebo, C3 minus C1:")
    for field, name in names.items():
        ci = bootstrap_ci(pairs, f"p_{field}")
        if ci["mean"] is None:
            continue
        print(f"  {name:10} {ci['mean']:+12.1f}  [{ci['lo']:+.1f}, {ci['hi']:+.1f}]")

    print("\nrun-to-run spread within an arm (C2 tokens):")
    sds = [p["tokens_sd_c2"] for p in pairs if p.get("tokens_sd_c2") is not None]
    if sds:
        print(f"  mean SD over {len(sds)} fact(s): {_mean(sds):,.0f} tokens")
        print("  compare that against the C2 minus C1 mean above before claiming an effect")
    else:
        print("  unknown, every arm has a single run")

    print("\nby fact kind:")
    print(f"  {'kind':18} {'n':>3} {'runs':>5} {'kept':>6} {'deleted':>8} {'placebo':>8}")
    for kind, s in by_kind(pairs).items():
        if not s["n"]:
            continue
        pl = f"{s['placebo']:.0%}" if s["placebo"] is not None else "-"
        print(f"  {LABEL[kind]:18} {s['n']:3d} {s['trials']:5.1f} "
              f"{s['kept']:6.0%} {s['deleted']:8.0%} {pl:>8}")

    print(f"\ntask solved: kept {_mean([p['resolved_c1'] for p in pairs]):.0%}, "
          f"deleted {_mean([p['resolved_c2'] for p in pairs]):.0%}")

    path = figure(pairs)
    if path:
        print(f"\nfigure written to {path} and {path.with_suffix('.pdf')}")
    return {"pairs": len(pairs), "mcnemar": m}


def _selfcheck() -> None:
    """Checks the stats on data where the right answer is known by hand."""
    assert abs(_binom_two_sided(0, 1) - 1.0) < 1e-9
    assert abs(_binom_two_sided(0, 5) - 2 * 0.5**5) < 1e-9
    assert abs(_binom_two_sided(0, 10) - 2 * 0.5**10) < 1e-9
    assert _binom_two_sided(5, 10) == 1.0
    assert _mean([1, 2, 3]) == 2 and _mean([]) is None
    assert _sd([5]) is None, "one trial cannot have a spread"
    assert abs(_sd([1, 3]) - math.sqrt(2)) < 1e-9

    def row(task, fid, arm, relearned, tokens, calls, *, trial=0, redundant=0,
            resolved=True, kind="F_test", needed=True, tool_ms=100.0, model_ms=900.0,
            steps=None):
        return {"task": task, "fact_id": fid, "arm": arm, "fact_kind": kind,
                "fact_needed": needed, "trial": trial, "relearned": relearned,
                "total_tokens": tokens, "calls": calls, "redundant_total": redundant,
                "resolved": resolved, "tool_ms": tool_ms, "model_ms": model_ms,
                "steps": calls if steps is None else steps}

    # three tasks, two facts each, two runs each. Deleting always causes a
    # re-learn, keeping never does, and the second run costs 100 more.
    rows = []
    for t in range(3):
        for i in range(2):
            fid = f"F_test:t{t}-{i}"
            for trial in range(2):
                bump = 100 * trial
                rows += [
                    row(f"task{t}", fid, "C1", False, 1000 + bump, 10, trial=trial),
                    row(f"task{t}", fid, "C2", True, 1400 + bump, 14, trial=trial, redundant=2),
                    row(f"task{t}", fid, "C3", False, 1020 + bump, 10, trial=trial),
                ]

    agg = aggregate(rows)
    assert len(agg) == 18, len(agg)  # 3 tasks x 2 facts x 3 arms
    one = agg[("task0", "F_test:t0-0", "C1")]
    assert one["trials"] == 2 and one["total_tokens"] == 1050.0, one
    assert abs(one["total_tokens_sd"] - math.sqrt(5000)) < 1e-6, one["total_tokens_sd"]
    assert one["relearn_rate"] == 0.0 and agg[("task0", "F_test:t0-0", "C2")]["relearn_rate"] == 1.0

    pairs = pair_facts(agg)
    assert len(pairs) == 6, len(pairs)
    assert pairs[0]["d_total_tokens"] == 400.0, pairs[0]
    assert pairs[0]["p_total_tokens"] == 20.0, pairs[0]  # placebo barely moves
    assert pairs[0]["trials"] == 2

    m = mcnemar(pairs)
    assert (m["deleted_only"], m["kept_only"]) == (6, 0), m
    assert m["rate_c1"] == 0.0 and m["rate_c2"] == 1.0, m
    assert abs(m["p_value"] - 2 * 0.5**6) < 1e-9, m["p_value"]

    ci = bootstrap_ci(pairs, "d_total_tokens", reps=500)
    assert ci["mean"] == 400.0 and ci["tasks"] == 3, ci
    assert ci["lo"] == ci["hi"] == 400.0, ci  # every task identical, no spread

    k = by_kind(pairs)
    assert k["F_test"]["n"] == 6 and k["F_test"]["deleted"] == 1.0, k["F_test"]
    assert k["F_test"]["trials"] == 2.0 and k["F_test"]["placebo"] == 0.0, k["F_test"]
    assert k["F_loc"]["n"] == 0

    # a failed arm must not become data, and must not break the aggregate
    rows.append({"task": "task9", "fact_id": "F_test:x", "arm": "C2", "fact_kind": "F_test",
                 "trial": 0, "failed": "ModelUnavailable: cap"})
    assert len(pair_facts(aggregate(rows))) == 6, "a failed arm leaked in"
    # an unpaired fact is ignored rather than half-counted
    rows.append(row("task9", "F_test:lonely", "C1", False, 900, 9))
    assert len(pair_facts(aggregate(rows))) == 6, "an unpaired C1 was counted"

    # a single-run arm reports no spread instead of a fake zero
    solo = [row("t", "f", "C1", False, 100, 1), row("t", "f", "C2", True, 200, 2)]
    sp = pair_facts(aggregate(solo))
    assert sp[0]["trials"] == 1 and sp[0]["tokens_sd_c2"] is None, sp[0]

    # a run that made no tool call is not an observation
    none_done = [row("z", "f", "C1", False, 0, 0, steps=0),
                 row("z", "f", "C2", True, 100, 5, steps=5)]
    assert ("z", "f", "C1") not in aggregate(none_done), "a zero-step run became data"
    assert ("z", "f", "C2") in aggregate(none_done), "a real run was dropped"

    # no disagreement means no claim
    flat = [row("t", "f", "C1", True, 1, 1), row("t", "f", "C2", True, 1, 1)]
    m2 = mcnemar(pair_facts(aggregate(flat)))
    assert m2["discordant"] == 0 and m2["p_value"] == 1.0, m2

    for k_, n_ in ((0, 1), (1, 1), (3, 6), (0, 0)):
        lo, hi = wilson(k_, n_)
        assert 0.0 <= lo <= hi <= 1.0, (k_, n_, lo, hi)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = figure(pairs, Path(tmp) / "f1.png")
        assert out and out.exists() and out.stat().st_size > 5000, out
        assert out.with_suffix(".pdf").exists()

    print("selfcheck ok: trial averaging, spread, McNemar, bootstrap, per-kind, figure")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--results", default=str(RUNS / "results.jsonl"))
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    rows = load_rows(Path(args.results))
    if not rows:
        print(f"{args.results} is empty or missing. Run the pilot first.")
        return
    report(rows)


if __name__ == "__main__":
    main()
