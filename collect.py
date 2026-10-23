"""Merges a campaign's per-task results into one file, and refuses to mix sets.

Each task writes its own results.jsonl so parallel jobs cannot interleave. This
walks those files, checks they came from one machine under one parallelism, and
writes the canonical file the analysis reads plus a manifest describing the
campaign. Mixing runs from different hardware is refused rather than warned
about: core count changes how a parallel test suite spends CPU, so pooling a
4-core run with a 16-core one silently corrupts the comparison.

    python collect.py runs-main        # merge and describe
    python analyse.py --results runs-main/all-results.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def merge(root: Path, *, strict: bool = True) -> dict:
    files = sorted(root.glob("*/results.jsonl"))
    rows: list[dict] = []
    for f in files:
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    if not rows:
        return {"rows": 0, "tasks": 0}

    # only rows that carry measurements decide the setup; a failed run has no
    # numbers in it and must not look like a second machine
    hw = Counter((r.get("nproc"), r.get("parallel")) for r in rows if not r.get("failed"))
    if strict and len(hw) > 1:
        raise SystemExit(
            "refusing to merge: these rows came from more than one setup "
            f"(nproc, parallel) -> {dict(hw)}. Split them before analysing."
        )

    out = root / "all-results.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if not r.get("failed")]
    arms = [r for r in ok if r.get("fact_id")]
    pairs = {(r["task"], r["fact_id"]) for r in arms if r["arm"] in ("C1", "C2")}
    shares = [r["context_share"] for r in arms if r.get("context_share") is not None]
    manifest = {
        "rows": len(rows),
        "tasks_attempted": len(files),
        "tasks_with_arms": len({r["task"] for r in arms}),
        "pairs": len(pairs),
        "with_placebo": len({(r["task"], r["fact_id"]) for r in arms if r.get("placebo_id")}),
        "failed_arms": len(rows) - len(ok),
        "failures": dict(Counter(
            "session limit" if "session limit" in (r.get("failed") or "")
            else (r.get("failed") or "").split(":")[0]
            for r in rows if r.get("failed"))),
        "setup": [{"nproc": k[0], "parallel": k[1], "rows": v} for k, v in hw.items()],
        "by_arm": dict(Counter(r["arm"] for r in ok)),
        "fact_kinds": dict(Counter(r["fact_kind"] for r in arms if r.get("fact_kind"))),
        "context_share": {
            "min": round(min(shares), 4), "max": round(max(shares), 4),
            "median": round(sorted(shares)[len(shares) // 2], 4),
        } if shares else None,
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in ok), 2),
        "merged_file": str(out),
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs-main")
    if not root.is_dir():
        print(f"{root} not found")
        return
    m = merge(root, strict="--allow-mixed" not in sys.argv)
    if not m.get("rows"):
        print(f"no rows under {root}")
        return
    print(f"{m['rows']} rows from {m['tasks_attempted']} tasks -> {m['merged_file']}")
    print(f"  usable pairs: {m['pairs']}   with a placebo: {m['with_placebo']}")
    print(f"  arms: {m['by_arm']}")
    print(f"  fact kinds: {m['fact_kinds']}")
    if m["context_share"]:
        c = m["context_share"]
        print(f"  context share: {c['min']:.1%} to {c['max']:.1%}, median {c['median']:.1%}")
    print(f"  setup: {m['setup']}")
    print(f"  spend: ${m['cost_usd']}")
    print(f"\nnext: python analyse.py --results {m['merged_file']}")


if __name__ == "__main__":
    main()
