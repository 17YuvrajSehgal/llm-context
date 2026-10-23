"""Prints the collected rows as a table, including the kernel columns.

Separate from analyse.py on purpose: this is for watching a campaign while it
runs, so it shows raw per-run rows rather than averaging anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs-vm/results.jsonl")
    if not path.exists():
        print(f"{path} not found")
        return
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"{len(rows)} rows in {path}\n")

    head = (f"{'task':22} {'arm':3} {'t':1} {'kind':7} {'step':>4} {'tokens':>8} "
            f"{'cpu_ms':>7} {'rd_MB':>6} {'wr_MB':>6} {'majflt':>6} "
            f"{'relrn':>5} {'solved':>6} {'tamp':>4} {'disc':>4} {'trace_MB':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        task = r.get("task", "?").split("__")[-1][:22]
        if r.get("failed"):
            print(f"{task:22} {r.get('arm','?'):3} {r.get('trial','?')} FAILED  {r['failed'][:60]}")
            continue
        kt = r.get("ktrace") or {}
        print(
            f"{task:22} {r['arm']:3} {r.get('trial', 0)} "
            f"{str(r.get('fact_kind') or '-'):7} {r['steps']:4} {r['total_tokens']:8} "
            f"{(r.get('k_usage_usec') or 0) / 1000:7.0f} "
            f"{(r.get('k_rbytes') or 0) / 1e6:6.1f} {(r.get('k_wbytes') or 0) / 1e6:6.1f} "
            f"{r.get('k_pgmajfault') or 0:6} "
            f"{str(r.get('relearned', '-')):>5} {str(r['resolved']):>6} "
            f"{str(r.get('tampered')):>4} {str(r.get('ktrace_discarded')):>4} "
            f"{(kt.get('bytes_after') or 0) / 1e6:8.1f}"
        )

    ok = [r for r in rows if not r.get("failed")]
    if ok:
        print(f"\nspend ${sum(r.get('cost_usd', 0) for r in ok):.2f} over {len(ok)} runs")
        traced = [r for r in ok if (r.get('ktrace') or {}).get('bytes_after')]
        if traced:
            total = sum(r['ktrace']['bytes_after'] for r in traced) / 1e6
            print(f"kernel traces: {len(traced)} archives, {total:.0f} MB compressed")
        dropped = [r for r in ok if r.get("ktrace_discarded")]
        if dropped:
            print(f"WARNING: {len(dropped)} run(s) discarded kernel events and are not "
                  f"trustworthy for the trace analysis")


if __name__ == "__main__":
    main()
