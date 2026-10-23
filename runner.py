"""Runs one task through all four arms and writes the results.

The order here is not arbitrary. C0 has to run before the arms are built,
because C3's placebo is chosen from facts that C0 never went back for. Get the
order wrong and the placebo is picked using information from the arm it is
meant to be a control for.

  1. start the container, apply the task's tests
  2. phase 1 until the checkpoint: first failing test after an edit
  3. commit the container, so every arm starts from identical bytes
  4. pull the facts out of the phase-1 trace
  5. run C0 from the checkpoint with the context untouched
  6. per target fact: build C1, C2, C3 from one block set, run each
  7. write one result row per arm run

The resume budget is a fraction of the full trace rather than a fixed number,
so a long investigation is cut down the same way a real compaction would cut it.
The block set is chosen once and every arm scrubs inside that same set, so the
arms differ by exactly the lines removed and nothing else.

Run `python runner.py --selfcheck` for the offline checks, which fake the
container with temp directories and a scripted model.
Run `python runner.py --pilot` for the real thing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from facts import extract_facts, relearn_cost, relearned
from harness import Workspace, phase1, resume, run
from resume import build_resume, render_blocks, task_reserve, tiktoken_counter
from tracelog import Trace, load, summarise

RUNS = Path("runs")
# Django's runner prints "test_name (module.Class)". pytest ids pass through.
_DJANGO_ID = re.compile(r"^(?P<name>\w+)\s+\((?P<path>[\w.]+)\)$")


def test_target(raw: str) -> str:
    """One SWE-bench test id, in the form its runner accepts."""
    raw = (raw or "").strip()
    if m := _DJANGO_ID.match(raw):
        return f"{m['path']}.{m['name']}"
    return raw


def as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            out = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return out if isinstance(out, list) else [out]
    return []


def task_prompt(task: dict, steps: int = 16) -> str:
    """The problem statement, how to check the fix, and the step budget.

    Three things it has to say, each learned from a run that failed without it.

    How to test. Without it the agent edits blindly and never runs a test, so
    the checkpoint never fires: one run spent 18 steps and 3 edits with zero
    tests. It cannot write its own reproduction script either, because it has
    no tool to create a file.

    But not WHICH test. Naming the test id put it in the prompt, and the prompt
    is never scrubbed, so the F_test and F_neg facts could never be deleted
    from a resume context. Both were dropped as leaked. `run_tests` with no
    arguments now runs the task's tests, so the ids stay out of the prompt.

    The step budget. One arm spent all 16 steps reading and never tested, which
    makes its pass rate meaningless. Telling it the budget pushes it to verify.

    These are hints and they belong in the method section beside the test-patch
    choice. Every arm gets the identical prompt, so the comparison is
    unaffected. Only absolute solve rates are.
    """
    return (
        f"{task['problem_statement']}\n\n"
        f"Call run_tests with no arguments to run the tests for this issue. They "
        f"fail now and should pass once the bug is fixed. You have about {steps} "
        f"tool calls, so test early rather than reading everything first."
    )


def label_facts(facts: list[dict], task: dict) -> dict:
    """Mark each fact needed or irrelevant using the task's official patch.

    This is the study's ground truth and it costs nothing: every SWE-bench task
    ships the patch that fixes the bug, so the files that must change are known
    in advance. The labels come from the benchmark, not from our agent's own
    behaviour, so using them to pick what to delete is not circular.

    Without this the target was picked by rank alone, and the first pilot
    deleted a fact about a Postgres file while the agent worked on SQLite.
    Nothing happened, and nothing could have: the agent never needed it. A
    deletion that cannot matter produces no signal and wastes the run.

    A failing test is always needed. It is the thing that has to pass, so it is
    load-bearing by definition even though it has no source path.
    """
    from tasks import patch_files

    gold = set(patch_files(task.get("patch", "")))
    for f in facts:
        path = f["fields"].get("path")
        f["needed"] = f["kind"] == "F_test" or bool(path and path in gold)
    needed = [f for f in facts if f["needed"]]
    return {"gold_files": sorted(gold), "needed": len(needed),
            "irrelevant": len(facts) - len(needed)}


def fact_slug(fact: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", fact["id"])[:80]


def test_file_hashes(ws, task: dict) -> dict:
    """Content hashes of the files the task's test patch touched.

    Taken right after the test patch is applied, so the baseline is the state
    the agent starts from rather than the original commit.
    """
    from tasks import patch_files

    hasher = getattr(ws, "hash_file", None)
    if hasher is None:
        return {}
    return {f: hasher(f) for f in patch_files(task.get("test_patch", ""))}


def tampered(ws, task: dict, baseline: dict) -> bool:
    """Did the agent edit the test file to make the failing test pass.

    Borrowed from the os-noise collector, which flags a run invalid if the
    protected test file changed. Without this an agent can delete the failing
    assertion and we would record the task as resolved. A pass on a test the
    agent rewrote is not a pass.
    """
    if not baseline:
        return False
    hasher = getattr(ws, "hash_file", None)
    if hasher is None:
        return False
    return any(hasher(f) != h for f, h in baseline.items())


def evaluate(ws, task: dict, baseline: dict | None = None) -> dict:
    """The guardrail that sits next to every cost number.

    Both halves matter. The failing tests have to pass, and the tests that
    already passed must still pass. Without the second half an agent could
    look cheap and resolved while having broken seventy other tests, and the
    guardrail would be blind to exactly the behaviour it exists to catch.

    The regression suite runs as one invocation, so it costs one test run
    rather than seventy. This is called after the trace is closed, so these
    checks never land in the agent's trace or its redundancy counts.
    """
    f2p = [test_target(t) for t in as_list(task.get("FAIL_TO_PASS")) if t]
    p2p = [test_target(t) for t in as_list(task.get("PASS_TO_PASS")) if t]

    fixed = {t: ws.run_tests(t).fields["passed"] for t in f2p}
    regress = ws.run_tests(p2p) if p2p else None
    no_regression = regress.fields["passed"] if regress is not None else True
    cheated = tampered(ws, task, baseline or {})

    return {
        "targets": len(fixed),
        "passed": sum(fixed.values()),
        "fixed": bool(fixed) and all(fixed.values()),
        "p2p_count": len(p2p),
        "no_regression": no_regression,
        "tampered": cheated,
        # a pass on a test the agent rewrote is not a pass
        "resolved": bool(fixed) and all(fixed.values()) and no_regression and not cheated,
        "detail": fixed,
    }


class DockerEnv:
    """Real environment. Each arm gets a fresh container off the checkpoint image."""

    def start(self, task: dict, arm: str):
        from dockerws import start_task

        return start_task(task, arm=arm)

    def commit(self, ws, tag: str) -> str:
        return ws.commit(tag)

    def resume(self, checkpoint: str, task: dict, arm: str):
        from dockerws import default_target, from_checkpoint

        return from_checkpoint(checkpoint, task["repo"], arm=arm,
                               targets=default_target(task))

    def hash_fn(self, ws):
        return ws.hash_file

    def release(self, ws) -> None:
        ws.remove()

    def drop(self, checkpoint: str) -> None:
        from dockerws import _run

        _run(["docker", "image", "rm", "-f", checkpoint], timeout=300)


def _trace(path: Path, ws, task: dict, arm: str, env, meta: dict) -> Trace:
    path.parent.mkdir(parents=True, exist_ok=True)
    return Trace(
        path,
        getattr(ws, "root", "."),
        run={
            "model": meta["model"],
            "tokenizer": meta["tokenizer"],
            "context_limit": meta["context_limit"],
            "task": task["instance_id"],
            "arm": arm,
            "commit": task["base_commit"],
        },
        hash_file=env.hash_fn(ws),
    )


def run_arm(
    *,
    env,
    task: dict,
    checkpoint: str,
    arm: str,
    trial: int,
    make_model,
    trace_path: Path,
    history: list[dict] | None = None,
    resume_text: str | None = None,
    fact: dict | None = None,
    built: dict | None = None,
    max_steps: int,
    meta: dict,
    test_baseline: dict | None = None,
    ktrace: bool = False,
) -> dict:
    """One arm, one trial. Either continues `history` (C0) or starts from `resume_text`."""
    started = time.time()
    session = None
    ktrace_dir = trace_path.parent / f"ktrace-{arm}-t{trial}"
    if ktrace:
        from ktrace import LttngSession, drop_caches

        # Arms run back to back from one image, so the later arm would find the
        # repo already cached and its disk reads would look cheaper for no
        # reason but ordering. Every arm starts cold.
        drop_caches()
        session = LttngSession(f"{task['instance_id']}-{arm}-{trial}", ktrace_dir)
        session.start()
    ws = env.resume(checkpoint, task, f"{arm}-{trial}")
    if ktrace:
        from ktrace import attribution

        # written while the container is alive: its pid and namespace inodes
        # cannot be read once it is gone
        ktrace_dir.mkdir(parents=True, exist_ok=True)
        (ktrace_dir / "attribution.json").write_text(
            json.dumps(attribution(getattr(ws, "container", None),
                                   extra={"task": task["instance_id"], "arm": arm,
                                          "trial": trial, "trace": str(trace_path)}),
                       indent=2),
            encoding="utf-8")
    model = make_model()
    try:
        with _trace(trace_path, ws, task, arm, env, meta) as tr:
            if history is not None:
                out = run(model, ws, tr, history=list(history), max_steps=max_steps)
            else:
                out = resume(model, ws, tr, resume_text, max_steps=max_steps)
        verdict = evaluate(ws, task, baseline=test_baseline)
    finally:
        env.release(ws)
        ktrace_info = session.stop() if session else {}
        if ktrace_info.get("traced"):
            from ktrace import compress

            ktrace_info.update(compress(ktrace_info["trace_dir"]))

    _, events = load(trace_path)
    totals = summarise(events)
    row = {
        "task": task["instance_id"],
        "arm": arm,
        "trial": trial,
        "fact_id": fact["id"] if fact else None,
        "fact_kind": fact["kind"] if fact else None,
        "fact_needed": fact.get("needed") if fact else None,
        # provenance on the row itself, so sets from different machines or
        # different parallelism can never be pooled by accident
        "nproc": meta.get("nproc"),
        "parallel": meta.get("parallel"),
        # how big a share of the context the removed fact was, and whether a
        # placebo existed for this pair
        "context_share": (built or {}).get("check", {}).get("spread"),
        "placebo_id": (built or {}).get("check", {}).get("placebo_id"),
        "reason": out["reason"],
        "steps": out["steps"],
        "calls": totals["calls"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"],
        "redundant": totals["redundant"],
        "redundant_total": totals["redundant_total"],
        "resolved": verdict["resolved"],
        "fixed": verdict["fixed"],
        "no_regression": verdict["no_regression"],
        "tampered": verdict["tampered"],
        "tests_passed": verdict["passed"],
        "tests_total": verdict["targets"],
        "p2p_count": verdict["p2p_count"],
        "model_ms": totals["model_ms"],
        "tool_ms": totals["tool_ms"],
        "k_usage_usec": sum(e.get("k_usage_usec") or 0 for e in events) or None,
        "k_rbytes": sum(e.get("k_rbytes") or 0 for e in events) or None,
        "k_wbytes": sum(e.get("k_wbytes") or 0 for e in events) or None,
        "k_pgmajfault": sum(e.get("k_pgmajfault") or 0 for e in events) or None,
        "ktrace": ktrace_info or None,
        # a run that dropped kernel records is not trustworthy for the trace
        # analysis, so the count travels with the row
        "ktrace_discarded": ktrace_info.get("discarded") if ktrace_info else None,
        "seconds": round(time.time() - started, 1),
        "cost_usd": round(getattr(model, "cost_usd", 0.0), 4),
        "trace": str(trace_path),
    }
    if fact is not None:
        row["relearned"] = relearned(fact, events, 0)
        row["relearn_cost"] = relearn_cost(fact, events, 0)
    return row


def run_task(
    instance_id: str,
    make_model,
    *,
    env=None,
    facts_per_task: int = 1,
    trials: int = 1,
    # C0 only provides the no-interruption baseline and the pool the placebo is
    # drawn from, so it does not need the arms' repeat count.
    c0_trials: int | None = None,
    budget_fraction: float = 0.6,
    budget: int | None = None,
    min_history_tokens: int = 1500,
    # largest share of the context the removed fact may be; see build_resume
    tolerance: float = 0.05,
    max_steps: int = 30,
    resume_steps: int | None = None,
    count_tokens=None,
    tokenizer: str = "cl100k_base",
    model_name: str = "unknown",
    context_limit: int = 200_000,
    # how many tasks were running at once, recorded so timing columns from a
    # parallel campaign are never read as if they were serial
    parallel: int = 1,
    ktrace: bool = False,
    outdir: Path = RUNS,
    task: dict | None = None,
) -> dict:
    """Phase 1, checkpoint, then every arm. Returns the task's result rows."""
    env = env or DockerEnv()
    # Token spend grows with the square of the step count, because every step
    # re-sends the whole history. Capping the resume arms is the single biggest
    # lever on both cost and wall clock. It costs us the success guardrail on
    # long tasks, not the re-acquisition signal, which lands in the first steps.
    resume_steps = resume_steps or max_steps
    count_tokens = count_tokens or tiktoken_counter()
    import os

    meta = {"model": model_name, "tokenizer": tokenizer, "context_limit": context_limit,
            "nproc": os.cpu_count(), "parallel": parallel}
    if task is None:
        from tasks import get_task

        task = get_task(instance_id)
    base = outdir / instance_id
    rows: list[dict] = []
    notes: list[str] = []

    ws = env.start(task, "phase1")
    checkpoint = None
    try:
        p1_path = base / "phase1.jsonl"
        with _trace(p1_path, ws, task, "phase1", env, meta) as tr:
            p1 = phase1(make_model(), ws, tr, task_prompt(task, max_steps), max_steps=max_steps)
        if not p1["checkpoint"]:
            return {
                "task": instance_id,
                "skipped": f"phase 1 never reached the checkpoint ({p1['reason']})",
                "rows": [],
                "notes": notes,
            }
        test_baseline = test_file_hashes(ws, task)
        checkpoint = env.commit(ws, f"cfa-ckpt/{instance_id}:latest".lower())
    finally:
        env.release(ws)

    _, p1_events = load(p1_path)
    facts = extract_facts(p1_events)
    labels = label_facts(facts, task)
    if not facts:
        env.drop(checkpoint)
        return {"task": instance_id, "skipped": "no facts extracted", "rows": [], "notes": notes}

    # work out the budget from the real trace so the arms can be matched
    blocks = render_blocks(p1_events, count_tokens)
    full = sum(b.tokens for b in blocks)
    # No floor here. A budget at or above the full trace means everything fits,
    # which leaves no spare history to pad the shorter arms with, and then no
    # The fraction applies to the history, not the whole context. The task text
    # is not optional, so its cost is added on top rather than competing with
    # the history for the same budget. The fraction decides how much history
    # the agent gets back; it no longer has to leave spare blocks behind,
    # because the arms are not padded.
    reserve = task_reserve(task_prompt(task, resume_steps), count_tokens)
    resume_budget = budget or reserve + int(full * budget_fraction)
    if full < min_history_tokens:
        env.drop(checkpoint)
        return {"task": instance_id,
                "skipped": f"history too thin to match arms ({full} tokens, need {min_history_tokens})",
                "rows": [], "notes": notes}

    try:
        def attempt(**kw) -> dict:
            """One arm. A failure is recorded and the batch carries on.

            The five-hour cap can stop a call at any point. Losing the arms
            that already finished because a later one failed would waste the
            most expensive part of the run.
            """
            try:
                return run_arm(**kw)
            except Exception as exc:
                row = {"task": task["instance_id"], "arm": kw["arm"], "trial": kw["trial"],
                       "fact_id": (kw.get("fact") or {}).get("id"),
                       "fact_kind": (kw.get("fact") or {}).get("kind"),
                       "fact_needed": (kw.get("fact") or {}).get("needed"),
                       # provenance belongs on a failure row too, or the merge
                       # guard sees a second "setup" that is really just errors
                       "nproc": meta.get("nproc"), "parallel": meta.get("parallel"),
                       "failed": f"{type(exc).__name__}: {exc}"[:500],
                       "trace": str(kw["trace_path"])}
                notes.append(f"{kw['arm']} trial {kw['trial']} failed: {row['failed'][:200]}")
                return row

        def keep(row: dict) -> dict:
            rows.append(row)
            _append(outdir, row)
            return row

        # C0 first. The placebo depends on what C0 never went back for.
        c0_rows = [
            keep(attempt(
                env=env, task=task, checkpoint=checkpoint, arm="C0", trial=t,
                make_model=make_model, trace_path=base / f"C0-t{t}.jsonl",
                history=p1["history"], max_steps=resume_steps, meta=meta,
                test_baseline=test_baseline, ktrace=ktrace,
            ))
            for t in range(c0_trials if c0_trials is not None else trials)
        ]
        done = [r for r in c0_rows if not r.get("failed")]
        if not done:
            return {"task": instance_id, "skipped": "every C0 trial failed",
                    "rows": rows, "notes": notes}
        _, c0_events = load(Path(done[0]["trace"]))

        # the hardest facts to lose first: negative knowledge, then test results
        order = {"F_neg": 0, "F_test": 1, "F_loc": 2, "F_dep": 3}
        facts.sort(key=lambda f: (order.get(f["kind"], 9), f["source_i"]))

        # Only a fact the fix actually depends on can show a cost when deleted.
        # The irrelevant ones become the placebo pool, which is what makes the
        # placebo arm a real control rather than a second arbitrary deletion.
        needed = [f for f in facts if f["needed"]]
        irrelevant = [f for f in facts if not f["needed"]]
        notes.append(f"{len(needed)} needed, {len(irrelevant)} irrelevant, "
                     f"from {len(labels['gold_files'])} files in the official patch")
        if not needed:
            return {"task": instance_id, "skipped": "no fact touches a file the fix changes",
                    "rows": rows, "notes": notes, "facts_found": len(facts),
                    "facts_used": 0, "resume_budget": resume_budget,
                    "task_reserve": reserve, "trace_tokens": full}

        used = 0
        for fact in needed:
            if used >= facts_per_task:
                break
            try:
                built = build_resume(
                    p1_events, fact, budget=resume_budget, count_tokens=count_tokens,
                    task_text=task_prompt(task, resume_steps), all_facts=irrelevant,
                    c0_events=c0_events, tolerance=tolerance,
                )
            except ValueError as exc:  # budget cannot hold the pinned block
                notes.append(f"dropped {fact['id']}: {exc}")
                continue
            if not built["check"]["ok"]:
                notes.append(f"dropped {fact['id']}: {'; '.join(built['check']['problems'])}")
                continue
            used += 1
            slug = fact_slug(fact)
            for arm in ("C1", "C2", "C3"):
                for t in range(trials):
                    keep(attempt(
                        env=env, task=task, checkpoint=checkpoint, arm=arm, trial=t,
                        make_model=make_model,
                        trace_path=base / slug / f"{arm}-t{t}.jsonl",
                        resume_text=built[arm].text, fact=fact, built=built,
                        max_steps=resume_steps, meta=meta, test_baseline=test_baseline,
                        ktrace=ktrace,
                    ))
            notes.append(
                f"{fact['id']}: arms at {built['check']['tokens']}, "
                f"spread {built['check']['spread']:.1%}, placebo {built['check']['placebo_id']}"
            )
    finally:
        env.drop(checkpoint)

    return {
        "task": instance_id,
        "facts_found": len(facts),
        "facts_needed": labels["needed"],
        "facts_irrelevant": labels["irrelevant"],
        "facts_used": used,
        "resume_budget": resume_budget,
        "task_reserve": reserve,
        "trace_tokens": full,
        "rows": rows,
        "notes": notes,
    }


def _append(outdir: Path, row: dict) -> None:
    """Write each row as it lands. A run that dies keeps what it already paid for."""
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def deltas(rows: list[dict]) -> list[dict]:
    """C2 minus C1 per fact. The study's headline number."""
    by: dict[tuple, dict] = {}
    for r in rows:
        if r["arm"] in ("C1", "C2", "C3") and r["fact_id"] and not r.get("failed"):
            by.setdefault((r["task"], r["fact_id"], r["trial"]), {})[r["arm"]] = r
    out = []
    for (task, fid, trial), arms in sorted(by.items()):
        if "C1" not in arms or "C2" not in arms:
            continue
        c1, c2, c3 = arms["C1"], arms["C2"], arms.get("C3")
        out.append({
            "task": task, "fact_id": fid, "fact_kind": c1["fact_kind"], "trial": trial,
            "relearned_c1": c1.get("relearned"), "relearned_c2": c2.get("relearned"),
            "relearned_c3": c3.get("relearned") if c3 else None,
            "d_tokens": c2["total_tokens"] - c1["total_tokens"],
            "d_calls": c2["calls"] - c1["calls"],
            "d_redundant": c2["redundant_total"] - c1["redundant_total"],
            "resolved_c1": c1["resolved"], "resolved_c2": c2["resolved"],
            "fixed_c1": c1["fixed"], "fixed_c2": c2["fixed"],
            "regressed_c1": not c1["no_regression"], "regressed_c2": not c2["no_regression"],
        })
    return out


def write_results(results: list[dict], outdir: Path = RUNS) -> Path:
    """Write the per-invocation summary. Never touch results.jsonl.

    Rows are appended by `_append` as each arm lands. This used to rewrite
    results.jsonl with only the rows from the current invocation, which
    destroyed the campaign's earlier data: a task that got skipped had no rows,
    so the file was truncated to nothing and twelve completed runs were lost.
    Nothing here opens that file for writing again.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [r for res in results for r in res["rows"]]
    path = outdir / "results.jsonl"
    # one summary per task, so a later invocation cannot overwrite an earlier one
    tag = results[0]["task"] if len(results) == 1 else "batch"
    (outdir / f"summary-{tag}.json").write_text(
        json.dumps({"tasks": results, "deltas": deltas(rows)}, indent=2), encoding="utf-8"
    )
    return path


def report(results: list[dict]) -> None:
    rows = [r for res in results for r in res["rows"]]
    for res in results:
        if res.get("skipped"):
            print(f"  {res['task']}: SKIPPED, {res['skipped']}")
            continue
        print(f"  {res['task']}: {res['facts_used']}/{res['facts_found']} facts, "
              f"budget {res['resume_budget']} = {res.get('task_reserve', 0)} task text "
              f"+ 60% of {res['trace_tokens']} history")
        for n in res["notes"]:
            print(f"      {n}")
    if not rows:
        print("\nno arm runs completed")
        return
    print(f"\n  {'arm':4} {'fact kind':10} {'steps':>5} {'tokens':>8} {'redund':>6} "
          f"{'relearn':>7} {'resolved':>8} {'cost':>7}")
    for r in rows:
        if r.get("failed"):
            print(f"  {r['arm']:4} {str(r['fact_kind'] or '-'):10}  FAILED  {r['failed'][:60]}")
            continue
        print(f"  {r['arm']:4} {str(r['fact_kind'] or '-'):10} {r['steps']:5d} "
              f"{r['total_tokens']:8d} {r['redundant_total']:6d} "
              f"{str(r.get('relearned', '-')):>7} {str(r['resolved']):>8} "
              f"${r['cost_usd']:.2f}")
    d = deltas(rows)
    if d:
        print(f"\n  C2 minus C1, per fact:")
        for x in d:
            print(f"    {x['fact_kind']:7} relearned C1={x['relearned_c1']} C2={x['relearned_c2']} "
                  f"C3={x['relearned_c3']}  d_tokens={x['d_tokens']:+d} "
                  f"d_calls={x['d_calls']:+d} d_redundant={x['d_redundant']:+d}")
    print(f"\n  total spend ${sum(r['cost_usd'] for r in rows):.2f} over {len(rows)} arm runs")


# -- offline check --------------------------------------------------------


class _LocalEnv:
    """Fakes the container with temp directories, so the ordering can be tested."""

    def __init__(self, root: Path, files: dict[str, str]):
        self.root, self.files, self.n = root, files, 0

    def _make(self, arm: str, src: Path | None = None) -> Workspace:
        self.n += 1
        d = self.root / f"{arm}-{self.n}"
        if src is None:
            d.mkdir(parents=True)
            for name, body in self.files.items():
                (d / name).write_text(body, encoding="utf-8")
        else:
            shutil.copytree(src, d)
        return Workspace(d, test_cmd=[sys.executable, "-m", "unittest", "-q"])

    def start(self, task, arm):
        return self._make(arm)

    def commit(self, ws, tag):
        keep = self.root / "ckpt"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(ws.root, keep)
        return str(keep)

    def resume(self, checkpoint, task, arm):
        return self._make(arm, src=Path(checkpoint))

    def hash_fn(self, ws):
        return None  # the local Workspace hashes from disk already

    def release(self, ws):
        pass  # temp directories go when the tempdir does

    def drop(self, checkpoint):
        pass


def _selfcheck() -> None:
    """Checks the id normalising, the ordering, and that result rows come out whole."""
    import tempfile

    from harness import BUGGY, TESTS, ScriptedModel

    assert test_target("test_query_convert_timezones (timezones.tests.NewDatabaseTests)") == (
        "timezones.tests.NewDatabaseTests.test_query_convert_timezones"
    ), test_target("test_query_convert_timezones (timezones.tests.NewDatabaseTests)")
    assert test_target("tests/test_pay.py::test_charge") == "tests/test_pay.py::test_charge"
    assert test_target("") == ""
    assert as_list('["a", "b"]') == ["a", "b"] and as_list(None) == []
    # runs of unsafe characters collapse to one dash, so "::" does not double up
    assert fact_slug({"id": "F_test:tests/a.py::b"}) == "F_test-tests-a.py-b", \
        fact_slug({"id": "F_test:tests/a.py::b"})

    def count(text: str) -> int:
        return len(text.split())

    search = '{"tool": "search", "args": {"pattern": "def total", "scope": "."}}'
    read = '{"tool": "read_file", "args": {"path": "calc.py", "start": 1, "end": 6}}'
    bad_fix = ('{"tool": "edit_file", "args": {"path": "calc.py", '
               '"old": "out += price * 0.9", "new": "out += price * 0.95"}}')
    test = '{"tool": "run_tests", "args": {"target": "test_calc"}}'
    good_fix = ('{"tool": "edit_file", "args": {"path": "calc.py", '
                '"old": "out += price * 0.95", "new": "out += price"}}')
    good_fix2 = ('{"tool": "edit_file", "args": {"path": "calc.py", '
                 '"old": "    return out", "new": "    return out * 0.9"}}')
    done = '{"tool": "finish", "args": {"summary": "fixed"}}'

    # Filler history of varied size. Without spare material the shorter arms
    # cannot be padded back up to C1 and every fact gets dropped.
    repo = {"calc.py": BUGGY, "test_calc.py": TESTS}
    for n in range(6):
        repo[f"mod{n}.py"] = "\n".join(
            f"def helper{n}_{j}(x):\n    return x + {j}" for j in range(1 + n)
        )
    filler = [
        f'{{"tool": "read_file", "args": {{"path": "mod{n}.py", "start": 1, "end": 40}}}}'
        for n in range(6)
    ]
    script = [*filler, search, read, bad_fix, test, done]
    after = [read, read, test, good_fix, good_fix2, test, done]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = _LocalEnv(root / "envs", repo)
        calls = {"n": 0}

        def make_model():
            calls["n"] += 1
            return ScriptedModel(list(script if calls["n"] == 1 else after))

        task = {
            "instance_id": "toy__calc-1",
            "repo": "toy/calc",
            "base_commit": "0" * 40,
            "problem_statement": "The discount is applied once per item instead of once in total.",
            "FAIL_TO_PASS": '["test_calc.TestTotal.test_single_discount"]',
            "PASS_TO_PASS": "[]",
            "test_patch": "",
            # the official fix touches calc.py only, so facts about mod*.py are
            # the irrelevant pile and the placebo has to come from there
            "patch": (
                "diff --git a/calc.py b/calc.py\n"
                "--- a/calc.py\n"
                "+++ b/calc.py\n"
            ),
        }

        res = run_task(
            "toy__calc-1", make_model, env=env, task=task, facts_per_task=1, trials=1,
            count_tokens=count, tokenizer="words", max_steps=30, outdir=root / "runs",
            # the toy repo and word counter are both small; the 1500 floor is
            # calibrated for real traces measured with tiktoken
            min_history_tokens=50,
            # the toy context is ~200 words, so any removal is a large share
            # of it by arithmetic; real traces are ten times bigger
            tolerance=0.15,
        )

        assert not res.get("skipped"), res
        # the labelling has to split the facts, not label everything one way
        assert res["facts_needed"] and res["facts_irrelevant"], res
        arms = [r["arm"] for r in res["rows"]]
        # the deleted fact must be one the fix depends on
        targets = {r["fact_needed"] for r in res["rows"] if r["arm"] in ("C1", "C2")}
        assert targets == {True}, f"a target was picked from the irrelevant pile: {targets}"
        # C0 must come first, because C3's placebo is chosen from what C0 skipped
        assert arms[0] == "C0", arms
        assert set(arms) == {"C0", "C1", "C2", "C3"}, arms
        assert res["facts_used"] == 1, res

        for r in res["rows"]:
            for key in ("steps", "calls", "total_tokens", "redundant_total", "resolved", "seconds"):
                assert key in r, (key, r)
            assert r["total_tokens"] > 0, r
        for r in res["rows"]:
            if r["arm"] != "C0":
                assert "relearned" in r, r

        # the resumes re-read and re-run, so the rules must see it
        resumed = [r for r in res["rows"] if r["arm"] == "C2"]
        assert resumed and resumed[0]["redundant_total"] > 0, resumed

        # the scripted fix really resolves the task, so the guardrail can move
        assert any(r["resolved"] for r in res["rows"]), [r["resolved"] for r in res["rows"]]

        d = deltas(res["rows"])
        assert len(d) == 1 and d[0]["fact_kind"], d
        for key in ("d_tokens", "d_calls", "d_redundant", "relearned_c1", "relearned_c2"):
            assert key in d[0], (key, d[0])

        out = write_results([res], outdir=root / "runs")
        assert out.exists() and out.read_text(encoding="utf-8").strip(), "no results written"
        assert (root / "runs" / "summary-toy__calc-1.json").exists()
        # a skipped task must not wipe the rows a finished task already wrote
        before = out.read_text(encoding="utf-8")
        write_results([{"task": "toy__skipped", "skipped": "no checkpoint",
                        "rows": [], "notes": []}], outdir=root / "runs")
        assert out.read_text(encoding="utf-8") == before, "a skipped task truncated results"

    print(f"selfcheck ok: C0 ran first, {len(res['rows'])} arm runs, "
          f"{len(d)} delta row, results written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="run the first --tasks of the pool")
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--facts", type=int, default=1)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--c0-trials", type=int, default=None,
                    help="runs of the no-interruption arm; defaults to --trials")
    ap.add_argument("--max-steps", type=int, default=30, help="cap for phase 1")
    ap.add_argument("--resume-steps", type=int, default=16, help="cap for each resume arm")
    ap.add_argument("--jobs", type=int, default=1, help="tasks to run at once")
    ap.add_argument("--parallel", type=int, default=1,
                    help="tasks running at once, recorded in every row")
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="largest share of the context the removed fact may be")
    ap.add_argument("--ktrace", action="store_true",
                    help="collect kernel counters and an LTTng trace (needs Linux and sudo)")
    ap.add_argument("--outdir", default=str(RUNS),
                    help="where runs go; use a separate one on the trace VM")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--only", default=None, help="one instance id instead of the pool")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    if args.ktrace:
        from ktrace import set_comm

        set_comm("cfa-harness")
    if not args.pilot and not args.only:
        ap.error("pass --pilot, --only <instance_id>, or --selfcheck")

    from clicode import ClaudeCliModel
    from tasks import pool_ids

    ids = [args.only] if args.only else pool_ids()[: args.tasks]
    print(f"running {len(ids)} task(s) on {args.model}: {', '.join(ids)}\n")

    def make_model():
        return ClaudeCliModel(args.model)

    def one(iid: str) -> dict:
        try:
            return run_task(
                iid, make_model, facts_per_task=args.facts, trials=args.trials,
                c0_trials=args.c0_trials,
                max_steps=args.max_steps, resume_steps=args.resume_steps,
                model_name=args.model, ktrace=args.ktrace, outdir=Path(args.outdir),
                tolerance=args.tolerance, parallel=args.parallel,
            )
        except Exception as exc:  # one bad task must not sink the batch
            return {"task": iid, "skipped": f"{type(exc).__name__}: {exc}", "rows": [], "notes": []}

    if args.jobs > 1:
        # Threads, not processes: every step is a subprocess call, so this is
        # all IO wait. Total tokens do not change, only how fast they are spent,
        # which means a usage cap arrives sooner.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(one, iid): iid for iid in ids}
            results = []
            for fut in futures:
                results.append(fut.result())
                done = results[-1]
                print(f"  done {done['task']}: "
                      f"{len(done['rows'])} rows{', ' + done['skipped'][:80] if done.get('skipped') else ''}")
    else:
        results = []
        for iid in ids:
            print(f"--- {iid} ---")
            results.append(one(iid))

    print()
    report(results)
    print(f"\nwritten to {write_results(results, outdir=Path(args.outdir))}")


if __name__ == "__main__":
    main()
