"""Builds the task pool for the fact-removal study from SWE-bench Verified.

The plan needs tasks that produce long traces. Short traces are useless to us
for two reasons: they give the agent no chance to learn a fact worth deleting,
and a short history means any single removed fact is a large share of the
context, so the deleted arm is no longer comparable to the kept one.

So we filter on what is knowable before running anything:
  - the real patch touches 2 or more files
  - the task is not in the quickest difficulty tier
  - there is at least one failing test to resolve

Then we rank by how much work the task is likely to take, and cap how many
come from any one repo so the sample is not one project in disguise.

Downloads through the HuggingFace rows API, so no heavy dependencies. The
dataset is cached, so this is a one-time fetch.

Run `python tasks.py` to fetch, filter and write the pool.
Run `python tasks.py --selfcheck` for the offline checks.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Verified"
CACHE = Path("data/swebench_verified.jsonl")
POOL = Path("data/task_pool.json")
PAGE = 100

KEEP = (
    "instance_id", "repo", "base_commit", "environment_setup_commit", "version",
    "problem_statement", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
    "difficulty",
)

# Human-annotated effort tiers, cheapest first. Anything matching the first
# entry is dropped: a sub-15-minute fix cannot generate recovery work.
DIFFICULTY_ORDER = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]

_DIFF_FILE = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)", re.M)


def fetch(cache: Path = CACHE, *, force: bool = False) -> list[dict]:
    """All 500 rows, cached as JSONL after the first call."""
    if cache.exists() and not force:
        return [json.loads(ln) for ln in cache.read_text(encoding="utf-8").splitlines() if ln.strip()]

    rows, offset, total = [], 0, None
    while total is None or offset < total:
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset": DATASET, "config": "default", "split": "test",
             "offset": offset, "length": PAGE}
        )
        with urllib.request.urlopen(url, timeout=120) as resp:
            page = json.load(resp)
        total = page["num_rows_total"]
        batch = [r["row"] for r in page["rows"]]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        print(f"  fetched {offset}/{total}", file=sys.stderr)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in KEEP}, ensure_ascii=False) + "\n")
    return [{k: r.get(k) for k in KEEP} for r in rows]


def get_task(instance_id: str, cache: Path = CACHE) -> dict:
    """One task row from the cache. Run `python tasks.py` first to populate it."""
    for task in fetch(cache):
        if task["instance_id"] == instance_id:
            return task
    raise KeyError(f"{instance_id} is not in {cache}")


def pool_ids(pool: Path = POOL) -> list[str]:
    """Selected instance ids, in the order they were ranked."""
    data = json.loads(pool.read_text(encoding="utf-8"))
    return [f["instance_id"] for f in data["selected"]]


def image_for(instance_id: str, tag: str = "latest") -> str:
    """Prebuilt SWE-bench environment image for a task.

    The published names escape the double underscore in an instance id as
    `_1776_`, so `django__django-11138` becomes `django_1776_django-11138`.
    Each image is roughly 1.1 GB.
    """
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:{tag}"


def patch_files(patch: str) -> list[str]:
    """Paths the patch touches, taken from its `diff --git` headers."""
    return sorted({m.group("b") for m in _DIFF_FILE.finditer(patch or "")})


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def features(task: dict) -> dict:
    """Everything we can know about effort before running the agent."""
    files = patch_files(task.get("patch", ""))
    difficulty = (task.get("difficulty") or "").strip()
    return {
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "files_touched": len(files),
        "files": files,
        "patch_lines": len((task.get("patch") or "").splitlines()),
        "n_fail_to_pass": len(_as_list(task.get("FAIL_TO_PASS"))),
        "n_pass_to_pass": len(_as_list(task.get("PASS_TO_PASS"))),
        "problem_chars": len(task.get("problem_statement") or ""),
        "difficulty": difficulty,
        "difficulty_rank": (
            DIFFICULTY_ORDER.index(difficulty) if difficulty in DIFFICULTY_ORDER else 1
        ),
    }


def eligible(f: dict) -> tuple[bool, str]:
    """The plan's filter. Returns why a task was dropped so the counts are reportable."""
    if f["difficulty"] == DIFFICULTY_ORDER[0]:
        return False, "quickest tier"
    if f["files_touched"] < 2:
        return False, "patch touches one file"
    if f["n_fail_to_pass"] < 1:
        return False, "no failing test"
    return True, "kept"


def score(f: dict) -> tuple:
    """Rank by likely trace length. Effort tier first, then breadth of the change."""
    return (
        -f["difficulty_rank"],
        -f["files_touched"],
        -f["n_fail_to_pass"],
        -f["problem_chars"],
        f["instance_id"],  # stable tie-break so the pool is reproducible
    )


def build_pool(tasks: list[dict], *, size: int = 10, max_per_repo: int = 3) -> dict:
    """Filter, rank, then take `size` with at most `max_per_repo` from one repo."""
    feats = [features(t) for t in tasks]
    dropped: dict[str, int] = {}
    kept = []
    for f in feats:
        ok, why = eligible(f)
        if ok:
            kept.append(f)
        else:
            dropped[why] = dropped.get(why, 0) + 1

    kept.sort(key=score)
    chosen, per_repo = [], {}
    for f in kept:
        if per_repo.get(f["repo"], 0) >= max_per_repo:
            continue
        chosen.append(f)
        per_repo[f["repo"]] = per_repo.get(f["repo"], 0) + 1
        if len(chosen) >= size:
            break

    return {
        "dataset": DATASET,
        "considered": len(feats),
        "eligible": len(kept),
        "dropped": dropped,
        "selected": chosen,
        "repos": per_repo,
        "max_per_repo": max_per_repo,
    }


def _selfcheck() -> None:
    """Offline. Checks the patch parsing, the filter and the ranking."""
    two_files = (
        "diff --git a/src/pay.py b/src/pay.py\n--- a/src/pay.py\n+++ b/src/pay.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/src/bill.py b/src/bill.py\n--- a/src/bill.py\n+++ b/src/bill.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    assert patch_files(two_files) == ["src/bill.py", "src/pay.py"], patch_files(two_files)
    assert patch_files("") == []

    assert image_for("django__django-11138") == (
        "swebench/sweb.eval.x86_64.django_1776_django-11138:latest"
    ), image_for("django__django-11138")

    # FAIL_TO_PASS arrives as a JSON string in the dataset, not a list
    assert _as_list('["a::b", "c::d"]') == ["a::b", "c::d"]
    assert _as_list([1, 2]) == [1, 2]
    assert _as_list("") == []
    assert _as_list(None) == []

    def make(tid, patch, diff, f2p):
        return {"instance_id": tid, "repo": "o/r", "patch": patch,
                "difficulty": diff, "FAIL_TO_PASS": f2p, "problem_statement": "x" * 100}

    good = features(make("t-good", two_files, "1-4 hours", '["a::b"]'))
    assert eligible(good)[0], eligible(good)
    assert good["files_touched"] == 2 and good["n_fail_to_pass"] == 1

    quick = features(make("t-quick", two_files, "<15 min fix", '["a::b"]'))
    assert eligible(quick) == (False, "quickest tier"), eligible(quick)

    one_file = features(make("t-one", "diff --git a/x.py b/x.py\n", "1-4 hours", '["a::b"]'))
    assert eligible(one_file) == (False, "patch touches one file"), eligible(one_file)

    no_test = features(make("t-notest", two_files, "1-4 hours", "[]"))
    assert eligible(no_test) == (False, "no failing test"), eligible(no_test)

    # an unknown difficulty string must not crash or be treated as the quickest
    odd = features(make("t-odd", two_files, "unlabelled", '["a::b"]'))
    assert eligible(odd)[0], eligible(odd)

    # the harder task ranks first
    harder = features(make("t-hard", two_files, ">4 hours", '["a::b"]'))
    assert score(harder) < score(good), (score(harder), score(good))

    # the per-repo cap holds
    many = [make(f"t-{n}", two_files, "1-4 hours", '["a::b"]') for n in range(8)]
    pool = build_pool(many, size=5, max_per_repo=2)
    assert len(pool["selected"]) == 2, pool["selected"]
    assert pool["eligible"] == 8, pool

    print("selfcheck ok: patch parsing, filter reasons, ranking and the repo cap all behave")


def main() -> None:
    if "--selfcheck" in sys.argv:
        _selfcheck()
        return

    print(f"fetching {DATASET} ...", file=sys.stderr)
    tasks = fetch()
    pool = build_pool(tasks, size=12, max_per_repo=3)

    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps(pool, indent=2), encoding="utf-8")

    print(f"\nconsidered {pool['considered']}, eligible {pool['eligible']}")
    for why, n in sorted(pool["dropped"].items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:3d}  {why}")
    print(f"\nselected {len(pool['selected'])}, at most {pool['max_per_repo']} per repo:\n")
    print(f"  {'instance_id':34} {'files':>5} {'f2p':>4} {'lines':>6}  difficulty")
    for f in pool["selected"]:
        print(f"  {f['instance_id']:34} {f['files_touched']:5d} {f['n_fail_to_pass']:4d} "
              f"{f['patch_lines']:6d}  {f['difficulty']}")
    print(f"\nrepos: {pool['repos']}")
    print(f"written to {POOL}")


if __name__ == "__main__":
    main()
