"""Runs the agent's tools inside a SWE-bench container.

Claude Code never goes in the container. The model runs on the host and only
the tools execute inside, over `docker exec`. So there is nothing to install or
log into in the image.

The interface matches harness.Workspace, so harness.run takes either one
without changes.

Why containers matter beyond environment setup. At the checkpoint we `docker
commit` the container into an image, and every arm then starts from a fresh
container built from that image. All four arms therefore begin with a
byte-identical repo, which is what lets us say the arms differ by one fact and
nothing else.

Every exec uses a login shell. With `bash -c` you land in conda base on Python
3.11 and the task's dependencies are gone; `bash -lc` activates the `testbed`
env the image ships.

Two Windows traps. Git Bash rewrites `/testbed` into a Windows path, so a
`docker exec` typed by hand needs `MSYS_NO_PATHCONV=1`. Nothing here is
affected, because subprocess is given argument lists rather than a shell
string. And the containers' Python 3.6 prints `--version` to stderr.

Run `python dockerws.py` for the self-check. It needs Docker and one pulled
image, and it makes no model calls.
"""

from __future__ import annotations

import difflib
import re
import shlex
import subprocess

from harness import EDIT, MAX_HITS, MAX_READ_LINES, READ, SEARCH, TEST, ToolResult

WORKDIR = "/testbed"

# Most SWE-bench repos run pytest. Django ships its own runner.
PYTEST = ["python", "-m", "pytest", "-rA", "--no-header"]
TEST_CMD = {
    "django/django": ["./tests/runtests.py", "--verbosity", "1", "--settings=test_sqlite"],
}

_GREP_PREFIX = re.compile(r"^\./")


def _run(args: list[str], *, timeout: int = 900, stdin: bytes | None = None):
    return subprocess.run(
        args, capture_output=True, timeout=timeout,
        input=stdin if stdin is not None else None,
    )


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def docker_ok() -> bool:
    try:
        return _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def have_image(image: str) -> bool:
    return _run(["docker", "image", "inspect", image], timeout=120).returncode == 0


def pull(image: str) -> None:
    proc = _run(["docker", "pull", image], timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"could not pull {image}: {_text(proc.stderr)[:300]}")


def start_container(image: str, name: str) -> str:
    """A container that stays up so we can exec into it. Replaces one of the same name."""
    _run(["docker", "rm", "-f", name], timeout=120)
    proc = _run(["docker", "run", "-d", "--name", name, "-w", WORKDIR, image, "sleep", "infinity"])
    if proc.returncode != 0:
        raise RuntimeError(f"could not start {name}: {_text(proc.stderr)[:300]}")
    return _text(proc.stdout).strip()


class DockerWorkspace:
    """Tool access to a repo inside a container, same shape as harness.Workspace."""

    def __init__(self, container: str, repo: str, *, workdir: str = WORKDIR, timeout: int = 900,
                 default_target: list[str] | None = None):
        self.container, self.repo, self.workdir, self.timeout = container, repo, workdir, timeout
        self.test_cmd = TEST_CMD.get(repo, PYTEST)
        # see harness.Workspace: keeps the task's test ids out of the prompt
        self.default_target = list(default_target or [])
        # kernel counters for this container, when the host can provide them
        self.counters = None

    # -- plumbing --

    def _exec(self, script: str, *, stdin: bytes | None = None, timeout: int | None = None):
        args = ["docker", "exec"]
        if stdin is not None:
            args.append("-i")
        args += [self.container, "bash", "-lc", script]
        return _run(args, timeout=timeout or self.timeout, stdin=stdin)

    def _safe(self, path: str) -> str:
        """Keep the agent inside the repo and keep its paths out of the shell.

        The container is a sandbox, so an escape is not dangerous to the host.
        It would still wreck the run, and an unquoted path would let a crafted
        argument run arbitrary commands and corrupt the trace.
        """
        clean = (path or "").strip().replace("\\", "/").lstrip("/")
        if not clean or ".." in clean.split("/"):
            raise ValueError(f"{path!r} is outside the repo")
        return clean

    def _cat(self, path: str) -> str | None:
        proc = self._exec(f"cat -- {shlex.quote(path)}", timeout=120)
        return _text(proc.stdout) if proc.returncode == 0 else None

    def hash_file(self, path: str) -> str | None:
        """Matches tracelog's hash: sha256, first 16 hex characters."""
        try:
            rel = self._safe(path)
        except ValueError:
            return None
        proc = self._exec(f"sha256sum -- {shlex.quote(rel)} 2>/dev/null", timeout=120)
        out = _text(proc.stdout).split()
        return out[0][:16] if proc.returncode == 0 and out else None

    # -- tools --

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> ToolResult:
        try:
            rel = self._safe(path)
        except ValueError as exc:
            return ToolResult(READ, str(exc), {"path": path, "lines": None})
        body = self._cat(rel)
        if body is None:
            return ToolResult(READ, f"no such file: {path}", {"path": rel, "lines": None})
        lines = body.splitlines()
        lo = max(1, start or 1)
        hi = min(len(lines), end or min(len(lines), lo + MAX_READ_LINES - 1))
        return ToolResult(READ, "\n".join(lines[lo - 1 : hi]), {"path": rel, "lines": (lo, hi)})

    def search(self, pattern: str, scope: str = ".") -> ToolResult:
        """grep -rn, which already emits the `path:line:text` shape facts.py reads.

        grep uses POSIX extended regex, not Python's dialect. The same engine
        runs for every arm, so the comparison is unaffected.
        """
        fields = {"pattern": pattern, "scope": scope}
        if not pattern:
            return ToolResult(SEARCH, "no pattern given", fields)
        try:
            rel = "." if scope in ("", ".") else self._safe(scope)
        except ValueError as exc:
            return ToolResult(SEARCH, str(exc), fields)
        cmd = (
            f"grep -rnE --include='*.py' --exclude-dir=.git -m {MAX_HITS} -- "
            f"{shlex.quote(pattern)} {shlex.quote(rel)} 2>/dev/null | head -n {MAX_HITS}"
        )
        proc = self._exec(cmd, timeout=300)
        hits = [_GREP_PREFIX.sub("", ln) for ln in _text(proc.stdout).splitlines() if ln.strip()]
        return ToolResult(SEARCH, "\n".join(hits) if hits else "no matches", fields)

    def edit_file(self, path: str, old: str, new: str) -> ToolResult:
        try:
            rel = self._safe(path)
        except ValueError as exc:
            return ToolResult(EDIT, str(exc), {"failed": True, "path": path})
        before = self._cat(rel)
        if before is None:
            return ToolResult(EDIT, f"no such file: {path}", {"failed": True, "path": rel})
        found = before.count(old)
        if not old or found != 1:
            why = "not found" if found == 0 else f"found {found} times, must be unique"
            return ToolResult(EDIT, f"edit rejected: {why}", {"failed": True, "path": rel})
        after = before.replace(old, new, 1)
        proc = self._exec(f"cat > {shlex.quote(rel)}", stdin=after.encode("utf-8"), timeout=120)
        if proc.returncode != 0:
            return ToolResult(EDIT, f"write failed: {_text(proc.stderr)[:200]}",
                              {"failed": True, "path": rel})
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}",
            )
        )
        return ToolResult(EDIT, f"edited {rel}", {"path": rel, "post_text": after, "diff": diff})

    def run_tests(self, target: str | list[str] = "") -> ToolResult:
        """`target` may be a list, so a regression suite runs in one invocation."""
        targets = [t for t in ([target] if isinstance(target, str) else list(target)) if t] \
            or list(self.default_target)
        parts = [shlex.quote(c) for c in self.test_cmd] + [shlex.quote(t) for t in targets]
        proc = self._exec(" ".join(parts))
        out = (_text(proc.stdout) + _text(proc.stderr)).strip()
        label = " ".join(targets) if targets else " ".join(self.test_cmd)
        return ToolResult(TEST, out, {"test": label, "passed": proc.returncode == 0})

    # -- checkpoint --

    def commit(self, image: str) -> str:
        """Freeze the repo state so every arm can start from the same bytes."""
        proc = _run(["docker", "commit", self.container, image], timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(f"could not commit {self.container}: {_text(proc.stderr)[:300]}")
        return image

    def remove(self) -> None:
        _run(["docker", "rm", "-f", self.container], timeout=180)

    def __enter__(self) -> "DockerWorkspace":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def apply_test_patch(ws: DockerWorkspace, test_patch: str) -> None:
    """Add the task's tests before the agent starts.

    The target test lives in the test patch, so without this there is no
    failing test for the agent to hit and the checkpoint never fires. It does
    show the agent the expected behaviour. Every arm gets the identical setup,
    so the comparison holds; only absolute pass rates are affected, and we make
    no claim about those. State it in the method section.
    """
    proc = ws._exec("git apply -v -", stdin=test_patch.encode("utf-8"), timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"test patch did not apply: {_text(proc.stderr)[:400]}")


def attach_counters(ws: "DockerWorkspace") -> bool:
    """Point the workspace at its container's cgroup, if the host has one.

    Silent no-op off Linux, which is how the same code runs locally for
    development and on the trace VM for collection.
    """
    try:
        from ktrace import Counters
    except ImportError:
        return False
    counters = Counters(ws.container)
    ws.counters = counters if counters.available else None
    return ws.counters is not None


def default_target(task: dict) -> list[str]:
    """The task's failing tests, in the form its runner accepts."""
    from runner import as_list, test_target

    return [test_target(t) for t in as_list(task.get("FAIL_TO_PASS")) if t]


def start_task(task: dict, *, arm: str = "C0", auto_pull: bool = True) -> DockerWorkspace:
    """Container for one task, tests applied, ready for the agent."""
    from tasks import image_for

    image = image_for(task["instance_id"])
    if not have_image(image):
        if not auto_pull:
            raise RuntimeError(f"{image} is not present and auto_pull is off")
        pull(image)
    name = f"cfa-{task['instance_id']}-{arm}".lower().replace("/", "-")
    start_container(image, name)
    ws = DockerWorkspace(name, task["repo"], default_target=default_target(task))
    attach_counters(ws)
    apply_test_patch(ws, task["test_patch"])
    return ws


def from_checkpoint(image: str, repo: str, *, arm: str,
                    targets: list[str] | None = None) -> DockerWorkspace:
    """A fresh container from a committed checkpoint, for one arm."""
    name = f"cfa-resume-{arm}-{abs(hash(image)) % 10**8}".lower()
    start_container(image, name)
    ws = DockerWorkspace(name, repo, default_target=targets)
    attach_counters(ws)
    return ws


def _selfcheck() -> None:
    """Exercises every tool against a real container. No model calls."""
    from tasks import get_task, image_for

    instance = "django__django-11138"
    image = image_for(instance)

    if not docker_ok():
        print("skipped: docker is not running")
        return
    if not have_image(image):
        print(f"skipped: {image} is not pulled yet")
        return

    task = get_task(instance)
    ws = start_task(task, arm="selfcheck")
    try:
        # the guard holds, and a crafted path cannot reach the shell
        for bad in ("../etc/passwd", "/etc/passwd", "a/../../x", ""):
            assert ws.read_file(bad).fields["lines"] is None, bad
        assert ws.hash_file("../etc/passwd") is None

        # The login shell matters. With `bash -c` you get conda base, Python
        # 3.11, and the task's dependencies are missing. Python 3.6 also prints
        # its version to stderr, so both streams have to be read.
        probe = ws._exec("python --version")
        ver = (_text(probe.stdout) + _text(probe.stderr)).strip()
        assert ver.startswith("Python 3.6"), f"wrong interpreter: {ver!r}"
        where = _text(ws._exec("command -v python").stdout).strip()
        assert "envs/testbed" in where, f"conda env not active: {where!r}"

        target = "django/db/backends/sqlite3/operations.py"
        got = ws.read_file(target, 1, 20)
        assert "import" in got.output, got.output[:200]
        assert got.fields["lines"] == (1, 20), got.fields

        h1 = ws.hash_file(target)
        assert h1 and len(h1) == 16, h1

        hits = ws.search("def convert_datetimefield_value", scope="django/db/backends")
        assert ".py:" in hits.output, hits.output[:300]
        assert not hits.output.startswith("./"), hits.output[:80]

        # a unique edit lands and the content hash moves with it
        first = hits.output.splitlines()[0].split(":")[0]
        body = ws.read_file(first, 1, 10**6).output  # whole file, so uniqueness is real
        h_before = ws.hash_file(first)
        marker = "  # cfa-selfcheck-marker"
        unique = next(ln for ln in body.splitlines() if ln.strip() and body.count(ln) == 1)
        edit = ws.edit_file(first, unique, unique + marker)
        assert not edit.fields.get("failed"), edit.output
        assert edit.fields["diff"], "no diff recorded"
        assert marker in edit.fields["post_text"], "post_text does not hold the edit"
        h_after = ws.hash_file(first)
        assert h_after and h_after != h_before, (h_before, h_after)

        # a non-unique or missing edit is refused rather than guessed at
        assert ws.edit_file(first, "", "x").fields.get("failed")
        assert ws.edit_file("nope.py", "a", "b").fields.get("failed")

        # edit then revert puts the bytes back. This is the case the redundant
        # read rule has to get right: the hash matches again, yet re-reading is
        # fair because the agent changed the file in between.
        ws._exec(f"git checkout -- {shlex.quote(first)}")
        assert ws.hash_file(first) == h_before, "revert did not restore the bytes"

        # the real target test fails, and the output carries the identifiers
        from facts import failure_ids

        test_id = "timezones.tests.NewDatabaseTests.test_query_convert_timezones"
        res = ws.run_tests(test_id)
        assert res.fields["passed"] is False, res.output[-400:]
        ids = failure_ids(res.output)
        assert any(i.endswith(".py") for i in ids), ids
        assert any("Error" in i or "assert" in i for i in ids), ids

        # the checkpoint image gives every arm the same bytes to start from
        ws._exec("echo cfa-checkpoint-sentinel > .cfa-sentinel")
        ckpt = ws.commit(f"cfa-ckpt/{instance}:selfcheck".lower())
        with from_checkpoint(ckpt, task["repo"], arm="verify") as arm2:
            assert arm2.hash_file(first) == ws.hash_file(first), "checkpoint bytes differ"
            assert "cfa-checkpoint-sentinel" in (arm2._cat(".cfa-sentinel") or ""), \
                "the checkpoint did not carry the working state"
            # the applied test patch has to be there too, or the arm has no failing test
            dirty = _text(arm2._exec("git status --porcelain").stdout)
            assert "tests/timezones/tests.py" in dirty, f"test patch missing: {dirty!r}"
        _run(["docker", "image", "rm", "-f", ckpt], timeout=300)
    finally:
        ws.remove()

    print(f"selfcheck ok: tools, guard, failing test and checkpoint all work in {instance}")
    print(f"  {len(ids)} identifiers from the real failure: {ids[:2]}")


if __name__ == "__main__":
    _selfcheck()
