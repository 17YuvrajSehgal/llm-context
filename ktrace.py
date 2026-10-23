"""Kernel-level cost of each tool call, for the runs on the trace VM.

Two layers, because they answer different questions.

Layer 1, cgroup v2 counters. The kernel already accounts CPU time, disk bytes
and page faults per container. Reading the counters either side of a tool call
gives that call's exact cost with no sampling and almost no overhead. This is
the layer the numbers come from.

Layer 2, an LTTng kernel trace for the whole run. Syscall-level detail for the
cases where the counters say something happened but not what. LTTng is what the
lab's VM already has set up, so we use it rather than introducing bpftrace.
Sessions are named with a distinct prefix so they cannot collide with the other
experiment on that machine.

The two join to our own trace by time. Every tool call already logs a timestamp
and its duration, so a kernel event belongs to the call whose window contains
it.

Run `python ktrace.py` for the self-check. The parsers are checked against real
file contents offline; the live parts are skipped when not on Linux.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
SESSION_PREFIX = "cfa"  # keeps us clear of the other experiment's sessions

# Everything the kernel offers: 431 syscalls and 233 tracepoints on this VM.
# Collecting the lot costs disk, but a narrower set would mean re-running the
# whole campaign for the power and energy questions later. power_cpu_frequency
# and power_cpu_idle in particular are what energy modelling needs, and they
# are only in the trace if they were enabled at collection time.
ENABLE_ALL = True

# Ring buffers have to grow with the event rate or the kernel drops records
# silently. 8 MiB x 8 per CPU is 64 MiB per CPU, 256 MiB on 4 CPUs, against
# 15 GiB of RAM. The discarded count is recorded either way, so a run that
# dropped events can be thrown out rather than quietly believed.
CHANNEL = "channel0"
SUBBUF_SIZE = "8M"
NUM_SUBBUF = "8"

# Kept only as the fallback when something refuses `--all`.
SYSCALLS = [
    "openat", "read", "pread64", "close", "write",
    "execve", "clone", "newfstatat", "statx",
]


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def ns_inodes(pid: int) -> dict:
    """Namespace inode numbers for a pid.

    These are the exact values LTTng puts in its `pid_ns`, `mnt_ns` and
    `cgroup_ns` context fields, so recording them for our containers and for
    our own process is what lets the analysis say which events are agent work.
    """
    names = ("pid", "mnt", "cgroup", "net", "ipc", "user")
    out = {}
    for ns in names:
        try:
            link = os.readlink(f"/proc/{pid}/ns/{ns}")
        except OSError:
            continue
        if "[" in link:
            out[f"{ns}_ns"] = int(link.split("[", 1)[1].rstrip("]"))
    if out:
        return out
    # A container's init process is owned by root, so an unprivileged readlink
    # on its /proc/<pid>/ns/* fails and the container would end up with no
    # namespace ids at all. That is the attribution silently going missing, so
    # retry through sudo rather than return an empty map.
    paths = " ".join(f"/proc/{pid}/ns/{n}" for n in names)
    res = _run(["sudo", "sh", "-c", f"readlink {paths} 2>/dev/null"])
    for ns, link in zip(names, res.stdout.splitlines()):
        if "[" in link:
            out[f"{ns}_ns"] = int(link.split("[", 1)[1].rstrip("]"))
    return out


def set_comm(name: str) -> bool:
    """Rename our own process so it is not another anonymous `python3`.

    The os-noise runs hit this: the shebang makes comm `python3`, which
    collides with the agent's own python3 and with the container's test
    processes. A distinct name makes host-side harness work obvious in the
    trace without needing to resolve pids.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        buf = ctypes.create_string_buffer(name.encode()[:15])
        return libc.prctl(15, buf, 0, 0, 0) == 0  # 15 = PR_SET_NAME
    except (OSError, AttributeError):
        return False


def attribution(container: str | None = None, *, extra: dict | None = None) -> dict:
    """Everything needed to tell agent work from system noise in the trace.

    Written next to each arm's trace. The kernel trace carries pids and
    namespace inodes; this file says which of those belong to the container
    doing the agent's work, and which belong to our harness. Joined with the
    tool-call log, which already has a timestamp and a duration per call, every
    kernel event can be placed either inside a specific tool call or outside
    all of them.
    """
    import platform
    import socket

    me = os.getpid()
    out = {
        "harness": {
            "pid": me, "comm": "cfa-harness",
            "ns": ns_inodes(me),
        },
        "host": {
            "hostname": socket.gethostname(),
            "kernel": platform.release(),
            "nproc": os.cpu_count(),
        },
        "clock": anchor("attribution"),
    }
    if container:
        probe = _run(["docker", "inspect", "--format",
                      "{{.Id}} {{.State.Pid}}", container])
        if probe.returncode == 0 and probe.stdout.split():
            parts = probe.stdout.split()
            cid, init_pid = parts[0], int(parts[1])
            cg = find_cgroup(container)
            out["container"] = {
                "name": container, "id": cid, "init_pid": init_pid,
                "cgroup": str(cg) if cg else None,
                "ns": ns_inodes(init_pid),
            }
    if extra:
        out.update(extra)
    return out


def anchor(label: str) -> dict:
    """All three clocks at one instant, so the traces can be lined up.

    LTTng timestamps are monotonic or boottime; our tool-call log is realtime.
    Without a pair of anchors there is no way to convert between them and the
    kernel events cannot be attributed to a tool call. The os-noise collector
    writes these at the start and end of every run, and so do we.
    """
    import time as _t

    # clock_gettime is Linux-only. Collection always runs on the VM, but the
    # same code is developed and self-checked on Windows, so the three clocks
    # degrade to what that platform has rather than raising.
    if not hasattr(_t, "clock_gettime"):
        return {"label": label, "realtime": _t.time(),
                "monotonic": _t.monotonic(), "boottime": None}
    return {
        "label": label,
        "realtime": _t.clock_gettime(_t.CLOCK_REALTIME),
        "monotonic": _t.clock_gettime(_t.CLOCK_MONOTONIC),
        "boottime": _t.clock_gettime(_t.CLOCK_BOOTTIME),
    }


# Namespace context is not optional. The work runs inside containers, and our
# harness is python3 while the container's tests are also python3, so procname
# alone cannot tell them apart. The namespace fields can.
CONTEXTS = ["pid", "tid", "procname"]
NS_CONTEXTS = ["cgroup_ns", "pid_ns", "mnt_ns", "net_ns", "ipc_ns", "user_ns"]


def parse_cpu_stat(text: str) -> dict:
    """cgroup v2 cpu.stat. Microseconds of CPU, split user and system."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("usage_usec", "user_usec", "system_usec"):
            out[parts[0]] = int(parts[1])
    return out


def parse_io_stat(text: str) -> dict:
    """cgroup v2 io.stat. One line per device, so the devices are summed.

    These are bytes that actually reached the block layer, so a read served
    from the page cache does not count. That is the point: it separates work
    the machine really did from work it got for free.
    """
    totals = {"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}
    for line in text.splitlines():
        for key, value in re.findall(r"(\w+)=(\d+)", line):
            if key in totals:
                totals[key] += int(value)
    return totals


def parse_memory_stat(text: str) -> dict:
    """cgroup v2 memory.stat. Page faults, major ones meaning a disk hit."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("pgfault", "pgmajfault"):
            out[parts[0]] = int(parts[1])
    return out


def find_cgroup(container: str) -> Path | None:
    """The cgroup directory for a running container.

    Docker lays this out differently depending on the cgroup driver, so the
    known layouts are tried in turn rather than assumed.
    """
    probe = _run(["docker", "inspect", "--format", "{{.Id}}", container])
    if probe.returncode != 0:
        return None
    cid = probe.stdout.strip()
    if not cid:
        return None
    candidates = [
        CGROUP_ROOT / "system.slice" / f"docker-{cid}.scope",
        CGROUP_ROOT / "docker" / cid,
        CGROUP_ROOT / "system.slice" / "docker.service" / f"docker-{cid}.scope",
    ]
    for path in candidates:
        if (path / "cpu.stat").exists():
            return path
    # last resort: the container's own view of where it lives
    hits = list(CGROUP_ROOT.glob(f"**/*{cid[:16]}*"))
    return next((h for h in hits if (h / "cpu.stat").exists()), None)


class Counters:
    """Reads one container's kernel counters. `delta` is what a tool call cost."""

    FIELDS = ("usage_usec", "user_usec", "system_usec",
              "rbytes", "wbytes", "rios", "wios", "pgfault", "pgmajfault")

    def __init__(self, container: str):
        self.container = container
        self.path = find_cgroup(container)

    @property
    def available(self) -> bool:
        return self.path is not None

    def read(self) -> dict:
        if self.path is None:
            return {}
        out = {}
        for name, parser in (("cpu.stat", parse_cpu_stat),
                             ("io.stat", parse_io_stat),
                             ("memory.stat", parse_memory_stat)):
            try:
                out.update(parser((self.path / name).read_text()))
            except OSError:
                continue
        return out

    @staticmethod
    def delta(before: dict, after: dict) -> dict:
        """What changed between two reads. Missing on either side means skip.

        Counters are monotonic, so a negative difference means the container
        was replaced underneath us and the value is dropped rather than logged
        as a nonsense number.
        """
        out = {}
        for key in Counters.FIELDS:
            if key not in before or key not in after:
                continue
            diff = after[key] - before[key]
            if diff >= 0:
                out[f"k_{key}"] = diff
        return out


class LttngSession:
    """An LTTng kernel trace around one agent run.

    Needs root, so every command goes through sudo. Nothing here touches
    sessions it did not create.
    """

    def __init__(self, name: str, outdir: str | Path, *, syscalls: list[str] | None = None):
        self.name = f"{SESSION_PREFIX}-{name}"[:60]
        self.outdir = Path(outdir)
        self.syscalls = syscalls or SYSCALLS
        self.started = False
        self.error: str | None = None
        self.anchors: list[dict] = []

    @staticmethod
    def available() -> bool:
        return bool(shutil.which("lttng"))

    def _lttng(self, *args: str) -> subprocess.CompletedProcess:
        return _run(["sudo", "lttng", *args])

    def start(self) -> bool:
        if not self.available():
            self.error = "lttng is not installed"
            return False
        self.outdir.mkdir(parents=True, exist_ok=True)
        # clear only our own leftover session, never anyone else's
        self._lttng("destroy", self.name)
        created = self._lttng("create", self.name, f"--output={self.outdir}")
        if created.returncode != 0:
            self.error = (created.stderr or created.stdout).strip()[:300]
            return False
        # the channel has to exist before the events, or it is created with
        # default buffers too small for this event rate
        self._lttng("enable-channel", "-k", f"--subbuf-size={SUBBUF_SIZE}",
                    f"--num-subbuf={NUM_SUBBUF}", CHANNEL)
        if ENABLE_ALL:
            tp = self._lttng("enable-event", "-c", CHANNEL, "-k", "--all")
            sc = self._lttng("enable-event", "-c", CHANNEL, "-k", "--syscall", "--all")
            if tp.returncode != 0 and sc.returncode != 0:
                self.error = (tp.stderr or tp.stdout).strip()[:300]
                self._lttng("destroy", self.name)
                return False
        else:
            enabled = self._lttng("enable-event", "-c", CHANNEL, "-k",
                                  "--syscall", ",".join(self.syscalls))
            if enabled.returncode != 0:
                self.error = (enabled.stderr or enabled.stdout).strip()[:300]
                self._lttng("destroy", self.name)
                return False
            # block_* shows real disk I/O, mm_vmscan_* shows reclaim. The prefix
            # is mm_vmscan, not vmscan: the wrong wildcard records nothing.
            for group in ("sched_*", "block_*", "mm_vmscan_*", "writeback_*"):
                self._lttng("enable-event", "-c", CHANNEL, "-k", group)
        self._lttng("add-context", "-k", *[a for c in CONTEXTS for a in ("-t", c)])
        for ctx in NS_CONTEXTS:
            self._lttng("add-context", "-k", "-t", ctx)
        manifest = self._lttng("list", self.name)
        (self.outdir / "lttng_enabled.txt").write_text(
            manifest.stdout or manifest.stderr, encoding="utf-8")
        self.anchors = [anchor("start")]
        if self._lttng("start").returncode != 0:
            self.error = "could not start the session"
            self._lttng("destroy", self.name)
            return False
        self.started = True
        return True

    def stop(self) -> dict:
        if not self.started:
            return {"traced": False, "error": self.error}
        self.anchors.append(anchor("end"))
        self._lttng("stop")
        # read the discarded count before destroying, or the session is gone
        # and we cannot tell whether the trace lost records
        listing = self._lttng("list", self.name)
        discarded = None
        for line in (listing.stdout or "").splitlines():
            if "iscarded event" in line:
                digits = re.findall(r"\d+", line)
                if digits:
                    discarded = int(digits[-1])
                break
        self._lttng("destroy", self.name)
        self.started = False
        # lttng's consumer writes the trace as root; hand it back or the
        # analysis cannot read it
        _run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(self.outdir)])
        (self.outdir / "clock_anchors.jsonl").write_text(
            "\n".join(json.dumps(a) for a in self.anchors) + "\n", encoding="utf-8")
        size = sum(f.stat().st_size for f in self.outdir.rglob("*") if f.is_file())
        return {"traced": True, "trace_dir": str(self.outdir), "trace_bytes": size,
                "discarded": discarded, "anchors": self.anchors}

    def __enter__(self) -> "LttngSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def compress(trace_dir: str | Path, *, remove: bool = True) -> dict:
    """Pack a finished trace. Everything-enabled traces are large and
    compress well, and the VM shares its disk with another project."""
    d = Path(trace_dir)
    if not d.is_dir():
        return {"compressed": False}
    before = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    archive = d.with_suffix(".tar.gz")
    res = _run(["tar", "czf", str(archive), "-C", str(d.parent), d.name], timeout=1800)
    if res.returncode != 0:
        return {"compressed": False, "error": (res.stderr or "")[:200]}
    after = archive.stat().st_size
    if remove:
        shutil.rmtree(d, ignore_errors=True)
    return {"compressed": True, "archive": str(archive),
            "bytes_before": before, "bytes_after": after,
            "ratio": round(before / after, 1) if after else None}


def drop_caches() -> bool:
    """Start every arm with a cold page cache.

    Arms run one after another from the same image, so the second arm would
    find the repo already cached and its disk-read counters would look lower
    for no reason but running order. The os-noise collector hit the same thing
    and drops caches before its measured runs.
    """
    res = _run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    return res.returncode == 0


def _selfcheck() -> None:
    """Parsers against real file contents, then the live parts if available."""
    cpu = parse_cpu_stat(
        "usage_usec 1234567\nuser_usec 1000000\nsystem_usec 234567\n"
        "nr_periods 0\nnr_throttled 0\nthrottled_usec 0\n"
    )
    assert cpu == {"usage_usec": 1234567, "user_usec": 1000000, "system_usec": 234567}, cpu

    # two devices, so the totals must add up rather than take the last line
    io = parse_io_stat(
        "8:0 rbytes=4096 wbytes=8192 rios=2 wios=4 dbytes=0 dios=0\n"
        "8:16 rbytes=1024 wbytes=0 rios=1 wios=0 dbytes=0 dios=0\n"
    )
    assert io == {"rbytes": 5120, "wbytes": 8192, "rios": 3, "wios": 4}, io
    assert parse_io_stat("") == {"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}

    mem = parse_memory_stat("anon 100\nfile 200\npgfault 5000\npgmajfault 12\nslab 3\n")
    assert mem == {"pgfault": 5000, "pgmajfault": 12}, mem

    before = {"usage_usec": 100, "rbytes": 0, "pgfault": 10, "wbytes": 5}
    after = {"usage_usec": 350, "rbytes": 4096, "pgfault": 25, "wbytes": 5}
    d = Counters.delta(before, after)
    assert d == {"k_usage_usec": 250, "k_rbytes": 4096, "k_wbytes": 0, "k_pgfault": 15}, d

    # a counter that went backwards means the container was replaced: drop it
    assert "k_usage_usec" not in Counters.delta({"usage_usec": 500}, {"usage_usec": 100})
    # a field missing on either side is skipped, not guessed
    assert Counters.delta({"usage_usec": 1}, {}) == {}
    assert Counters.delta({}, {"usage_usec": 1}) == {}

    assert LttngSession("x", "/tmp/x").name.startswith("cfa-"), "sessions must be namespaced"

    # attribution has to carry our own pid and namespaces, or nothing in the
    # kernel trace can be tied back to agent work
    att = attribution()
    assert att["harness"]["pid"] == os.getpid(), att
    assert "host" in att and "clock" in att, att
    if os.name == "posix":
        assert att["harness"]["ns"], "no namespace inodes read for our own pid"
        assert set_comm("cfa-selfcheck") in (True, False)
    assert "mm_vmscan_*" in open(__file__, encoding="utf-8").read(), "reclaim prefix must be mm_vmscan"

    if hasattr(__import__("time"), "clock_gettime"):
        a = anchor("start")
        assert {"label", "realtime", "monotonic", "boottime"} <= set(a), a
        assert a["realtime"] > 1_700_000_000, "realtime looks wrong"
        assert a["monotonic"] != a["realtime"], "the clocks must be different domains"

    live = []
    if CGROUP_ROOT.exists():
        live.append(f"cgroup v2 root present at {CGROUP_ROOT}")
    if LttngSession.available():
        live.append("lttng on PATH")
    print("selfcheck ok: cpu, io and memory parsers, deltas, namespaced sessions")
    print(f"  live capability here: {', '.join(live) if live else 'none (not Linux)'}")


if __name__ == "__main__":
    _selfcheck()
