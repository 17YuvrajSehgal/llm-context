# Collected data, and what each set can be used for

Runs from different machines cannot be pooled. Core count changes how a
parallel test suite spends CPU, so a kernel measurement from a 4-core box does
not belong in the same table as one from a 16-core box. The run header did not
record the machine until 2026-10-23; from that point every header carries
`host`, `nproc` and `kernel`. The sets below predate that, so their provenance
is written down here instead.

All of it lives on the `agentic-noise` VM in project `teleeporter`, under
`/home/aylabsltd/llm-context/`. Results files are backed up in `vm-results/`.

## 1. `runs-broken-arms/` — 31 runs, 4 vCPU, serial, fully traced

The first traced campaign. 1.4 GB, 4 tasks.

- **Arm comparison: unusable.** The three arms were built three different ways,
  so no pairing isolates a single difference.
- **Kernel cost per tool call: usable.** Arm construction does not touch what a
  tool call costs the machine. This is the set behind the 345-call analysis in
  `kcost.py`: a test run costs 0.52x the tokens of a file read but 7.47x the
  CPU, and reads are 34% of tokens against 9% of CPU.

## 2. `runs-vm/` — 61 runs, 4 vCPU, serial, fully traced

The campaign with the arms fixed. 5 usable pairs across 5 tasks.

- **Arm comparison: valid but tiny.** No detectable fact-specific effect. The
  placebo moved further than the real cut, and the within-arm spread was
  comparable to the effect.
- **Kernel cost: usable, and the same machine as set 1**, so these two can be
  pooled for the machine-cost analysis.

## 3. `runs-main/` — 110 rows, 16 vCPU, 6 jobs in parallel, no LTTng

The campaign with the final arm construction. 11 usable pairs, 4 with a placebo.

- **Arm comparison: valid, and the only set where it is.** This is the one to
  extend. No detectable fact-specific effect yet; the placebo moved further
  than the real cut.
- **Kernel numbers: present but contended.** cgroup counters are collected on
  every run, not only traced ones, so these rows carry CPU and I/O figures.
  Six jobs were competing for the same cores, so they must not be pooled with
  the serial sets or read as clean per-call costs.

## Deleted

`runs-fast/` (33 rows) held nothing unique: pre-redesign arms, contended kernel
numbers, no LTTng, and superseded by `runs-main`. Removed locally; the copy on
the VM can go next time it is started.

## What has to be fixed before collecting again

1. Let the placebo come from any irrelevant fact whose lines actually appear in
   the chosen context, not only ones whose source block was chosen.
2. Set the context-share limit from measured traces, and record the share per
   pair so the analysis can test whether it matters rather than gating on it.
3. Decide the machine first and keep it. Mixing 4-core and 16-core runs in one
   analysis is not allowed.
4. The clean per-call kernel numbers come only from sets 1 and 2, which ran
   serially on 4 cores. If the behavioural set moves to a different machine,
   re-collect a small serial traced set there so both halves of the paper come
   from one configuration.
