#!/bin/bash
# The behavioural campaign. Parallel and untraced: tokens, tool calls and the
# redundancy rules do not care how busy the machine is, and the timing columns
# are not read from this set. Every row records nproc and parallel, and
# collect.py refuses to merge rows from different setups.
cd ~/llm-context
JOBS=${JOBS:-6}
OUT=${OUT:-runs-main}
mkdir -p "$OUT"
mapfile -t TASKS < <(.venv/bin/python -c "from tasks import pool_ids; print('\n'.join(pool_ids()))")
echo "$(date -u +%H:%M:%S)  ${#TASKS[@]} tasks, $JOBS at a time, into $OUT"

run_one() {
  t=$1
  .venv/bin/python runner.py --only "$t" --facts 1 --trials 3 --c0-trials 1 \
      --max-steps 24 --resume-steps 14 --model sonnet \
      --parallel "$JOBS" --outdir "$OUT/$t" > "$OUT/$t.log" 2>&1
  docker image rm -f "swebench/sweb.eval.x86_64.$(echo "$t" | sed 's/__/_1776_/')":latest >/dev/null 2>&1
  n=0; [ -f "$OUT/$t/results.jsonl" ] && n=$(wc -l < "$OUT/$t/results.jsonl")
  echo "$(date -u +%H:%M:%S)  $t  rows=$n  free=$(df -h / | tail -1 | awk '{print $4}')"
}
export -f run_one; export JOBS OUT

printf '%s\n' "${TASKS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one {}'
echo "$(date -u +%H:%M:%S)  runs finished, merging"
.venv/bin/python collect.py "$OUT"
