#!/bin/bash
# Re-runs the tasks the session limit killed. Same setup as the first pass, so
# the rows pool with it: 16 vCPU, 6 jobs, no kernel tracing.
cd ~/llm-context
JOBS=6; OUT=runs-main
mapfile -t TASKS < /tmp/retry.txt
echo "$(date -u +%H:%M:%S)  retrying ${#TASKS[@]} capped tasks"
run_one() {
  t=$1
  rm -rf "$OUT/$t"
  .venv/bin/python runner.py --only "$t" --facts 1 --trials 3 --c0-trials 1 \
      --max-steps 24 --resume-steps 14 --model sonnet \
      --parallel "$JOBS" --outdir "$OUT/$t" > "$OUT/$t.log" 2>&1
  docker image rm -f "swebench/sweb.eval.x86_64.$(echo "$t" | sed 's/__/_1776_/')":latest >/dev/null 2>&1
  n=0; [ -f "$OUT/$t/results.jsonl" ] && n=$(wc -l < "$OUT/$t/results.jsonl")
  cap=""; grep -q "session limit" "$OUT/$t.log" 2>/dev/null && cap=" CAP"
  echo "$(date -u +%H:%M:%S)  $t  rows=$n$cap"
}
export -f run_one; export JOBS OUT
printf '%s\n' "${TASKS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one {}'
echo "$(date -u +%H:%M:%S)  retry finished"
.venv/bin/python collect.py "$OUT"
