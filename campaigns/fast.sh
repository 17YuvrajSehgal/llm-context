#!/bin/bash
# Behavioural dataset, collected in parallel.
#
# No kernel tracing here, and that is what makes parallelism safe. Tokens,
# tool calls and the redundancy rules do not care how busy the machine is; the
# agent sees identical tool output either way. Only the timing columns would be
# contaminated by contention, and we already have 61 clean serial runs with
# full traces for anything timing- or kernel-related.
#
# Each job writes into its own directory so concurrent appends cannot interleave
# in one results file. They are concatenated at the end.
cd ~/llm-context
JOBS=${JOBS:-6}
mapfile -t TASKS < <(.venv/bin/python -c "from tasks import pool_ids; print('\n'.join(pool_ids()))")
echo "$(date -u +%H:%M:%S)  ${#TASKS[@]} tasks, $JOBS at a time"

run_one() {
  t=$1
  out=runs-fast/$t
  .venv/bin/python runner.py --only "$t" --facts 1 --trials 3 --c0-trials 1 \
      --max-steps 24 --resume-steps 14 --model sonnet --outdir "$out" \
      > "runs-fast/$t.log" 2>&1
  docker image rm -f "swebench/sweb.eval.x86_64.$(echo "$t" | sed 's/__/_1776_/')":latest >/dev/null 2>&1
  n=$(wc -l < "$out/results.jsonl" 2>/dev/null || echo 0)
  echo "$(date -u +%H:%M:%S)  done $t  rows=$n  disk=$(df -h / | tail -1 | awk '{print $4}')"
}
export -f run_one

mkdir -p runs-fast
printf '%s\n' "${TASKS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one {}'

cat runs-fast/*/results.jsonl > runs-fast/results.jsonl 2>/dev/null
echo "$(date -u +%H:%M:%S)  DONE  total rows: $(wc -l < runs-fast/results.jsonl)"
