#!/bin/bash
# Full-trace collection with the fixed arm construction.
#
# Serial on purpose: concurrent containers would compete for the same 4 CPUs
# and contaminate each other's kernel counters, which is the measurement we
# came for.
#
# Known-good tasks first, so if this is stopped early the data already covers
# several projects. Each image is 4 GB and the disk is shared with the
# os-noise project, so it is deleted once its runs are done.
cd ~/llm-context
TASKS="django__django-11138 pylint-dev__pylint-4551 astropy__astropy-13398
       sphinx-doc__sphinx-9461 sympy__sympy-14248 pylint-dev__pylint-8898
       astropy__astropy-14369 django__django-15629 sympy__sympy-16597
       django__django-16263 sphinx-doc__sphinx-7590 pydata__xarray-6992"
for t in $TASKS; do
  echo "=========== $t  $(date -u +%H:%M:%S) ==========="
  .venv/bin/python runner.py --only "$t" --facts 1 --trials 3 \
      --max-steps 24 --resume-steps 14 --model sonnet \
      --ktrace --outdir runs-vm 2>&1
  docker image rm -f "swebench/sweb.eval.x86_64.$(echo "$t" | sed 's/__/_1776_/')":latest >/dev/null 2>&1
  echo "disk: $(df -h / | tail -1 | awk '{print $4}') free | rows: $(wc -l < runs-vm/results.jsonl 2>/dev/null || echo 0)"
done
echo "=========== CAMPAIGN DONE $(date -u +%H:%M:%S) ==========="
