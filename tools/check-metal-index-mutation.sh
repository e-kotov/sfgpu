#!/bin/bash
# Require specific failures from separately compiled, deliberately broken kernels.
set -euo pipefail
mutation_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sfgpu-metal-index-mutation.XXXXXX")
trap 'rm -rf "$mutation_dir"' EXIT
for mode in wrong_ids all_candidates; do
  mkdir -p "$mutation_dir/$mode/source" "$mutation_dir/$mode/library"
  git archive HEAD | tar -x -C "$mutation_dir/$mode/source"
  python3 - "$mutation_dir/$mode/source/src/metal_radius_kernel.h" "$mode" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
if sys.argv[2] == 'wrong_ids':
    needle = 'output[slot] = {task.local_i, point.original_j};'
    replacement = 'output[slot] = {task.local_i, task.first + k};'
else:
    needle = 'if (point.y >= b.ymin && point.y <= b.ymax)'
    replacement = 'if (true)'
assert s.count(needle) == 1, 'Metal index mutation location changed'
p.write_text(s.replace(needle, replacement))
PY
  if ! R CMD INSTALL --configure-args=--enable-metal --library="$mutation_dir/$mode/library" \
      "$mutation_dir/$mode/source" > "$mutation_dir/$mode/install.log" 2>&1; then
    cat "$mutation_dir/$mode/install.log"
    exit 1
  fi
  if SFGPU_RADIUS_BACKEND=metal R_LIBS="$mutation_dir/$mode/library:${R_LIBS:-${R_LIBS_USER:-}}" \
      Rscript tools/validate-radius-index.R > "$mutation_dir/$mode/validation.log" 2>&1; then
    cat "$mutation_dir/$mode/validation.log"
    echo "ERROR: corrupted Metal index passed: $mode"
    exit 1
  fi
  cat "$mutation_dir/$mode/validation.log"
  expected=RADIUS_INDEX_MISMATCH
  if [[ "$mode" == all_candidates ]]; then expected=RADIUS_INDEX_FILTER_MISMATCH; fi
  if ! grep -q "$expected" "$mutation_dir/$mode/validation.log"; then
    echo "ERROR: index mutant failed for an unexpected reason: $mode"
    exit 1
  fi
  echo "METAL_INDEX_MUTATION_REJECTED=$mode"
done
