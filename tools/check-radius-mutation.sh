#!/bin/bash
# Both missing matches and a filter that merely passes everything must fail.
set -euo pipefail
mutation_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sfgpu-radius-mutation.XXXXXX")
trap 'rm -rf "$mutation_dir"' EXIT
for mode in zero one; do
  mkdir -p "$mutation_dir/$mode/source" "$mutation_dir/$mode/library"
  git archive HEAD | tar -x -C "$mutation_dir/$mode/source"
  python3 - "$mutation_dir/$mode/source/src/metal_radius_kernel.h" "$mode" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = '(p.x >= b.xmin && p.x <= b.xmax && p.y >= b.ymin && p.y <= b.ymax)'
assert s.count(needle) == 1, 'radius mutation location changed'
p.write_text(s.replace(needle, '0' if sys.argv[2] == 'zero' else '1'))
PY
  if ! R CMD INSTALL --configure-args=--enable-metal --library="$mutation_dir/$mode/library" \
      "$mutation_dir/$mode/source" > "$mutation_dir/$mode/install.log" 2>&1; then
    cat "$mutation_dir/$mode/install.log"
    exit 1
  fi
  if R_LIBS="$mutation_dir/$mode/library:${R_LIBS:-${R_LIBS_USER:-}}" \
      Rscript tools/validate-radius.R > "$mutation_dir/$mode/validation.log" 2>&1; then
    cat "$mutation_dir/$mode/validation.log"
    echo "ERROR: corrupted radius filter passed: $mode"
    exit 1
  fi
  cat "$mutation_dir/$mode/validation.log"
  expected=RADIUS_MEMBERSHIP_MISMATCH
  if [[ "$mode" == one ]]; then expected=RADIUS_FILTER_MISMATCH; fi
  if ! grep -q "$expected" "$mutation_dir/$mode/validation.log"; then
    echo "ERROR: mutant failed for an unexpected reason: $mode"
    exit 1
  fi
  echo "RADIUS_MUTATION_REJECTED=$mode"
done
