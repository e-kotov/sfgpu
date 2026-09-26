#!/bin/bash
# A source-only mutant proves the independent oracle consumes shader results.
set -euo pipefail
mutation_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sfgpu-metal-mutation.XXXXXX")
trap 'rm -rf "$mutation_dir"' EXIT
mkdir -p "$mutation_dir/source" "$mutation_dir/library"
git archive HEAD | tar -x -C "$mutation_dir/source"
python3 - "$mutation_dir/source/src/metal_kernel.h" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = 'out[ulong(gid.y) * ulong(nx) + ulong(gid.x)] = result;'
assert s.count(needle) == 1, 'shader mutation location changed'
p.write_text(s.replace(needle, 'out[ulong(gid.y) * ulong(nx) + ulong(gid.x)] = 0UL;'))
PY
if ! R CMD INSTALL --configure-args=--enable-metal --library="$mutation_dir/library" \
    "$mutation_dir/source" > "$mutation_dir/install.log" 2>&1; then
  cat "$mutation_dir/install.log"
  exit 1
fi
if R_LIBS="$mutation_dir/library:${R_LIBS:-${R_LIBS_USER:-}}" \
    Rscript tools/validate-metal.R > "$mutation_dir/validation.log" 2>&1; then
  cat "$mutation_dir/validation.log"
  echo 'ERROR: corrupted Metal shader passed validation'
  exit 1
fi
cat "$mutation_dir/validation.log"
if ! grep -q 'METAL_NUMERIC_MISMATCH' "$mutation_dir/validation.log"; then
  echo 'ERROR: mutant failed for a reason other than the expected numerical mismatch'
  exit 1
fi
printf '%s\n' 'METAL_MUTATION_REJECTED: zero-output shader detected by independent oracle'
