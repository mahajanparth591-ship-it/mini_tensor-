#!/usr/bin/env bash
# One command for the whole Common Test Specification submission:
#   bash tools/run_all.sh "<Team_ID>"
# Produces in ./submission/ : functional_results.csv, perf_results.csv(+_detail), reference_results.csv,
#                             results.csv (mandatory columns), summary.md, machine_info.txt
set -euo pipefail
TEAM="${1:-TEAM}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="$ROOT/submission"; mkdir -p "$OUT"; cd "$ROOT"
FLAGS="-std=c++17 -O3 -DNDEBUG -march=native -fopenmp -Iinclude"

{ echo "date: $(date -u)"; echo "commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "compiler: $(g++ --version | head -1)"; echo "flags: $FLAGS"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|MHz" || true; grep MemTotal /proc/meminfo || true
  echo "os: $(. /etc/os-release; echo $PRETTY_NAME)"; echo "kernel: $(uname -r)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unavailable)"
  python3 -c "import numpy; print('numpy:', numpy.__version__)"; } > "$OUT/machine_info.txt"

echo "== 1/5 functional tests (TC01-TC50) =="
g++ $FLAGS tests/common_test.cpp -o "$OUT/common_test"
"$OUT/common_test" "$TEAM" "$OUT/functional_results.csv" || true

echo "== 2/5 sanitizer run (TC46-TC50 with ASan+UBSan) =="
g++ -std=c++17 -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -fopenmp -Iinclude tests/common_test.cpp -o "$OUT/common_test_asan"
"$OUT/common_test_asan" "$TEAM" "$OUT/functional_results_asan.csv" > "$OUT/sanitizer_log.txt" 2>&1 || true
tail -3 "$OUT/sanitizer_log.txt"; echo "sanitizer error lines: $(grep -ci "ERROR: \(Address\|Leak\)Sanitizer\|runtime error" "$OUT/sanitizer_log.txt" || true)"

echo "== 3/5 student performance benchmarks =="
g++ $FLAGS benchmarks/spec_bench.cpp -o "$OUT/spec_bench"
"$OUT/spec_bench" "$TEAM" "$OUT/perf_results.csv" || true

echo "== 4/5 NumPy reference (single thread) =="
python3 benchmarks/reference_numpy.py "$OUT/perf_results_pb05_indices.txt" "$OUT/reference_results.csv"

echo "== 5/5 merge =="
python3 tools/merge_results.py "$OUT/functional_results.csv" "$OUT/perf_results.csv" "$OUT/reference_results.csv" "$OUT/results.csv" "$OUT/summary.md"
echo "Done. Submit the files in $OUT (plus the test driver sources)."
