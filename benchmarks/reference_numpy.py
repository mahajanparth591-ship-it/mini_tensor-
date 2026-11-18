#!/usr/bin/env python3
"""NumPy reference for the Common Test Specification benchmarks (single thread, float32).
Usage: python3 benchmarks/reference_numpy.py <pb05_indices.txt> [out_ref.csv] [--quick]
Uses the same protocol as spec_bench: 5 warm-ups, 30 timed runs, 3 independent repetitions."""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"          # single-thread baseline; must be set before importing numpy
import csv, statistics, time
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
quick = "--quick" in sys.argv
WARM, REPS, RUNS = (1, 3, 3) if quick else (5, 30, 3)
idx_file = args[0] if args else "perf_results_pb05_indices.txt"
out = args[1] if len(args) > 1 else "reference_results.csv"
F = np.float32
rows = []

def bench(cfg, timed, after=lambda: None):
    meds, allt = [], []
    for _ in range(RUNS):
        for _ in range(WARM): timed(); after()
        t = []
        for _ in range(REPS):
            a = time.perf_counter(); timed(); b = time.perf_counter()
            t.append((b - a) * 1e3); after()
        meds.append(statistics.median(t)); allt += t
    rows.append([cfg, statistics.median(meds), min(allt), statistics.mean(allt), statistics.pstdev(allt)])
    print(f"{cfg:26s} median {statistics.median(meds):10.4f} ms")

def creation(cfg, n, val):
    box = [None]
    def timed():
        a = np.empty((n, n), dtype=F); a.fill(val); box[0] = a     # empty+fill forces the init (np.zeros is lazy)
    bench(cfg, timed, lambda: box.__setitem__(0, None))

bench_cfgs = []
creation("PB01_100x100", 100, 0.0); creation("PB02_1000x1000", 1000, 1.0); creation("PB03_4000x4000", 4000, 1.0)

t = np.arange(1, 1000001, dtype=F).reshape(1000, 1000)
bench("PB04_traverse", lambda: t.sum(dtype=np.float64))

t0 = np.arange(1000000, dtype=F).reshape(1000, 1000)
if os.path.exists(idx_file):
    ij = np.loadtxt(idx_file, dtype=np.int64); r_, c_ = ij[:, 0], ij[:, 1]
    bench("PB05_random_reads", lambda: t0[r_, c_].sum(dtype=np.float64))   # vectorised fancy indexing
else:
    print("PB05 skipped: index file not found", idx_file)

for n in (100, 500, 1000, 4000):
    A, B = np.full((n, n), 1.25, F), np.full((n, n), 2.5, F); box = [None]
    bench(f"PB06_{n}x{n}_t1", lambda: box.__setitem__(0, A + B))
    A, B = np.full((n, n), 5.0, F), np.full((n, n), 1.5, F)
    bench(f"PB07_{n}x{n}_t1", lambda: box.__setitem__(0, A - B))
    A, B = np.full((n, n), 1.25, F), np.full((n, n), 2.0, F)
    bench(f"PB08_{n}x{n}_t1", lambda: box.__setitem__(0, A * B))

def mats(M, K, N):
    i = np.arange(M)[:, None]; j = np.arange(K)[None, :]
    A = (((i + j) % 7) * 0.1).astype(F)
    i = np.arange(K)[:, None]; j = np.arange(N)[None, :]
    B = (((2 * i + j) % 5) * 0.2).astype(F)
    return A, B
for cfg, (M, K, N) in {"PB09_64x64": (64, 64, 64), "PB10_256x256": (256, 256, 256), "PB11_512x512_t1": (512, 512, 512),
                       "MM_32x32": (32, 32, 32), "MM_128x128": (128, 128, 128), "MMR_128x256_256x64": (128, 256, 64),
                       "MMR_512x128_128x256": (512, 128, 256), "MMR_1024x64_64x512": (1024, 64, 512)}.items():
    A, B = mats(M, K, N); box = [None]
    bench(cfg, lambda: box.__setitem__(0, A @ B))

bench("PB12_copy", lambda: t0.reshape(100, 10000).copy())
bench("PB12_inplace", lambda: t0.reshape(100, 10000))       # NumPy reshape is a view (no copy)

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cfg", "ref_median_ms", "ref_min_ms", "ref_mean_ms", "ref_std_ms", "library"])
    for r in rows: w.writerow(r + [f"NumPy {np.__version__} single-thread"])
print("wrote", out)
