#!/usr/bin/env python3
"""Merge functional + performance + reference CSVs into the mandatory result CSV and a summary table.
Usage: python3 tools/merge_results.py functional.csv perf.csv reference.csv [results.csv] [summary.md]"""
import csv, re, sys
func, perf, ref = sys.argv[1:4]
out = sys.argv[4] if len(sys.argv) > 4 else "results.csv"
summ = sys.argv[5] if len(sys.argv) > 5 else "summary.md"
refs = {r["cfg"]: r for r in csv.DictReader(open(ref))}
fn = list(csv.DictReader(open(func)))
pf = list(csv.DictReader(open(perf)))
cols = list(pf[0].keys())
own = {}
for r in pf:
    m = re.search(r"cfg=([^;]+)", r["Notes"]); r["_cfg"] = m.group(1) if m else ""
    own[r["_cfg"]] = r
for r in pf:
    cfg = r["_cfg"]; key = re.sub(r"_t\d+$", "_t1", cfg)            # thread rows use the single-thread reference
    rr = refs.get(cfg) or refs.get(key)
    if rr and r["Status"] not in ("NOT_RUN", "OUT_OF_MEMORY") and r["Student_Median_ms"]:
        s = float(r["Student_Median_ms"]); ref_ms = float(rr["ref_median_ms"])
        r["Reference_Median_ms"] = f"{ref_ms:.6f}"; r["Speedup"] = f"{ref_ms / s:.4f}" if s > 0 else ""
        lab = "reference = " + rr["library"]
        if re.search(r"_t[248]$", cfg):
            base = own.get(key)
            if base and base["Student_Median_ms"]: lab += f"; self-speedup vs 1 thread = {float(base['Student_Median_ms']) / s:.3f}x"
            lab += "; multi-thread student vs SINGLE-thread reference"
        r["Notes"] += "; " + lab
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in fn: w.writerow({c: r.get(c, "") for c in cols})
    for r in pf: w.writerow({c: r.get(c, "") for c in cols})
cnt = lambda rows, s: sum(1 for r in rows if r["Status"] == s)
lines = ["| Category | Total | Passed | Failed | Not supported | Remarks |", "|---|---|---|---|---|---|",
         f"| Functional tests | {len(fn)} | {cnt(fn,'PASS')} | {cnt(fn,'FAIL')} | {cnt(fn,'NOT_SUPPORTED')} | see TC35 note |",
         f"| Performance benchmarks | {len(pf)} | {cnt(pf,'PASS')} | {cnt(pf,'FAIL')} | {cnt(pf,'NOT_SUPPORTED')} | {cnt(pf,'NOT_RUN')} not run (CPU count) |", "",
         "| cfg | student median ms | reference median ms | speedup (ref/student) |", "|---|---|---|---|"]
for r in pf:
    if r["Student_Median_ms"]: lines.append(f"| {r['_cfg']} | {float(r['Student_Median_ms']):.4f} | {r['Reference_Median_ms'] or '-'} | {r['Speedup'] or '-'} |")
open(summ, "w").write("\n".join(lines) + "\n")
print("wrote", out, "and", summ)
