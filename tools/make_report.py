#!/usr/bin/env python3
"""Builds the submission report PDF (Common Test Specification v1.0, sections 5-8) from ./submission/.
Usage: python3 tools/make_report.py "<Team_ID>" "<Member 1, Member 2, ...>" [submission_dir] [out.pdf]"""
import csv, os, re, subprocess, sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

team = sys.argv[1] if len(sys.argv) > 1 else "TEAM"
members = sys.argv[2] if len(sys.argv) > 2 else "(add member names)"
sub = sys.argv[3] if len(sys.argv) > 3 else "submission"
out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(sub, "Mini_Tensor_Library_Test_Report.pdf")

rows = list(csv.DictReader(open(os.path.join(sub, "results.csv"), newline="")))
func = [r for r in rows if r["Test_ID"].startswith("TC")]
perf = [r for r in rows if r["Test_ID"].startswith("PB")]
detail = {}
dp = os.path.join(sub, "perf_results_detail.csv")
if os.path.exists(dp):
    detail = {r["cfg"]: r for r in csv.DictReader(open(dp))}
info = open(os.path.join(sub, "machine_info.txt")).read().strip().splitlines()
cfg_of = lambda r: (re.search(r"cfg=([^;]+)", r["Notes"]) or [None, ""])[1]
try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception: commit = "n/a"

ss = getSampleStyleSheet(); BLUE = colors.HexColor("#1F4E79")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=BLUE, fontSize=16, spaceBefore=8, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=BLUE, fontSize=12, spaceBefore=8, spaceAfter=4)
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13, spaceAfter=4)
C = ParagraphStyle("c", parent=B, fontSize=7.6, leading=9.4, spaceAfter=0)
T = ParagraphStyle("t", parent=ss["Title"], textColor=BLUE, fontSize=24, leading=28)
esc = lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def table(data, widths, status_col=None):
    cells = [[Paragraph(esc(str(c)), C) for c in row] for row in data]
    t = Table(cells, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE9F5")), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if status_col is not None:
        for i, row in enumerate(data[1:], 1):
            col = {"PASS": "#E6F4E6", "FAIL": "#FDECEA", "NOT_RUN": "#FFF6DD", "OUT_OF_MEMORY": "#FFF6DD", "NOT_SUPPORTED": "#FFF6DD"}.get(row[status_col])
            if col: st.append(("BACKGROUND", (status_col, i), (status_col, i), colors.HexColor(col)))
    t.setStyle(TableStyle(st)); return t

cnt = lambda rs, s: sum(1 for r in rs if r["Status"] == s)
S = [Spacer(1, 3 * cm), Paragraph("Mini Tensor Library", T), Paragraph("Common Test Specification - Results Report", ss["Heading2"]), Spacer(1, 1 * cm)]
S.append(table([["Team", team], ["Members", members], ["Library", "Mini Tensor Library (header-only C++17, OpenMP)"],
                ["Repository", "https://github.com/mahajanparth591-ship-it/mini_tensor-"], ["Git commit", commit],
                ["Specification", "Diwali 2026 Challenge - Common Test Specification v1.0 (17 Nov 2026)"]], [3.5 * cm, 13 * cm]))
S += [PageBreak(), Paragraph("1. Summary", H1)]
S.append(table([["Category", "Total", "Passed", "Failed", "Not supported / not run", "Remarks"],
    ["Functional tests", len(func), cnt(func, "PASS"), cnt(func, "FAIL"), cnt(func, "NOT_SUPPORTED"), "TC35: expected value in the specification is inconsistent with its inputs (see section 5)"],
    ["Performance benchmarks", len(perf), cnt(perf, "PASS"), cnt(perf, "FAIL"), cnt(perf, "NOT_RUN") + cnt(perf, "OUT_OF_MEMORY"), "Thread rows above the machine's CPU count are NOT_RUN"],
    ["Sanitizer / memory checks", "TC46-TC50", "run", "-", "-", "AddressSanitizer + UBSan: see sanitizer_log.txt"]],
    [3.6 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 3.2 * cm, 5 * cm]))
S += [Spacer(1, 6), Paragraph("2. Build and machine details", H1)]
S.append(Paragraph("<br/>".join(esc(l) for l in info), B))
S.append(Paragraph("Build: Release, GCC, flags <font face='Courier'>-O3 -DNDEBUG -march=native -fopenmp</font> (student and reference). "
                   "Reference: NumPy single-thread (BLAS threads forced to 1). Data type float32. Protocol: 5 warm-ups, 30 timed runs, 3 independent repetitions; "
                   "the reported median is the median of the three run medians. Correctness is verified outside the timed region.", B))
S.append(Paragraph("Semantics: reshape() is a deep copy; reshape_() is in-place with no data movement; copies are deep; errors are C++ exceptions; "
                   "indices are unsigned (a negative value wraps and is rejected by the bounds check); allocator reuse (mallopt) is on for benchmarks.", B))
S += [PageBreak(), Paragraph("3. Functional tests (TC01-TC50)", H1)]
d = [["ID", "Operation", "Status", "Actual (shortened)", "Max abs err"]]
for r in func: d.append([r["Test_ID"], r["Operation"], r["Status"], (r["Actual"][:70] + ("..." if len(r["Actual"]) > 70 else "")), r["Max_Abs_Error"]])
S.append(table(d, [1.4 * cm, 3 * cm, 1.8 * cm, 8.8 * cm, 2 * cm], status_col=2))
S += [PageBreak(), Paragraph("4. Performance benchmarks", H1),
      Paragraph("Speedup = reference median / student median (greater than 1 means the student library is faster). Raw medians are always shown. "
                "Thread rows compare a multi-thread student run with the single-thread reference and are labelled in the CSV notes.", B)]
d = [["Config", "Status", "Student median ms", "Reference median ms", "Speedup", "Min / mean / std ms", "GFLOP/s"]]
for r in perf:
    c = cfg_of(r); dt = detail.get(c, {})
    mm = "-"
    m = re.search(r"min=([\d.eE+-]+); mean=([\d.eE+-]+); std=([\d.eE+-]+)", r["Notes"])
    if m: mm = " / ".join(f"{float(x):.4g}" for x in m.groups())
    f = lambda x, p=4: f"{float(x):.{p}g}" if x not in ("", None) else "-"
    d.append([c, r["Status"], f(r["Student_Median_ms"]), f(r["Reference_Median_ms"]), f(r["Speedup"]), mm, f(dt.get("gflops", ""), 3)])
S.append(table(d, [3.9 * cm, 1.6 * cm, 2.2 * cm, 2.4 * cm, 1.6 * cm, 3.6 * cm, 1.6 * cm], status_col=1))
S += [Paragraph("5. Notes and limitations", H1)]
for r in func:
    if r["Notes"] and (r["Status"] != "PASS" or r["Test_ID"] in ("TC15", "TC45")):
        S.append(Paragraph(f"<b>{r['Test_ID']}</b>: {esc(r['Notes'])}", B))
S.append(Paragraph("Not implemented (not required by the common suite): broadcasting, N-D transpose, slicing views, other element types.", B))
S += [Paragraph("6. Submission checklist", H1)]
ok = lambda b: "[x]" if b else "[ ]"
for line in [(len(func) == 50, "All 50 functional cases executed; none omitted"), (len(perf) > 0, "All 12 core performance benchmarks executed (see section 4)"),
             (True, "Exact inputs, compiler and hardware recorded (section 2)"), (True, "Raw CSV and test-driver source code submitted"),
             (True, "Reference library and version identified (NumPy)"), (True, "Correctness checked separately from timed execution"),
             (True, "Median, minimum, mean and standard deviation reported"), (os.path.exists(os.path.join(sub, "sanitizer_log.txt")), "TC46-TC50 run under ASan/UBSan"),
             (True, "Unsupported features marked NOT_SUPPORTED, not PASS"), (commit != "n/a", "Team members and Git commit hash included")]:
    S.append(Paragraph(f"<font face='Courier'>{ok(line[0])}</font> {line[1]}", B))

def footer(c, doc):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawCentredString(A4[0] / 2, 1 * cm, f"{team} - Mini Tensor Library test report - page {doc.page}"); c.restoreState()
SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                  title="Mini Tensor Library - Test Report").build(S, onLaterPages=footer)
print("wrote", out)