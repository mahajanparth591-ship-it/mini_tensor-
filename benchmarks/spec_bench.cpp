// Common Test Specification v1.0 - performance benchmarks PB01..PB12 plus required scaling runs.
// Protocol: 5 warm-ups, 30 timed runs, 3 independent repetitions; report median/min/mean/std.
// Usage: ./spec_bench [Team_ID] [out.csv] [--quick]      (--quick = fewer reps, for a smoke test only)
#include "tensor.hpp"
#include <omp.h>
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;   // monotonic
static volatile double g_sink = 0;         // consume outputs so nothing is optimised away
static int WARM = 5, REPS = 30, RUNS = 3;

struct Stats { double median, min, mean, sd; };
static Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double med = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / n, ss = 0;
    for (double x : v) ss += (x - mean) * (x - mean);
    return {med, v.front(), mean, n > 1 ? std::sqrt(ss / (n - 1)) : 0};
}
static double peakMB() { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss / 1024.0; }

struct Row {
    std::string id, op, status, expected, actual, cfg, notes;
    double maxerr = -1, median = -1, peak = -1;
    double runs[3] = {0, 0, 0}; Stats st{0, 0, 0, 0}; double gflops = -1, gbs = -1;
};
static std::vector<Row> rows;

// timed() is measured; after() (consume/cleanup/verify) is not.
static void bench(Row r, const std::function<void()>& timed, const std::function<void()>& after,
                  const std::function<bool(std::string&, double&)>& verify = nullptr) {
    std::vector<double> all, meds;
    try {
        for (int run = 0; run < RUNS; ++run) {
            for (int w = 0; w < WARM; ++w) { timed(); after(); }
            std::vector<double> t;
            for (int k = 0; k < REPS; ++k) {
                auto a = Clock::now(); timed(); auto b = Clock::now();
                t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
                after();
            }
            meds.push_back(stats(t).median);
            all.insert(all.end(), t.begin(), t.end());
        }
        for (int i = 0; i < 3; ++i) r.runs[i] = meds[i];
        r.st = stats(all);
        r.median = stats(meds).median;           // overall median of the three medians
        r.status = "PASS";
        if (verify) { std::string act; double me = -1; bool ok = verify(act, me); r.actual = act; r.maxerr = me; if (!ok) r.status = "FAIL"; }
    } catch (const std::bad_alloc&) { r.status = "OUT_OF_MEMORY"; r.actual = "allocation failed"; }
    r.peak = peakMB();
    rows.push_back(r);
    std::printf("%-26s %-6s median %10.4f ms  (runs %.4f %.4f %.4f)\n", r.cfg.c_str(), r.status.c_str(), r.median, r.runs[0], r.runs[1], r.runs[2]);
}

static Tensor makeA(size_t M, size_t K) { Tensor A({M, K}); for (size_t i = 0; i < M; ++i) for (size_t j = 0; j < K; ++j) A[i * K + j] = ((i + j) % 7) * 0.1f; return A; }
static Tensor makeB(size_t K, size_t N) { Tensor B({K, N}); for (size_t i = 0; i < K; ++i) for (size_t j = 0; j < N; ++j) B[i * N + j] = ((2 * i + j) % 5) * 0.2f; return B; }
// Reference product in double precision (outside timing).
static void refMatmul(const Tensor& A, const Tensor& B, std::vector<double>& C) {
    size_t M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
    C.assign(M * N, 0.0);
    for (size_t i = 0; i < M; ++i) for (size_t k = 0; k < K; ++k) { double a = A[i * K + k]; for (size_t j = 0; j < N; ++j) C[i * N + j] += a * B[k * N + j]; }
}
static std::string dims(size_t a, size_t b) { return std::to_string(a) + "x" + std::to_string(b); }
static double checksum(const Tensor& t) { double s = 0; for (size_t i = 0; i < t.size(); ++i) s += t[i]; return s; }

// ---- benchmark families ----
static void creation(const std::string& id, size_t n, float fillv, bool zero, const std::string& cfg) {
    std::unique_ptr<Tensor> p;
    Row r; r.id = id; r.op = "create"; r.cfg = cfg; r.expected = "created and initialised " + dims(n, n);
    bench(r, [&] { p.reset(zero ? new Tensor({n, n}) : new Tensor(Tensor::full({n, n}, fillv))); },
          [&] { g_sink += (*p)[0]; p.reset(); });
    Row& last = rows.back(); last.gbs = (double)n * n * 4 / (last.median * 1e-3) / 1e9;
    last.notes = "peak memory is process high-water mark";
}
static void traversal(const char* id, const char* cfg) {
    Tensor t({1000, 1000}); for (size_t i = 0; i < t.size(); ++i) t[i] = (float)(i + 1);
    double cs = 0;
    Row r; r.id = id; r.op = "index"; r.cfg = cfg; r.expected = "checksum 500000500000";
    bench(r, [&] { double s = 0; const float* d = t.data(); for (size_t i = 0; i < t.size(); ++i) s += d[i]; cs = s; },
          [&] { g_sink += cs; },
          [&](std::string& a, double& me) { a = "checksum " + std::to_string((long long)cs); me = std::fabs(cs - 500000500000.0); return me == 0; });
}
static void randomReads(const char* id, const char* cfg, const std::string& idxfile) {
    Tensor t({1000, 1000}); for (size_t i = 0; i < t.size(); ++i) t[i] = (float)i;
    std::mt19937 gen(20261117u);
    std::vector<std::pair<size_t, size_t>> idx(100000);
    std::ofstream f(idxfile);
    for (auto& p : idx) { p.first = gen() % 1000; p.second = gen() % 1000; f << p.first << ' ' << p.second << '\n'; }
    double expect = 0; for (auto& p : idx) expect += (double)(p.first * 1000 + p.second);
    double cs = 0;
    Row r; r.id = id; r.op = "index"; r.cfg = cfg; r.expected = "checksum " + std::to_string((long long)expect);
    bench(r, [&] { double s = 0; for (auto& p : idx) s += t.at({p.first, p.second}); cs = s; },
          [&] { g_sink += cs; },
          [&](std::string& a, double& me) { a = "checksum " + std::to_string((long long)cs); me = std::fabs(cs - expect); return me == 0; });
    rows.back().notes = "uses checked at(); index list written to " + idxfile;
}
static void elementwise(const char* id, const char* op, size_t n, float av, float bv, int kind, double expectEach, const std::string& cfg) {
    Tensor A = Tensor::full({n, n}, av), B = Tensor::full({n, n}, bv);
    std::unique_ptr<Tensor> C;
    Row r; r.id = id; r.op = op; r.cfg = cfg; r.expected = "each element " + std::to_string(expectEach);
    bench(r, [&] { C.reset(new Tensor(kind == 0 ? A + B : kind == 1 ? A - B : A.mul(B))); }, [&] { g_sink += (*C)[0]; },
          [&](std::string& a, double& me) {
              double s = checksum(*C), e = expectEach * (double)n * n; a = "sum " + std::to_string(s); me = std::fabs(s - e);
              return me <= 1e-5 + 1e-4 * e; });
    Row& l = rows.back(); l.gbs = 3.0 * n * n * 4 / (l.median * 1e-3) / 1e9;
}
static void matmulCase(const char* id, size_t M, size_t K, size_t N, const std::string& cfg) {
    Tensor A = makeA(M, K), B = makeB(K, N);
    std::vector<double> ref; refMatmul(A, B, ref);
    std::unique_ptr<Tensor> C;
    Row r; r.id = id; r.op = "matmul"; r.cfg = cfg; r.expected = "matches double-precision reference";
    bench(r, [&] { C.reset(new Tensor(Tensor::matmul(A, B))); }, [&] { g_sink += (*C)[0]; },
          [&](std::string& a, double& me) {
              me = 0; bool ok = true;
              for (size_t i = 0; i < ref.size(); ++i) { double d = std::fabs((*C)[i] - ref[i]); me = std::max(me, d); if (d > 1e-5 + 1e-4 * std::fabs(ref[i])) ok = false; }
              a = "max abs err vs reference " + std::to_string(me) + "; checksum " + std::to_string(checksum(*C)); return ok; });
    Row& l = rows.back(); l.gflops = 2.0 * M * K * N / (l.median * 1e-3) / 1e9;
}
static void reshapeCases() {
    Tensor t({1000, 1000}); for (size_t i = 0; i < t.size(); ++i) t[i] = (float)i;
    std::unique_ptr<Tensor> out;
    Row r; r.id = "PB12"; r.op = "reshape"; r.cfg = "PB12_copy"; r.expected = "shape {100,10000}, values 0..999999 in order";
    bench(r, [&] { out.reset(new Tensor(t.reshape({100, 10000}))); }, [&] { g_sink += (*out)[999999]; },
          [&](std::string& a, double&) { a = "shape {" + std::to_string(out->shape()[0]) + "," + std::to_string(out->shape()[1]) + "}";
              return out->shape()[0] == 100 && out->shape()[1] == 10000 && (*out)[999999] == 999999.0f; });
    rows.back().notes = "COPY semantics: reshape() deep-copies 4 MB";
    Tensor u = t;   // in-place variant: toggle between two valid shapes
    bool flip = false;
    Row r2; r2.id = "PB12"; r2.op = "reshape"; r2.cfg = "PB12_inplace"; r2.expected = "shape changes, no data movement";
    bench(r2, [&] { u.reshape_(flip ? std::vector<size_t>{1000, 1000} : std::vector<size_t>{100, 10000}); }, [&] { flip = !flip; g_sink += u[999999]; },
          [&](std::string& a, double&) { a = "ndim " + std::to_string(u.ndim()); return u[999999] == 999999.0f; });
    rows.back().notes = "IN-PLACE semantics: reshape_() O(1), no copy; comparable to a NumPy view (not to the copy row)";
}

int main(int argc, char** argv) {
    std::string team = "TEAM", out = "perf_results.csv"; int pos = 0; bool reuse = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick") { WARM = 1; REPS = 3; RUNS = 3; }
        else if (a == "--default-malloc") reuse = false;
        else if (pos++ == 0) team = a; else out = a;
    }
    bool tuned = reuse && Tensor::enable_memory_reuse();   // opt-in allocator tuning (disable with --default-malloc)
    const int procs = omp_get_num_procs();
    std::printf("procs=%d  warmup=%d reps=%d runs=%d  allocator_reuse=%s\n", procs, WARM, REPS, RUNS, tuned ? "on" : "off");
    omp_set_num_threads(1);   // single-thread baseline for everything below
    std::string base = out.substr(0, out.rfind('.'));

    creation("PB01", 100, 0, true, "PB01_100x100");
    creation("PB02", 1000, 1.0f, false, "PB02_1000x1000");
    creation("PB03", 4000, 1.0f, false, "PB03_4000x4000");
    traversal("PB04", "PB04_traverse");
    randomReads("PB05", "PB05_random_reads", base + "_pb05_indices.txt");
    for (size_t n : {100, 500, 1000, 4000}) {
        std::string s = dims(n, n);
        elementwise("PB06", "add", n, 1.25f, 2.5f, 0, 3.75, "PB06_" + s + "_t1");
        elementwise("PB07", "subtract", n, 5.0f, 1.5f, 1, 3.5, "PB07_" + s + "_t1");
        elementwise("PB08", "elementwise_multiply", n, 1.25f, 2.0f, 2, 2.5, "PB08_" + s + "_t1");
    }
    matmulCase("PB09", 64, 64, 64, "PB09_64x64");
    matmulCase("PB10", 256, 256, 256, "PB10_256x256");
    matmulCase("PB11", 512, 512, 512, "PB11_512x512_t1");
    for (size_t n : {32, 128}) matmulCase("PB09", n, n, n, "MM_" + dims(n, n));
    matmulCase("PB09", 128, 256, 64, "MMR_128x256_256x64");
    matmulCase("PB10", 512, 128, 256, "MMR_512x128_128x256");
    matmulCase("PB11", 1024, 64, 512, "MMR_1024x64_64x512");
    reshapeCases();

    // OpenMP scaling: PB06 and PB11 at 1,2,4,8 threads (only where the machine has that many CPUs)
    for (int t : {2, 4, 8}) {   // 1-thread rows already measured above (PB06_1000x1000_t1, PB11_512x512_t1)
        if (t > procs) {
            for (const char* c : {"PB06_1000x1000", "PB11_512x512"}) {
                Row r; r.id = c[3] == '6' ? "PB06" : "PB11"; r.op = c[3] == '6' ? "add" : "matmul"; r.status = "NOT_RUN";
                r.cfg = std::string(c) + "_t" + std::to_string(t); r.notes = "machine has only " + std::to_string(procs) + " CPUs"; rows.push_back(r);
            }
            continue;
        }
        omp_set_num_threads(t);
        elementwise("PB06", "add", 1000, 1.25f, 2.5f, 0, 3.75, "PB06_1000x1000_t" + std::to_string(t));
        matmulCase("PB11", 512, 512, 512, "PB11_512x512_t" + std::to_string(t));
    }

    std::ofstream f(out), d(base + "_detail.csv");
    f << "Team_ID,Test_ID,Operation,Status,Expected,Actual,Max_Abs_Error,Student_Median_ms,Reference_Median_ms,Speedup,Peak_Memory_MB,Notes\n";
    d << "cfg,run1_median_ms,run2_median_ms,run3_median_ms,overall_median_ms,min_ms,mean_ms,std_ms,gflops,est_bandwidth_GBps\n";
    auto qt = [](const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"') o += "\"\""; else o += c; } return o + "\""; };
    for (auto& r : rows) {
        std::ostringstream notes; notes << "cfg=" << r.cfg << "; allocator_reuse=" << (tuned ? "on" : "off");
        if (r.status != "NOT_RUN") notes << "; medians(ms)=" << r.runs[0] << "/" << r.runs[1] << "/" << r.runs[2] << "; min=" << r.st.min << "; mean=" << r.st.mean << "; std=" << r.st.sd;
        if (!r.notes.empty()) notes << "; " << r.notes;
        f << team << ',' << r.id << ',' << r.op << ',' << r.status << ',' << qt(r.expected) << ',' << qt(r.actual) << ','
          << (r.maxerr >= 0 ? std::to_string(r.maxerr) : "") << ',' << (r.median >= 0 ? std::to_string(r.median) : "") << ",,,"
          << (r.peak >= 0 ? std::to_string(r.peak) : "") << ',' << qt(notes.str()) << '\n';
        if (r.status != "NOT_RUN")
            d << r.cfg << ',' << r.runs[0] << ',' << r.runs[1] << ',' << r.runs[2] << ',' << r.median << ',' << r.st.min << ',' << r.st.mean << ',' << r.st.sd << ','
              << (r.gflops >= 0 ? std::to_string(r.gflops) : "") << ',' << (r.gbs >= 0 ? std::to_string(r.gbs) : "") << '\n';
    }
    int bad = 0; for (auto& r : rows) if (r.status == "FAIL") ++bad;
    std::printf("\n%zu benchmark rows written to %s (%d FAIL). sink=%g\n", rows.size(), out.c_str(), bad, (double)g_sink);
    return bad ? 1 : 0;
}
