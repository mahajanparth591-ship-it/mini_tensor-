// Common Test Specification v1.0 - functional tests TC01..TC50.
// Writes CSV rows using the mandatory column names. No assert() is used, so this is NDEBUG-safe.
// Usage: ./common_test [Team_ID] [out.csv]
#include "tensor.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

struct Res {
    std::string op, status = "PASS", expected, actual, notes;
    double maxerr = -1;
    bool specError = false;   // FAIL caused by an erroneous expected value in the specification, not by the library
};
using Vals = std::vector<float>;
using Shape = std::vector<size_t>;

static std::string shp(const Shape& s) {
    std::string r = "{";
    for (size_t i = 0; i < s.size(); ++i) r += std::to_string(s[i]) + (i + 1 < s.size() ? "," : "");
    return r + "}";
}
static std::string vls(const Tensor& t, size_t maxn = 24) {
    std::ostringstream o; o << "[";
    for (size_t i = 0; i < t.size() && i < maxn; ++i) o << t[i] << (i + 1 < t.size() ? " " : "");
    if (t.size() > maxn) o << "...";
    o << "]"; return o.str();
}
static std::string vls(const Vals& v) {
    std::ostringstream o; o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << v[i] << (i + 1 < v.size() ? " " : "");
    o << "]"; return o.str();
}
// spec tolerance: |actual-expected| <= 1e-5 + 1e-4*|expected|
static bool within(float a, float e) { return std::fabs((double)a - e) <= 1e-5 + 1e-4 * std::fabs((double)e); }

// Checks shape and values; fills expected/actual/maxerr; marks FAIL on mismatch.
static void check(Res& r, const Tensor& t, const Shape& es, const Vals& ev) {
    r.expected = "shape " + shp(es) + " values " + vls(ev);
    r.actual = "shape " + shp(t.shape()) + " values " + vls(t);
    bool ok = (t.shape() == es) && t.size() == ev.size();
    double me = 0;
    if (ok) for (size_t i = 0; i < ev.size(); ++i) {
        me = std::max(me, std::fabs((double)t[i] - ev[i]));
        if (!within(t[i], ev[i])) ok = false;
    }
    r.maxerr = me;
    if (!ok) r.status = "FAIL";
}
static Vals iota(size_t n, float start = 0) { Vals v(n); for (size_t i = 0; i < n; ++i) v[i] = start + (float)i; return v; }

// Runs fn expecting it to throw E (documented error). Any other outcome = FAIL.
template <class E> static void expectThrow(Res& r, const std::function<void()>& fn, const std::string& what) {
    r.expected = "documented error (" + what + ")";
    try { fn(); r.status = "FAIL"; r.actual = "no error raised"; }
    catch (const E& e) { r.actual = std::string("exception: ") + e.what(); }
    catch (const std::exception& e) { r.status = "FAIL"; r.actual = std::string("wrong exception type: ") + e.what(); }
}

static std::vector<std::pair<std::string, Res>> results;
static void run(const std::string& id, const std::string& op, const std::function<void(Res&)>& fn) {
    Res r; r.op = op;
    try { fn(r); }
    catch (const std::exception& e) { r.status = "FAIL"; r.actual = std::string("unexpected exception: ") + e.what(); }
    catch (...) { r.status = "FAIL"; r.actual = "unexpected unknown exception"; }
    results.emplace_back(id, r);
    std::printf("%-5s %-6s %s\n", id.c_str(), r.status.c_str(), r.notes.c_str());
}

static std::string q(const std::string& s) {  // CSV quoting
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += "\"\""; else if (c == '\n') o += ' '; else o += c; }
    return o + "\"";
}

int main(int argc, char** argv) {
    std::string team = argc > 1 ? argv[1] : "TEAM";
    std::string out = argc > 2 ? argv[2] : "functional_results.csv";
    const Shape S22{2, 2};
    Vals a22{1, 2, 3, 4}, b22{5, 6, 7, 8};
    Vals sa{-2, 0.5f, 3, -4}, sb{1, -0.5f, -3, 2};

    // ---- A. creation ----
    run("TC01", "create", [&](Res& r) {
        Tensor s(Shape{}, Vals{5});
        r.expected = "rank 0; numel=1; value=5";
        r.actual = "rank " + std::to_string(s.ndim()) + "; numel=" + std::to_string(s.size()) + "; value=" + std::to_string(s.at({}));
        if (s.ndim() != 0 || s.size() != 1 || s.at({}) != 5) r.status = "FAIL";
    });
    run("TC02", "create", [&](Res& r) { Tensor t({5}, iota(5, 1)); check(r, t, {5}, iota(5, 1)); });
    run("TC03", "create", [&](Res& r) { Tensor t({2, 3}, iota(6, 1)); check(r, t, {2, 3}, iota(6, 1)); });
    run("TC04", "create", [&](Res& r) { Tensor t({2, 2, 3}, iota(12)); check(r, t, {2, 2, 3}, iota(12)); });
    run("TC05", "create", [&](Res& r) { Tensor t({2, 3, 2, 2}, iota(24)); check(r, t, {2, 3, 2, 2}, iota(24)); });
    run("TC06", "create", [&](Res& r) { Tensor t({2, 3}); check(r, t, {2, 3}, Vals(6, 0)); r.notes = "API promises zero-initialisation"; });
    run("TC07", "memory", [&](Res& r) {
        Tensor t({2, 3}, iota(6));
        bool ok = true;
        for (size_t i = 0; i + 1 < 6; ++i)
            if ((const char*)&t[i + 1] - (const char*)&t[i] != (long)sizeof(float)) ok = false;
        r.expected = "address(i+1)-address(i)=4 bytes for all 5 pairs";
        r.actual = ok ? "all 5 gaps = 4 bytes" : "gap mismatch";
        r.notes = "base pointer 64-byte aligned: " + std::string(((uintptr_t)t.data() % 64) == 0 ? "yes" : "no");
        if (!ok) r.status = "FAIL";
    });
    run("TC08", "memory", [&](Res& r) {
        Tensor t = Tensor::full({1000, 1000}, 2.0f);
        double sum = 0; for (size_t i = 0; i < t.size(); ++i) sum += t[i];
        r.expected = "numel=1000000; sum=2000000";
        r.actual = "numel=" + std::to_string(t.size()) + "; sum=" + std::to_string(sum);
        r.maxerr = std::fabs(sum - 2e6);
        if (t.size() != 1000000 || r.maxerr > 1e-5 + 1e-4 * 2e6) r.status = "FAIL";
    });

    // ---- B. indexing ----
    Tensor T({2, 3}, iota(6, 1));
    auto rd = [&](Res& r, size_t i, size_t j, float e) {
        float v = T.at({i, j});
        r.expected = std::to_string(e); r.actual = std::to_string(v); r.maxerr = std::fabs(v - e);
        if (!within(v, e)) r.status = "FAIL";
    };
    run("TC09", "index", [&](Res& r) { rd(r, 0, 0, 1); });
    run("TC10", "index", [&](Res& r) { rd(r, 1, 2, 6); });
    run("TC11", "index", [&](Res& r) { rd(r, 1, 1, 5); });
    run("TC12", "index", [&](Res& r) {
        Tensor t({2, 3}, iota(6, 1));
        t.at({0, 1}) = 20;
        check(r, t, {2, 3}, Vals{1, 20, 3, 4, 5, 6});
    });
    run("TC13", "index", [&](Res& r) { expectThrow<std::out_of_range>(r, [&] { T.at({2, 0}); }, "out_of_range"); });
    run("TC14", "index", [&](Res& r) { expectThrow<std::out_of_range>(r, [&] { T.at({0, 3}); }, "out_of_range"); });
    run("TC15", "index", [&](Res& r) {
        int neg = -1;
        expectThrow<std::out_of_range>(r, [&] { T.at({(size_t)neg, 0}); }, "out_of_range");
        r.notes = "API takes unsigned indices; a signed -1 converts to SIZE_MAX and is rejected by the bounds check";
    });
    run("TC16", "index", [&](Res& r) {
        Tensor t({2, 3, 4}, iota(24));
        float v = t.at({1, 2, 3});
        r.expected = "23 (offset 1*12+2*4+3)"; r.actual = std::to_string(v) + " (offset " + std::to_string(t.offset({1, 2, 3})) + ")";
        if (v != 23 || t.offset({1, 2, 3}) != 23) r.status = "FAIL";
    });

    // ---- C. add / subtract ----
    Tensor A22({2, 2}, a22), B22({2, 2}, b22), Z22({2, 2}), SA({2, 2}, sa), SB({2, 2}, sb);
    run("TC17", "add", [&](Res& r) { check(r, A22 + B22, S22, {6, 8, 10, 12}); });
    run("TC18", "subtract", [&](Res& r) { check(r, A22 - B22, S22, {-4, -4, -4, -4}); });
    run("TC19", "add", [&](Res& r) { check(r, A22 + Z22, S22, a22); });
    run("TC20", "subtract", [&](Res& r) { check(r, A22 - Z22, S22, a22); });
    run("TC21", "add", [&](Res& r) { check(r, SA + SB, S22, {-1, 0, 0, -2}); });
    run("TC22", "subtract", [&](Res& r) { check(r, SA - SB, S22, {-3, 1, 6, -6}); });
    run("TC23", "add", [&](Res& r) {
        Tensor A({2, 3}, iota(6, 1)), B({3, 2}, iota(6, 1));
        expectThrow<std::invalid_argument>(r, [&] { auto C = A + B; }, "shape mismatch");
    });
    run("TC24", "subtract", [&](Res& r) {
        Tensor A({2, 3}, iota(6, 1)), B({6}, iota(6, 1));
        expectThrow<std::invalid_argument>(r, [&] { auto C = A - B; }, "shape mismatch");
    });

    // ---- D. element-wise multiply ----
    Tensor O22 = Tensor::ones({2, 2});
    run("TC25", "elementwise_multiply", [&](Res& r) { check(r, A22.mul(B22), S22, {5, 12, 21, 32}); });
    run("TC26", "elementwise_multiply", [&](Res& r) { check(r, A22.mul(Z22), S22, {0, 0, 0, 0}); });
    run("TC27", "elementwise_multiply", [&](Res& r) { check(r, A22.mul(O22), S22, a22); });
    run("TC28", "elementwise_multiply", [&](Res& r) { check(r, SA.mul(SB), S22, {-2, -0.25f, -9, -8}); });
    run("TC29", "elementwise_multiply", [&](Res& r) {
        Tensor A({2, 2}, Vals{0.1f, 0.2f, 0.3f, 0.4f}), B({2, 2}, Vals{0.2f, 0.3f, 0.4f, 0.5f});
        check(r, A.mul(B), S22, {0.02f, 0.06f, 0.12f, 0.20f});
    });
    run("TC30", "elementwise_multiply", [&](Res& r) {
        Tensor A({2, 3}, iota(6, 1)), B({3, 2}, iota(6, 1));
        expectThrow<std::invalid_argument>(r, [&] { auto C = A.mul(B); }, "shape mismatch");
    });

    // ---- E. matmul ----
    auto mm = [&](Res& r, Shape sa_, Vals da, Shape sb_, Vals db, Shape es, Vals ev) {
        Tensor A(sa_, da), B(sb_, db);
        check(r, Tensor::matmul(A, B), es, ev);
    };
    run("TC31", "matmul", [&](Res& r) { mm(r, {2, 3}, {1, 2, 3, 4, 5, 6}, {3, 2}, {7, 8, 9, 10, 11, 12}, {2, 2}, {58, 64, 139, 154}); });
    run("TC32", "matmul", [&](Res& r) { mm(r, {3, 3}, iota(9, 1), {3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3}, {30, 24, 18, 84, 69, 54, 138, 114, 90}); });
    run("TC33", "matmul", [&](Res& r) { mm(r, {2, 3}, iota(6, 1), {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {2, 3}, iota(6, 1)); });
    run("TC34", "matmul", [&](Res& r) { mm(r, {2, 2}, a22, {2, 3}, Vals(6, 0), {2, 3}, Vals(6, 0)); });
    run("TC35", "matmul", [&](Res& r) {
        // Spec text: [[190,200,210,220],[440,475,510,545]].  The mathematically correct product of the
        // given inputs is [[175,190,205,220],[400,440,480,520]] (verified with NumPy).
        Tensor A({2, 5}, iota(10, 1)), B({5, 4}, iota(20, 1));
        Tensor C = Tensor::matmul(A, B);
        check(r, C, {2, 4}, {190, 200, 210, 220, 440, 475, 510, 545});   // spec as written
        if (r.status == "FAIL") {
            Res c2; check(c2, C, {2, 4}, {175, 190, 205, 220, 400, 440, 480, 520});
            r.specError = (c2.status == "PASS");
            r.notes = c2.status == "PASS"
                ? "Library output equals the mathematically correct product [[175,190,205,220],[400,440,480,520]] (NumPy-verified); the expected values printed in the specification do not match its own inputs - spec error to confirm with evaluator"
                : "Output matches neither the specification nor the NumPy-verified product";
            if (c2.status != "PASS") r.notes += " [LIBRARY BUG]";
        }
    });
    run("TC36", "matmul", [&](Res& r) { mm(r, {1, 1}, {3.5f}, {1, 1}, {2.0f}, {1, 1}, {7.0f}); });
    run("TC37", "matmul", [&](Res& r) {
        Tensor A({2, 3}, iota(6)), B({4, 2}, iota(8));
        expectThrow<std::invalid_argument>(r, [&] { auto C = Tensor::matmul(A, B); }, "inner dimension mismatch");
    });
    run("TC38", "matmul", [&](Res& r) { mm(r, {3, 2}, {1, 2, 3, 4, 5, 6}, {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, {3, 4}, {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68}); });

    // ---- F. reshape ----
    Tensor V({12}, iota(12));
    run("TC39", "reshape", [&](Res& r) { check(r, V.reshape({3, 4}), {3, 4}, iota(12)); });
    run("TC40", "reshape", [&](Res& r) { check(r, V.reshape({3, 4}).reshape({2, 2, 3}), {2, 2, 3}, iota(12)); });
    run("TC41", "reshape", [&](Res& r) { check(r, V.reshape({2, 2, 3}).reshape({12}), {12}, iota(12)); });
    run("TC42", "reshape", [&](Res& r) { Tensor t({2, 3}, iota(6, 1)); check(r, t.reshape({3, 2}), {3, 2}, iota(6, 1)); });
    run("TC43", "reshape", [&](Res& r) {
        Tensor t({2, 3}, iota(6, 1));
        expectThrow<std::invalid_argument>(r, [&] { auto x = t.reshape({4, 4}); }, "element count mismatch");
        Res c; check(c, t, {2, 3}, iota(6, 1));
        if (c.status != "PASS") { r.status = "FAIL"; r.notes = "original tensor changed"; } else r.notes = "original tensor unchanged";
    });
    run("TC44", "reshape", [&](Res& r) { check(r, V.reshape({1, 12, 1}), {1, 12, 1}, iota(12)); });
    run("TC45", "reshape", [&](Res& r) {
        Tensor t = V; bool ok = true;
        for (int k = 0; k < 100 && ok; ++k) {
            Tensor a = t.reshape({3, 4}); Tensor b = a.reshape({2, 2, 3}); Tensor c = b.reshape({12});
            Res c2; check(c2, c, {12}, iota(12)); ok = (c2.status == "PASS");
        }
        r.expected = "100 round trips keep shape {12} and values 0..11"; r.actual = ok ? "stable" : "corrupted";
        r.notes = "reshape() copies data (deep copy semantics); reshape_() is the zero-copy in-place variant";
        if (!ok) r.status = "FAIL";
    });

    // ---- G. ownership / memory ----
    run("TC46", "memory", [&](Res& r) {
        double acc = 0;
        for (int k = 0; k < 10000; ++k) { Tensor t = Tensor::full({100, 100}, 1.0f); acc += t[k % 10000]; }
        r.expected = "10000 create/fill/destroy cycles, no leak or invalid free";
        r.actual = "completed; checksum " + std::to_string(acc);
        r.notes = "run this binary built with -fsanitize=address,undefined (see README) for the real check";
        if (acc != 10000) r.status = "FAIL";
    });
    run("TC47", "memory", [&](Res& r) {
        Tensor A({2, 2}, a22); Tensor B(A);
        B.at({0, 0}) = 99;
        Res c1, c2; check(c1, A, S22, a22); check(c2, B, S22, {99, 2, 3, 4});
        r.expected = "A=[1 2 3 4]; B=[99 2 3 4]"; r.actual = "A=" + vls(A) + "; B=" + vls(B);
        r.notes = "deep copy"; if (c1.status != "PASS" || c2.status != "PASS") r.status = "FAIL";
    });
    run("TC48", "memory", [&](Res& r) {
        Tensor A({2, 2}, a22), B({1}, Vals{9});
        B = A; B.at({1, 1}) = 77;
        Res c1, c2; check(c1, A, S22, a22); check(c2, B, S22, {1, 2, 3, 77});
        r.expected = "B shape {2,2}=[1 2 3 77]; A unchanged"; r.actual = "A=" + vls(A) + "; B=" + shp(B.shape()) + vls(B);
        r.notes = "deep copy assignment"; if (c1.status != "PASS" || c2.status != "PASS") r.status = "FAIL";
    });
    run("TC49", "memory", [&](Res& r) {
        bool ok = true; double me = 0;
        for (int it = 1; it <= 100; ++it) {
            Tensor t = Tensor::full({500, 500}, (float)it);
            double s = 0; for (size_t i = 0; i < t.size(); ++i) s += t[i];
            me = std::max(me, std::fabs(s - 250000.0 * it));
            if (std::fabs(s - 250000.0 * it) > 1e-5 + 1e-4 * 250000.0 * it) ok = false;
        }
        r.expected = "100 iterations, each sum = 250000*iteration"; r.actual = ok ? "all sums correct" : "sum mismatch"; r.maxerr = me;
        if (!ok) r.status = "FAIL";
    });
    run("TC50", "memory", [&](Res& r) {
        const size_t big = (size_t)1 << 31;
        r.expected = "overflow/allocation failure reported safely (no crash, hang or overflow)";
        try { Tensor t({big, big}); r.status = "FAIL"; r.actual = "constructed (!)"; }
        catch (const std::length_error& e) { r.actual = std::string("std::length_error: ") + e.what(); }
        catch (const std::bad_alloc& e) { r.actual = std::string("std::bad_alloc: ") + e.what(); }
    });

    // ---- CSV + summary ----
    std::ofstream f(out);
    f << "Team_ID,Test_ID,Operation,Status,Expected,Actual,Max_Abs_Error,Student_Median_ms,Reference_Median_ms,Speedup,Peak_Memory_MB,Notes\n";
    int pass = 0, fail = 0, specErr = 0;
    for (auto& [id, r] : results) {
        (r.status == "PASS" ? pass : fail)++;
        if (r.specError) ++specErr;
        f << team << ',' << id << ',' << r.op << ',' << r.status << ',' << q(r.expected) << ',' << q(r.actual) << ','
          << (r.maxerr >= 0 ? std::to_string(r.maxerr) : "") << ",,,,," << q(r.notes) << '\n';
    }
    std::printf("\nFunctional tests: %zu run, %d PASS, %d FAIL (%d of them are specification errors, see Notes)  -> %s\n", results.size(), pass, fail, specErr, out.c_str());
    return (fail - specErr) ? 1 : 0;   // exit code ignores only verified specification errors
}
