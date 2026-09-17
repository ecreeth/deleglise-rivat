#ifndef MERTENS_DR_WINDOWED_HPP
#define MERTENS_DR_WINDOWED_HPP
// Windowed (virtualized) Deléglise-Rivat: streams the M table in cache-sized
// windows built from a retained odd-only mu sieve. Bit-identical S1/S2 math
// to mertens_dr.hpp baseline (fast_div exact + same M values).
// Merged: q lookups use fast_div (>2^53 integer fallback) so windowed is
// exact to 1e17 like the dense path.
//
// The sieve itself lives in SieveTable (mertens_dr.hpp) — this header only
// adds the windowed S1 consumption on top.

#include "mertens_dr.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mertens_windowed {
using mertens_dr::int64;

struct S1Interval {
    int64 y = 0;
    int64 lo = 1, hi = 0; // n-range inclusive
    int mode = 0;         // 0=all, 1=odd only, 2=even only
    int acc = 0;          // 0 -> s1a, 1 -> s1b
    int sign = 1;         // +1 / -1 internal (S1_diff convention)
};

struct TermW {
    int64 k = 0;
    int8_t mu_k = 0;
    uint8_t range = 4;
    int64 s1a = 0, s1b = 0;
    // precomputed constants for final combine (non-S1 parts)
    int64 C0 = 0, S2_0 = 0; // for first task (comb2 or single)
    int64 C1 = 0, S2_1 = 0; // for second task (range1/2 only)
    int64 one = 0;          // 1 for single tasks (eval_single returns 1-...)
    int64 one1 = 0;
    std::vector<S1Interval> ivs;
};

static inline int64 compute_mertens_windowed(int64 X, int num_threads = 0, int64 block_M = (1<<19), bool verbose = false) {
    using namespace mertens_dr;
    if (X < 1) return 0;
    if (X == 1) return 1;
    if (X == 2) return 0;
    if (X == 3) return -1;
    int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
        omp_get_max_threads()
#else
        1
#endif
    );
    if (X <= 100000LL) return DelégliseRivatEngine::compute_mertens(X, threads);

    int64 u = DelégliseRivatEngine::choose_sieve_limit(X, threads);
    double cx = DelégliseRivatEngine::choose_cx(X);
    int64 N = X / u;
    int64 A_max = (int64)(cx * std::sqrt((double)X)) + 1000;
    int64 mu_limit = std::max(A_max, N + 1000);

    if (X <= u) {
        SieveTable t(u, threads);
        return t.get_M(X);
    }

    auto wt0 = std::chrono::high_resolution_clock::now();
    // Hybrid: SieveTable builds mu_odd (retained) + mu/M up to mu_limit only.
    // No 1.2GB full M table; large M windows are streamed from mu_odd below.
    SieveTable stable(u, threads, mu_limit, true, mu_limit);
    const int8_t* mu_small = stable.mu.data();
    const int16_t* M_small = stable.data();
    const int8_t* mu_odd_ptr = stable.mu_odd_data();
    auto wt1 = std::chrono::high_resolution_clock::now();

    int64 n6 = N / 6, n3 = N / 3, n2 = N / 2;

    std::vector<TermW> terms;
    terms.reserve((size_t)(N / 3 + 16));
    for (int64 k = 1; k <= N; ++k) {
        if ((k % 2 == 0) || (k % 3 == 0)) continue;
        int8_t mk = mu_small[k];
        if (mk == 0) continue;
        TermW t; t.k = k; t.mu_k = mk;
        if (k <= n6) t.range = 1;
        else if (k <= n3) t.range = 2;
        else if (k <= n2) t.range = 3;
        else t.range = 4;
        terms.push_back(std::move(t));
    }

    // precompute per-term tasks/intervals + S2/C (parallel, same math as baseline)
    #pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
    for (long long ti = 0; ti < (long long)terms.size(); ++ti) {
        TermW& t = terms[(size_t)ti];
        int64 y = X / t.k;
        if (t.range == 4) {
            int64 A = (int64)(cx * std::sqrt((double)y));
            if (A >= y) A = y - 1; if (A < 1) A = 1;
            int64 kappa = y / (A + 1);
            int64 start = y / u + 1;
            t.C0 = kappa * (int64)M_small[A];
            t.S2_0 = eval_s2_single(y, A, mu_small);
            t.one = 1;
            if (kappa >= start) t.ivs.push_back({y, start, kappa, 0, 0, 1});
        } else if (t.range == 3) {
            int64 y2 = y / 2;
            int64 A = (int64)(cx * std::sqrt((double)y));
            int64 B = (int64)(cx * std::sqrt((double)y2));
            if (A >= y) A = y - 1; if (A < 1) A = 1;
            if (B >= y2) B = y2 - 1; if (B < 1) B = 1;
            int64 kappa = y / (A + 1);
            int64 kappa2 = y2 / (B + 1);
            int64 start_odd = y / u + 1; if ((start_odd & 1) == 0) ++start_odd;
            t.C0 = kappa * (int64)M_small[A] - kappa2 * (int64)M_small[B];
            t.S2_0 = eval_s2_combined(y, A, B, mu_small);
            if (kappa >= start_odd) t.ivs.push_back({y, start_odd, kappa, 1, 0, 1});
            int64 tk2 = 2 * kappa2;
            if (tk2 > kappa) {
                int64 s = kappa + 1; if ((s & 1) != 0) ++s;
                if (s <= tk2) t.ivs.push_back({y, s, tk2, 2, 0, -1});
            } else if (tk2 < kappa) {
                int64 s = tk2 + 1; if ((s & 1) != 0) ++s;
                if (s <= kappa) t.ivs.push_back({y, s, kappa, 2, 0, 1});
            }
        } else if (t.range == 2) {
            // taskA comb2(y)
            {
                int64 y2 = y / 2;
                int64 A = (int64)(cx * std::sqrt((double)y));
                int64 B = (int64)(cx * std::sqrt((double)y2));
                if (A >= y) A = y - 1; if (A < 1) A = 1;
                if (B >= y2) B = y2 - 1; if (B < 1) B = 1;
                int64 kappa = y / (A + 1);
                int64 kappa2 = y2 / (B + 1);
                int64 start_odd = y / u + 1; if ((start_odd & 1) == 0) ++start_odd;
                t.C0 = kappa * (int64)M_small[A] - kappa2 * (int64)M_small[B];
                t.S2_0 = eval_s2_combined(y, A, B, mu_small);
                if (kappa >= start_odd) t.ivs.push_back({y, start_odd, kappa, 1, 0, 1});
                int64 tk2 = 2 * kappa2;
                if (tk2 > kappa) {
                    int64 s = kappa + 1; if ((s & 1) != 0) ++s;
                    if (s <= tk2) t.ivs.push_back({y, s, tk2, 2, 0, -1});
                } else if (tk2 < kappa) {
                    int64 s = tk2 + 1; if ((s & 1) != 0) ++s;
                    if (s <= kappa) t.ivs.push_back({y, s, kappa, 2, 0, 1});
                }
            }
            // taskB single(y/3)
            {
                int64 y3 = y / 3;
                int64 A = (int64)(cx * std::sqrt((double)y3));
                if (A >= y3) A = y3 - 1; if (A < 1) A = 1;
                int64 kappa = y3 / (A + 1);
                int64 start = y3 / u + 1;
                t.C1 = kappa * (int64)M_small[A];
                t.S2_1 = eval_s2_single(y3, A, mu_small);
                t.one1 = 1;
                if (kappa >= start) t.ivs.push_back({y3, start, kappa, 0, 1, 1});
            }
        } else { // range 1: comb2(y)-comb2(y/3)
            {
                int64 y2 = y / 2;
                int64 A = (int64)(cx * std::sqrt((double)y));
                int64 B = (int64)(cx * std::sqrt((double)y2));
                if (A >= y) A = y - 1; if (A < 1) A = 1;
                if (B >= y2) B = y2 - 1; if (B < 1) B = 1;
                int64 kappa = y / (A + 1);
                int64 kappa2 = y2 / (B + 1);
                int64 start_odd = y / u + 1; if ((start_odd & 1) == 0) ++start_odd;
                t.C0 = kappa * (int64)M_small[A] - kappa2 * (int64)M_small[B];
                t.S2_0 = eval_s2_combined(y, A, B, mu_small);
                if (kappa >= start_odd) t.ivs.push_back({y, start_odd, kappa, 1, 0, 1});
                int64 tk2 = 2 * kappa2;
                if (tk2 > kappa) {
                    int64 s = kappa + 1; if ((s & 1) != 0) ++s;
                    if (s <= tk2) t.ivs.push_back({y, s, tk2, 2, 0, -1});
                } else if (tk2 < kappa) {
                    int64 s = tk2 + 1; if ((s & 1) != 0) ++s;
                    if (s <= kappa) t.ivs.push_back({y, s, kappa, 2, 0, 1});
                }
            }
            {
                int64 y3 = y / 3, y6 = y / 6;
                int64 A = (int64)(cx * std::sqrt((double)y3));
                int64 B = y6 > 0 ? (int64)(cx * std::sqrt((double)y6)) : 1;
                if (A >= y3) A = y3 - 1; if (A < 1) A = 1;
                if (y6 > 0) { if (B >= y6) B = y6 - 1; if (B < 1) B = 1; }
                else B = 1;
                int64 kappa = y3 / (A + 1);
                int64 kappa2 = y6 > 0 ? y6 / (B + 1) : 0;
                int64 start_odd = y3 / u + 1; if ((start_odd & 1) == 0) ++start_odd;
                t.C1 = (y6 > 0 ? (kappa * (int64)M_small[A] - kappa2 * (int64)M_small[B]) : (kappa * (int64)M_small[A]));
                t.S2_1 = (y6 > 0 ? eval_s2_combined(y3, A, B, mu_small) : eval_s2_single(y3, A, mu_small));
                if (kappa >= start_odd) t.ivs.push_back({y3, start_odd, kappa, 1, 1, 1});
                if (y6 > 0) {
                    int64 tk2 = 2 * kappa2;
                    if (tk2 > kappa) {
                        int64 s = kappa + 1; if ((s & 1) != 0) ++s;
                        if (s <= tk2) t.ivs.push_back({y3, s, tk2, 2, 1, -1});
                    } else if (tk2 < kappa) {
                        int64 s = tk2 + 1; if ((s & 1) != 0) ++s;
                        if (s <= kappa) t.ivs.push_back({y3, s, kappa, 2, 1, 1});
                    }
                }
            }
        }
    }

    // streaming M windows over [1..u] built from retained mu_odd (no per-window sieving)
    int64 B = block_M;
    if (B < 65536) B = 65536;
    B = (B + 3) & ~3LL; // quad-align for branchless fill

    std::vector<int16_t> M_win((size_t)B);

    int64 base_offset = (int64)M_small[mu_limit]; // M(mu_limit)
    int64 run_offset = base_offset;
    auto wt2 = std::chrono::high_resolution_clock::now();
    double t_consume = 0, t_build = 0;

    for (int64 L = 1; L <= u; L += B) {
        int64 R = std::min(u, L + B - 1);
        bool is_small = (R <= mu_limit);
        bool is_straddle = (L <= mu_limit && R > mu_limit);
        const int16_t* win_ptr = nullptr;
        int64 win_L = L;
        // build M window from retained mu_odd (no sieving)
        int64 sL = std::max(L, mu_limit + 1);
        if (!is_small) {
            int64 len = R - sL + 1;
            if ((int64)M_win.size() < len) { M_win.resize((size_t)len); }
            auto ts0 = std::chrono::high_resolution_clock::now();
            int64 run = run_offset;
            int16_t* Mp = M_win.data();
            int64 n = sL;
            for (; n <= R && (n & 3) != 1; ++n) {
                int8_t m;
                if (n & 1) m = mu_odd_ptr[(n + 1) >> 1];
                else if ((n >> 1) & 1) m = (int8_t)-mu_odd_ptr[((n >> 1) + 1) >> 1];
                else m = 0;
                run += m;
                Mp[n - sL] = (int16_t)run;
            }
            for (; n + 3 <= R; n += 4) {
                int64 m = (n - 1) >> 2; // n=4m+1
                int8_t m1 = mu_odd_ptr[2 * m + 1];
                int8_t m2 = (int8_t)-mu_odd_ptr[m + 1];
                int8_t m3 = mu_odd_ptr[2 * m + 2];
                int16_t s0 = (int16_t)(run += m1);
                int16_t s1 = (int16_t)(run += m2);
                int16_t s2 = (int16_t)(run += m3);
                int16_t s3 = (int16_t)run; // 4m+4 divisible by 4
                uint64_t quad = (uint64_t)(uint16_t)s0
                    | ((uint64_t)(uint16_t)s1 << 16)
                    | ((uint64_t)(uint16_t)s2 << 32)
                    | ((uint64_t)(uint16_t)s3 << 48);
                std::memcpy(&Mp[n - sL], &quad, sizeof(quad));
            }
            for (; n <= R; ++n) {
                int8_t m;
                if (n & 1) m = mu_odd_ptr[(n + 1) >> 1];
                else if ((n >> 1) & 1) m = (int8_t)-mu_odd_ptr[((n >> 1) + 1) >> 1];
                else m = 0;
                run += m;
                Mp[n - sL] = (int16_t)run;
            }
            auto ts1 = std::chrono::high_resolution_clock::now();
            t_build += std::chrono::duration<double>(ts1-ts0).count();
            win_ptr = M_win.data();
            win_L = sL;
        }
        // consume: hoisted small/large/straddle + reciprocal bounds (no per-iter branch, no int64 div)
        auto tc0 = std::chrono::high_resolution_clock::now();
        double invL = 1.0 / (double)L;
        double invR1 = 1.0 / (double)(R + 1);
        #pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
        for (long long ti = 0; ti < (long long)terms.size(); ++ti) {
            TermW& t = terms[(size_t)ti];
            for (const S1Interval& iv : t.ivs) {
                int64 y = iv.y;
                double dy = (double)y;
                // FP bounds + exact correction with 64-bit mul (no div)
                // Products (nhi+1)*L, nlo*(R+1) are ~y+L <= ~1e17+1.5G < 2^63,
                // so 64-bit correction stays safe through 1e17.
                int64 nhi = (int64)(dy * invL);
                int64 nlo = (int64)(dy * invR1) + 1;
                // correct nhi: want floor(y/L) (products < 2^63 for X<=1e17, no 128-bit needed)
                while ((nhi + 1) * L <= y) ++nhi;
                while (nhi * L > y) --nhi;
                while (nlo > 0 && (nlo - 1) * (R + 1) > y) --nlo;
                while (nlo * (R + 1) <= y) ++nlo;
                if (nlo < iv.lo) nlo = iv.lo;
                if (nhi > iv.hi) nhi = iv.hi;
                if (nlo > nhi) continue;
                int64 part = 0;
                if (is_small) {
                    if (iv.mode == 0) {
                        for (int64 n = nlo; n <= nhi; ++n) part += (int64)M_small[fast_div(y, dy, n)];
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    } else if (iv.mode == 1) {
                        int64 n = nlo; if ((n & 1) == 0) ++n;
                        for (; n <= nhi; n += 2) part += (int64)M_small[fast_div(y, dy, n)];
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    } else {
                        int64 n = nlo; if ((n & 1) != 0) ++n;
                        for (; n <= nhi; n += 2) part += (int64)M_small[fast_div(y, dy, n)];
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    }
                } else if (!is_straddle) {
                    // large block: q in [L,R] always > mu_limit, direct win lookup
                    if (iv.mode == 0) {
                        for (int64 n = nlo; n <= nhi; ++n) part += (int64)win_ptr[fast_div(y, dy, n) - win_L];
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    } else if (iv.mode == 1) {
                        int64 n = nlo; if ((n & 1) == 0) ++n;
                        for (; n <= nhi; n += 2) part += (int64)win_ptr[fast_div(y, dy, n) - win_L];
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    } else {
                        int64 n = nlo; if ((n & 1) != 0) ++n;
                        for (; n <= nhi; n += 2) part += (int64)win_ptr[fast_div(y, dy, n) - win_L];
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    }
                } else {
                    // straddle (single block): keep exact branch
                    if (iv.mode == 0) {
                        for (int64 n = nlo; n <= nhi; ++n) {
                            int64 q = fast_div(y, dy, n);
                            part += (q <= mu_limit) ? (int64)M_small[q] : (int64)win_ptr[q - win_L];
                        }
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    } else if (iv.mode == 1) {
                        int64 n = nlo; if ((n & 1) == 0) ++n;
                        for (; n <= nhi; n += 2) {
                            int64 q = fast_div(y, dy, n);
                            part += (q <= mu_limit) ? (int64)M_small[q] : (int64)win_ptr[q - win_L];
                        }
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    } else {
                        int64 n = nlo; if ((n & 1) != 0) ++n;
                        for (; n <= nhi; n += 2) {
                            int64 q = fast_div(y, dy, n);
                            part += (q <= mu_limit) ? (int64)M_small[q] : (int64)win_ptr[q - win_L];
                        }
                        if (iv.acc == 0) t.s1a += (iv.sign > 0 ? part : -part);
                        else t.s1b += (iv.sign > 0 ? part : -part);
                    }
                }
            }
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        t_consume += std::chrono::duration<double>(tc1-tc0).count();
        if (!is_small) {
            // advance run_offset by block sum: M_win last - run_offset
            int64 len = R - sL + 1;
            run_offset = (int64)M_win[(size_t)(len - 1)];
        }
    }
    auto wt3 = std::chrono::high_resolution_clock::now();
    if (verbose) {
        double t_small = std::chrono::duration<double>(wt1-wt0).count();
        double t_pre = std::chrono::duration<double>(wt2-wt1).count();
        double t_rest = std::chrono::duration<double>(wt3-wt2).count();
        std::cerr << "[win] mu_odd=" << t_small << "s pre(S2)=" << t_pre << "s buildM=" << t_build
                  << "s consume=" << t_consume << "s rest=" << (t_rest-t_build-t_consume) << "s terms=" << terms.size() << " u=" << u << std::endl;
    }

    int64 total = 0;
    #pragma omp parallel for reduction(+:total) schedule(dynamic, 64) num_threads(threads)
    for (long long ti = 0; ti < (long long)terms.size(); ++ti) {
        const TermW& t = terms[(size_t)ti];
        int64 y = X / t.k;
        int64 term = 0;
        if (t.range == 4) {
            term = t.one - t.s1a + t.C0 - t.S2_0;
        } else if (t.range == 3) {
            term = -t.s1a + t.C0 - t.S2_0;
        } else if (t.range == 2) {
            int64 c0 = -t.s1a + t.C0 - t.S2_0;
            int64 c1 = t.one1 - t.s1b + t.C1 - t.S2_1;
            term = c0 - c1;
        } else {
            int64 c0 = -t.s1a + t.C0 - t.S2_0;
            int64 c1 = -t.s1b + t.C1 - t.S2_1;
            term = c0 - c1;
        }
        total += (int64)t.mu_k * term;
        (void)y;
    }
    return total;
}

} // namespace mertens_windowed

#endif
