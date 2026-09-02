#ifndef MERTENS_DR_HPP
#define MERTENS_DR_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace mertens_dr {

using int64 = long long;
using uint64 = unsigned long long;
using int128 = __int128_t;
using uint128 = __uint128_t;

/**
 * Exact integer floor square root.
 */
inline int64 isqrt(int64 n) noexcept {
    if (n <= 0) return 0;
    int64 s = static_cast<int64>(std::sqrt(static_cast<double>(n)));
    while ((s + 1) * (s + 1) <= n) ++s;
    while (s * s > n) --s;
    return s;
}

/**
 * Exact floor division: Uses double precision division when y <= 2^53,
 * and falls back to 64-bit integer division when y > 2^53 (e.g. X >= 10^17).
 */
inline int64 fast_div(int64 y, double dy, int64 d) noexcept {
    if (y > 9007199254740992LL) {
        return y / d;
    } else {
        return static_cast<int64>(dy / static_cast<double>(d));
    }
}

/**
 * High-performance parallel segmented sieve for Mobius and Mertens.
 * Uses direct int16_t flat table for M(n) (since |M(n)| < 32768 for all n <= 7.6 * 10^9, Hurst 2018).
 * Takes only 2 bytes per entry with direct LDRH single-cycle memory load.
 */
class SieveTable {
public:
    int64 u;
    std::vector<int8_t> mu;
    std::vector<int16_t> M; // 2 bytes per entry, zero indirection

    explicit SieveTable(int64 limit, int num_threads = 0) : u(limit) {
        int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );

        mu.assign(static_cast<size_t>(u + 1), 0);
        M.assign(static_cast<size_t>(u + 1), 0);

        build_sieve(threads);
    }

    inline int32_t get_M(int64 t) const noexcept {
        return static_cast<int32_t>(M[t]);
    }

    inline int16_t operator()(int64 t) const noexcept {
        return M[t];
    }

    inline int8_t get_mu(int64 t) const noexcept {
        return mu[t];
    }

    inline const int16_t* data() const noexcept {
        return M.data();
    }

private:
    void build_sieve(int threads) {
        int64 half_u = (u + 1) / 2;

        // 1. Base primes up to sqrt(u)
        int sqrt_u = static_cast<int>(std::sqrt(static_cast<double>(u))) + 1;
        std::vector<int> primes;
        std::vector<uint8_t> is_p(sqrt_u + 1, 1);
        for (int i = 2; i <= sqrt_u; ++i) {
            if (is_p[i]) {
                if (i > 2) primes.push_back(i);
                for (int j = i * i; j <= sqrt_u; j += i) is_p[j] = 0;
            }
        }

        // 2. Parallel segmented sieve over odd numbers
        std::vector<int8_t> mu_odd(static_cast<size_t>(half_u + 1), 1);
        mu_odd[0] = 0;

        const int64 BLOCK_SIZE = 131072; // 128K entries fits in L2 cache
        int64 num_blocks = (half_u + BLOCK_SIZE - 1) / BLOCK_SIZE;

        #pragma omp parallel num_threads(threads)
        {
            std::vector<int32_t> rem(BLOCK_SIZE);
            #pragma omp for schedule(dynamic, 4)
            for (int64 b = 0; b < num_blocks; ++b) {
                int64 low_idx = b * BLOCK_SIZE + 1;
                int64 high_idx = std::min(half_u, (b + 1) * BLOCK_SIZE);
                int64 len = high_idx - low_idx + 1;
                if (len <= 0) continue;

                for (int64 i = 0; i < len; ++i) {
                    rem[i] = static_cast<int32_t>(2 * (low_idx + i) - 1);
                }

                int8_t* __restrict__ b_mu = &mu_odd[low_idx];

                for (int p : primes) {
                    int64 p2 = static_cast<int64>(p) * p;
                    int64 low_val = 2 * low_idx - 1;

                    int64 start_val = ((low_val + p - 1) / p) * p;
                    if (start_val % 2 == 0) start_val += p;
                    int64 start_i = (start_val - low_val) / 2;

                    int64 start_sq_val = ((low_val + p2 - 1) / p2) * p2;
                    if (start_sq_val % 2 == 0) start_sq_val += p2;
                    int64 start_sq_i = (start_sq_val - low_val) / 2;

                    for (int64 i = start_sq_i; i < len; i += p2) {
                        b_mu[i] = 0;
                    }

                    for (int64 i = start_i; i < len; i += p) {
                        if (b_mu[i] != 0) {
                            b_mu[i] = -b_mu[i];
                            rem[i] /= p;
                            while (rem[i] % p == 0) {
                                rem[i] /= p;
                                b_mu[i] = 0;
                            }
                        }
                    }
                }

                for (int64 i = 0; i < len; ++i) {
                    if (b_mu[i] != 0 && rem[i] > 1) {
                        b_mu[i] = -b_mu[i];
                    }
                }
            }
        }

        // 3. Parallel expansion to full mu
        int8_t* __restrict__ mu_ptr = mu.data();
        const int8_t* __restrict__ mu_odd_ptr = mu_odd.data();

        #pragma omp parallel for schedule(static, 65536) num_threads(threads)
        for (int64 i = 1; i <= u; ++i) {
            int8_t m;
            if (i & 1) {
                m = mu_odd_ptr[(i + 1) >> 1];
            } else if ((i >> 1) & 1) {
                m = -mu_odd_ptr[((i >> 1) + 1) >> 1];
            } else {
                m = 0;
            }
            mu_ptr[i] = m;
        }

        // 4. Parallel prefix sum for M
        int16_t* __restrict__ M_ptr = M.data();
        int num_t = threads;
        std::vector<int32_t> block_sums(num_t + 1, 0);

        #pragma omp parallel num_threads(num_t)
        {
            int tid = omp_get_thread_num();
            int64 chunk = (u + num_t - 1) / num_t;
            int64 start = tid * chunk + 1;
            int64 end = std::min(u, (tid + 1) * chunk);

            int32_t local_sum = 0;
            for (int64 i = start; i <= end; ++i) {
                local_sum += mu_ptr[i];
            }
            block_sums[tid + 1] = local_sum;
        }

        for (int i = 1; i <= num_t; ++i) {
            block_sums[i] += block_sums[i - 1];
        }

        #pragma omp parallel num_threads(num_t)
        {
            int tid = omp_get_thread_num();
            int64 chunk = (u + num_t - 1) / num_t;
            int64 start = tid * chunk + 1;
            int64 end = std::min(u, (tid + 1) * chunk);

            int32_t run_sum = block_sums[tid];
            for (int64 i = start; i <= end; ++i) {
                run_sum += mu_ptr[i];
                M_ptr[i] = static_cast<int16_t>(run_sum);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// LUT Lookup Tables for S2 Piecewise Reduction (Zero Integer Div / Mod)
// ---------------------------------------------------------------------------
alignas(16) static constexpr int8_t LUT_P1[12] = {0, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 2};
alignas(16) static constexpr int8_t LUT_P2[12] = {0, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 2};
alignas(16) static constexpr int8_t LUT_P3[12] = {0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1};
alignas(16) static constexpr int8_t LUT_P4[12] = {0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0};

alignas(16) static constexpr int8_t LUT_S1[12] = {0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4};
alignas(16) static constexpr int8_t LUT_S2[12] = {0, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3};

struct Piece1 { static inline int64 eval(int64 q) noexcept { int64 k = q / 12; int64 r = q % 12; return 2 * k + LUT_P1[r]; } };
struct Piece2 { static inline int64 eval(int64 q) noexcept { int64 k = q / 12; int64 r = q % 12; return 3 * k + LUT_P2[r]; } };
struct Piece3 { static inline int64 eval(int64 q) noexcept { int64 k = q / 12; int64 r = q % 12; return 1 * k + LUT_P3[r]; } };
struct Piece4 { static inline int64 eval(int64 q) noexcept { int64 k = q / 12; int64 r = q % 12; return -1 * k + LUT_P4[r]; } };
struct Piece5 { static inline int64 eval(int64 q) noexcept { return (q >> 2) + (q & 1); } };
struct Piece6 { static inline int64 eval(int64 q) noexcept { return (q & 1); } };
struct Piece7 { static inline int64 eval(int64 q) noexcept { return (q + 1) >> 1; } };
struct Piece8 { static inline int64 eval(int64 q) noexcept { return q; } };

struct SinglePiece1 { static inline int64 eval(int64 q) noexcept { int64 k = q / 12; int64 r = q % 12; return 4 * k + LUT_S1[r]; } };
struct SinglePiece2 { static inline int64 eval(int64 q) noexcept { int64 k = q / 12; int64 r = q % 12; return 2 * k + LUT_S2[r]; } };
struct SinglePiece3 { static inline int64 eval(int64 q) noexcept { return (q + 1) >> 1; } };
struct SinglePiece4 { static inline int64 eval(int64 q) noexcept { return q; } };

/**
 * Branchless, Modulo-Free, NEON-Vectorized S2 Interval Runner with LUT Summand.
 */
template <typename PieceType>
inline int64 run_s2_fast(int64 j_start, int64 j_end, double dy, const int8_t* __restrict mu_ptr) noexcept {
    if (j_start > j_end) return 0;
    int64 sum = 0;

    int64 m_start = (j_start + 5) / 6;
    int64 m_end = j_end / 6;

    // Head Boundary
    int64 head_end = std::min(j_end, m_start * 6);
    for (int64 j = j_start; j <= head_end; ++j) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = mu_ptr[j];
            if (m != 0) {
                int64 q = static_cast<int64>(dy / static_cast<double>(j));
                sum += static_cast<int64>(m) * PieceType::eval(q);
            }
        }
    }

#if defined(__ARM_NEON)
    // Main 2-way unrolled body with NEON SIMD vector division
    float64x2_t v_dy = vdupq_n_f64(dy);
    int64 m = m_start;
    for (; m + 1 < m_end; m += 2) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int64 j3 = 6 * (m + 1) + 1; int64 j4 = 6 * (m + 1) + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        int8_t m3 = mu_ptr[j3]; int8_t m4 = mu_ptr[j4];

        if ((m1 | m2 | m3 | m4) != 0) {
            float64x2_t v_ja = {static_cast<double>(j1), static_cast<double>(j2)};
            float64x2_t v_jb = {static_cast<double>(j3), static_cast<double>(j4)};
            float64x2_t v_qa = vdivq_f64(v_dy, v_ja);
            float64x2_t v_qb = vdivq_f64(v_dy, v_jb);

            int64 q1 = static_cast<int64>(vgetq_lane_f64(v_qa, 0));
            int64 q2 = static_cast<int64>(vgetq_lane_f64(v_qa, 1));
            int64 q3 = static_cast<int64>(vgetq_lane_f64(v_qb, 0));
            int64 q4 = static_cast<int64>(vgetq_lane_f64(v_qb, 1));

            if (m1) sum += static_cast<int64>(m1) * PieceType::eval(q1);
            if (m2) sum += static_cast<int64>(m2) * PieceType::eval(q2);
            if (m3) sum += static_cast<int64>(m3) * PieceType::eval(q3);
            if (m4) sum += static_cast<int64>(m4) * PieceType::eval(q4);
        }
    }
    for (; m < m_end; ++m) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        if (m1) sum += static_cast<int64>(m1) * PieceType::eval(static_cast<int64>(dy / static_cast<double>(j1)));
        if (m2) sum += static_cast<int64>(m2) * PieceType::eval(static_cast<int64>(dy / static_cast<double>(j2)));
    }
#else
    for (int64 m = m_start; m < m_end; ++m) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        if (m1) sum += static_cast<int64>(m1) * PieceType::eval(static_cast<int64>(dy / static_cast<double>(j1)));
        if (m2) sum += static_cast<int64>(m2) * PieceType::eval(static_cast<int64>(dy / static_cast<double>(j2)));
    }
#endif

    // Tail Boundary
    int64 tail_start = std::max(j_start, m_end * 6 + 1);
    for (int64 j = tail_start; j <= j_end; ++j) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = mu_ptr[j];
            if (m != 0) {
                int64 q = static_cast<int64>(dy / static_cast<double>(j));
                sum += static_cast<int64>(m) * PieceType::eval(q);
            }
        }
    }

    return sum;
}

/**
 * Fast stream evaluator for S2 combined range: S2(y) - S2(y/2).
 */
inline int64 eval_s2_combined(int64 y, int64 A, int64 B, const int8_t* __restrict mu_ptr) noexcept {
    double dy = static_cast<double>(y);
    int64 b6 = B / 6;
    int64 a6 = A / 6;
    int64 b3 = B / 3;
    int64 a3 = A / 3;
    int64 b2 = B / 2;
    int64 a2 = A / 2;

    int64 sum = 0;
    sum += run_s2_fast<Piece1>(1, b6, dy, mu_ptr);
    sum += run_s2_fast<Piece2>(b6 + 1, a6, dy, mu_ptr);
    sum += run_s2_fast<Piece3>(a6 + 1, b3, dy, mu_ptr);
    sum += run_s2_fast<Piece4>(b3 + 1, a3, dy, mu_ptr);
    sum += run_s2_fast<Piece5>(a3 + 1, b2, dy, mu_ptr);
    sum += run_s2_fast<Piece6>(b2 + 1, a2, dy, mu_ptr);
    sum += run_s2_fast<Piece7>(a2 + 1, B, dy, mu_ptr);
    sum += run_s2_fast<Piece8>(B + 1, A, dy, mu_ptr);

    return sum;
}

/**
 * Fast stream evaluator for S2 single range: S2(y).
 */
inline int64 eval_s2_single(int64 y, int64 A, const int8_t* __restrict mu_ptr) noexcept {
    double dy = static_cast<double>(y);
    int64 a6 = A / 6;
    int64 a3 = A / 3;
    int64 a2 = A / 2;

    int64 sum = 0;
    sum += run_s2_fast<SinglePiece1>(1, a6, dy, mu_ptr);
    sum += run_s2_fast<SinglePiece2>(a6 + 1, a3, dy, mu_ptr);
    sum += run_s2_fast<SinglePiece3>(a3 + 1, a2, dy, mu_ptr);
    sum += run_s2_fast<SinglePiece4>(a2 + 1, A, dy, mu_ptr);

    return sum;
}

/**
 * Master Deléglise-Rivat Mertens Solver.
 */
class DelégliseRivatEngine {
public:
    /**
     * Dynamically chooses optimal balance parameter cx based on hardware SIMD and memory characteristics.
     */
    static inline double choose_cx(int64 X) noexcept {
        if (X >= 1000000000000000LL) return 1.15;
        if (X >= 10000000000000LL) return 0.95;
        return 0.70;
    }

    /**
     * Dynamically chooses optimal sieve cutoff u for target X.
     */
    static inline int64 choose_sieve_limit(int64 X, int threads) {
        (void)threads;
        if (X <= 50000000LL) {
            return X; // Direct linear sieve for X <= 50M
        }

        double loglogX = std::log(std::max(2.0, std::log(static_cast<double>(X))));
        double fx = 0.95;
        if (X >= 1000000000000000LL) {
            fx = 1.25;
        }

        int64 u = static_cast<int64>(fx * std::pow(static_cast<double>(X) / loglogX, 2.0 / 3.0));
        int64 S = isqrt(X);
        if (u < 3 * S) u = 3 * S;
        
        // Cap u at 600 Million (~1.2 GB RAM) for maximum multi-core cache throughput
        return std::min(u, 600000000LL);
    }

    /**
     * Computes the Mertens Function M(X) = sum_{n=1}^X mu(n) in O(X^(2/3))
     * using the Mod-6 Full Wheel Deléglise-Rivat reduction.
     */
    static int64 compute_mertens(int64 X, int num_threads = 0) {
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

        if (X <= 50000000LL) {
            int64 u = X;
            int half_u = static_cast<int>((u + 1) / 2);
            std::vector<int8_t> mu_odd(half_u + 1, 0);
            std::vector<int> primes;
            primes.reserve(static_cast<size_t>(u / 10));
            std::vector<uint8_t> is_prime(half_u + 1, 1);
            mu_odd[0] = 0; mu_odd[1] = 1;

            for (int i = 2; 2 * i - 1 <= u; ++i) {
                int num = 2 * i - 1;
                if (is_prime[i]) { primes.push_back(num); mu_odd[i] = -1; }
                for (size_t j = 0; j < primes.size(); ++j) {
                    int p = primes[j];
                    int64 prod = static_cast<int64>(num) * p;
                    if (prod > u) break;
                    int idx = static_cast<int>((prod + 1) / 2);
                    is_prime[idx] = 0;
                    if (num % p == 0) { mu_odd[idx] = 0; break; }
                    else { mu_odd[idx] = -mu_odd[i]; }
                }
            }
            int64 run_sum = 0;
            for (int64 i = 1; i <= u; ++i) {
                int8_t m = (i % 2 != 0) ? mu_odd[(i + 1) / 2] : (((i / 2) % 2 != 0) ? -mu_odd[((i / 2) + 1) / 2] : 0);
                run_sum += m;
            }
            return run_sum;
        }

        int64 u = choose_sieve_limit(X, threads);

        // Precompute sieve table
        SieveTable table(u, threads);

        if (X <= u) {
            return table.get_M(X);
        }

        const double cx = choose_cx(X);
        const int64 N = X / u;
        const int16_t* __restrict M_ptr = table.data();
        const int8_t* __restrict mu_ptr = table.mu.data();

        int64 n6 = N / 6;
        int64 n3 = N / 3;
        int64 n2 = N / 2;

        std::vector<int64> k_p1; // k <= N/6: S(y) - S(y/2) - S(y/3) + S(y/6)
        std::vector<int64> k_p2; // N/6 < k <= N/3: S(y) - S(y/2) - S(y/3)
        std::vector<int64> k_p3; // N/3 < k <= N/2: S(y) - S(y/2)
        std::vector<int64> k_p4; // N/2 < k <= N: S(y)

        for (int64 k = 1; k <= N; ++k) {
            if (k % 2 == 0 || k % 3 == 0) continue;
            if (mu_ptr[k] != 0) {
                if (k <= n6) k_p1.push_back(k);
                else if (k <= n3) k_p2.push_back(k);
                else if (k <= n2) k_p3.push_back(k);
                else k_p4.push_back(k);
            }
        }

        int64 total_M = 0;

        auto eval_single_S = [&](int64 y) noexcept -> int64 {
            int64 A = static_cast<int64>(cx * std::sqrt(static_cast<double>(y)));
            if (A >= y) A = y - 1;
            if (A < 1) A = 1;
            int64 kappa_y = y / (A + 1);

            int64 S1 = 0;
            int64 start_n = y / u + 1;
            double dy = static_cast<double>(y);
            for (int64 n = start_n; n <= kappa_y; ++n) {
                S1 += M_ptr[static_cast<int64>(dy / static_cast<double>(n))];
            }
            int64 S2 = eval_s2_single(y, A, mu_ptr);
            return 1 - S1 + kappa_y * static_cast<int64>(M_ptr[A]) - S2;
        };

        auto eval_comb2 = [&](int64 y, int64 y2) noexcept -> int64 {
            int64 A = static_cast<int64>(cx * std::sqrt(static_cast<double>(y)));
            int64 B = static_cast<int64>(cx * std::sqrt(static_cast<double>(y2)));
            if (A >= y) A = y - 1;
            if (B >= y2) B = y2 - 1;
            if (A < 1) A = 1;
            if (B < 1) B = 1;

            int64 kappa_y = y / (A + 1);
            int64 kappa_y2 = y2 / (B + 1);
            int64 two_kappa_y2 = 2 * kappa_y2;

            int64 S1_diff = 0;
            int64 start_odd = (y / u + 1);
            if (start_odd % 2 == 0) ++start_odd;

            double dy = static_cast<double>(y);
            int64 n = start_odd;

#if defined(__ARM_NEON)
            float64x2_t v_dy = vdupq_n_f64(dy);
            for (; n + 15 <= kappa_y; n += 8) {
                int64 n_pref = n + 16;
                int64 q_p0 = static_cast<int64>(dy / static_cast<double>(n_pref));
                int64 q_p1 = static_cast<int64>(dy / static_cast<double>(n_pref + 4));
                __builtin_prefetch(&M_ptr[q_p0], 0, 3);
                __builtin_prefetch(&M_ptr[q_p1], 0, 3);

                float64x2_t v_n1 = {static_cast<double>(n), static_cast<double>(n + 2)};
                float64x2_t v_n2 = {static_cast<double>(n + 4), static_cast<double>(n + 6)};
                float64x2_t v_q1 = vdivq_f64(v_dy, v_n1);
                float64x2_t v_q2 = vdivq_f64(v_dy, v_n2);

                int64 q0 = static_cast<int64>(vgetq_lane_f64(v_q1, 0));
                int64 q1 = static_cast<int64>(vgetq_lane_f64(v_q1, 1));
                int64 q2 = static_cast<int64>(vgetq_lane_f64(v_q2, 0));
                int64 q3 = static_cast<int64>(vgetq_lane_f64(v_q2, 1));

                S1_diff += M_ptr[q0] + M_ptr[q1] + M_ptr[q2] + M_ptr[q3];
            }
#endif
            for (; n <= kappa_y; n += 2) {
                S1_diff += M_ptr[static_cast<int64>(dy / static_cast<double>(n))];
            }

            if (two_kappa_y2 > kappa_y) {
                int64 start_even = kappa_y + 1;
                if (start_even % 2 != 0) ++start_even;
                for (int64 ne = start_even; ne <= two_kappa_y2; ne += 2) {
                    S1_diff -= M_ptr[static_cast<int64>(dy / static_cast<double>(ne))];
                }
            } else if (two_kappa_y2 < kappa_y) {
                int64 start_even = two_kappa_y2 + 1;
                if (start_even % 2 != 0) ++start_even;
                for (int64 ne = start_even; ne <= kappa_y; ne += 2) {
                    S1_diff += M_ptr[static_cast<int64>(dy / static_cast<double>(ne))];
                }
            }

            int64 S2_diff = eval_s2_combined(y, A, B, mu_ptr);
            return -S1_diff + (kappa_y * static_cast<int64>(M_ptr[A]) - kappa_y2 * static_cast<int64>(M_ptr[B])) - S2_diff;
        };

        // Range 1: k <= N/6 -> (S(y) - S(y/2)) - (S(y/3) - S(y/6))
        #pragma omp parallel for reduction(+:total_M) schedule(guided) num_threads(threads)
        for (size_t i = 0; i < k_p1.size(); ++i) {
            int64 k = k_p1[i];
            int8_t mu_k = mu_ptr[k];
            int64 y = X / k;
            int64 term = eval_comb2(y, y / 2) - eval_comb2(y / 3, y / 6);
            total_M += static_cast<int64>(mu_k) * term;
        }

        // Range 2: N/6 < k <= N/3 -> (S(y) - S(y/2)) - S(y/3)
        #pragma omp parallel for reduction(+:total_M) schedule(guided) num_threads(threads)
        for (size_t i = 0; i < k_p2.size(); ++i) {
            int64 k = k_p2[i];
            int8_t mu_k = mu_ptr[k];
            int64 y = X / k;
            int64 term = eval_comb2(y, y / 2) - eval_single_S(y / 3);
            total_M += static_cast<int64>(mu_k) * term;
        }

        // Range 3: N/3 < k <= N/2 -> S(y) - S(y/2)
        #pragma omp parallel for reduction(+:total_M) schedule(guided) num_threads(threads)
        for (size_t i = 0; i < k_p3.size(); ++i) {
            int64 k = k_p3[i];
            int8_t mu_k = mu_ptr[k];
            int64 y = X / k;
            int64 term = eval_comb2(y, y / 2);
            total_M += static_cast<int64>(mu_k) * term;
        }

        // Range 4: N/2 < k <= N -> S(y)
        #pragma omp parallel for reduction(+:total_M) schedule(guided) num_threads(threads)
        for (size_t i = 0; i < k_p4.size(); ++i) {
            int64 k = k_p4[i];
            int8_t mu_k = mu_ptr[k];
            int64 y = X / k;
            int64 term = eval_single_S(y);
            total_M += static_cast<int64>(mu_k) * term;
        }

        return total_M;
    }
};

} // namespace mertens_dr

#endif // MERTENS_DR_HPP
