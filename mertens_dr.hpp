#ifndef MERTENS_DR_HPP
#define MERTENS_DR_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <bit>
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
#ifdef _MSC_VER
using int128 = long long;
using uint128 = unsigned long long;
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#else
using int128 = __int128_t;
using uint128 = __uint128_t;
#endif

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
 * Derives mu(n) from odd-only mu table: mu_odd[k] = mu(2k-1).
 */
inline int8_t mu_from_odd(int64 n, const int8_t* __restrict__ mu_odd) noexcept {
    if (n & 1) {
        return mu_odd[(n + 1) >> 1];
    } else if ((n >> 1) & 1) {
        return static_cast<int8_t>(-mu_odd[((n >> 1) + 1) >> 1]);
    } else {
        return 0;
    }
}

/**
 * Shared byte-log segmented Mobius sieve (Hurst 5.1/5.4 style, upstream winner).
 * Fills mu_odd[1..half_u] where mu_odd[k] = mu(2k-1). mu_odd must be
 * pre-allocated with size >= half_u+1. Single-array accumulator, no rem[],
 * no per-hit division: weight w(p)=ceil(log2 p)|1, p^2 -> 0x80 flag.
 * Max acc for n<=1.5G is ~40 (<50 for u<=8B), safely below 0x80.
 */
inline void sieve_mu_odd_byte_log(int8_t* __restrict__ mu_odd, int64 half_u, int64 u,
                                  int threads) {
    // 1. Base primes up to sqrt(u)
    // NOTE: i/j must be 64-bit: int i*i overflows past u > 46340^2
    // (~2.147e9), corrupting is_p and segfaulting (found at u=2.4e9).
    int64 sqrt_u = static_cast<int64>(std::sqrt(static_cast<double>(u))) + 1;
    std::vector<int> primes;
    std::vector<uint8_t> prime_w;
    primes.reserve(8192);
    prime_w.reserve(8192);
    std::vector<uint8_t> is_p(static_cast<size_t>(sqrt_u + 1), 1);
    for (int64 i = 2; i <= sqrt_u; ++i) {
        if (is_p[static_cast<size_t>(i)]) {
            if (i > 2) {
                primes.push_back(static_cast<int>(i));
                prime_w.push_back(static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(i - 1)) | 1u));
            }
            for (int64 j = i * i; j <= sqrt_u; j += i) is_p[static_cast<size_t>(j)] = 0;
        }
    }

    const int64 BLOCK_SIZE = 131072; // 128K byte entries: L2-resident per thread
    int64 num_blocks = (half_u + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(dynamic, 4)
        for (int64 b = 0; b < num_blocks; ++b) {
            int64 low_idx = b * BLOCK_SIZE + 1;
            int64 high_idx = std::min(half_u, (b + 1) * BLOCK_SIZE);
            int64 len = high_idx - low_idx + 1;
            if (len <= 0) continue;

            int8_t* __restrict__ b_mu = &mu_odd[low_idx];
            std::memset(b_mu, 0, static_cast<size_t>(len));

            for (size_t pi = 0; pi < primes.size(); ++pi) {
                int64 p = primes[pi];
                uint8_t w = prime_w[pi];
                int64 p2 = p * p;
                int64 low_val = 2 * low_idx - 1;

                int64 start_val = ((low_val + p - 1) / p) * p;
                if (start_val % 2 == 0) start_val += p;
                int64 start_i = (start_val - low_val) / 2;

                int64 start_sq_val = ((low_val + p2 - 1) / p2) * p2;
                if (start_sq_val % 2 == 0) start_sq_val += p2;
                int64 start_sq_i = (start_sq_val - low_val) / 2;

                for (int64 i = start_sq_i; i < len; i += p2) {
                    b_mu[i] |= static_cast<int8_t>(0x80);
                }

                for (int64 i = start_i; i < len; i += p) {
                    int8_t a = b_mu[i];
                    if ((a & 0x80) == 0) b_mu[i] = static_cast<int8_t>(a + w);
                }
            }

            // In-cache finalize: log-sum -> mu via exact threshold compare.
            for (int64 i = 0; i < len; ++i) {
                int8_t a = b_mu[i];
                int8_t m;
                if (a & 0x80) {
                    m = 0;
                } else {
                    uint64_t v = static_cast<uint64_t>(2 * (low_idx + i) - 1);
                    unsigned f = std::bit_width(v) - 1; // floor(log2 v), v >= 1
                    int8_t s = (a & 1) ? -1 : 1;
                    m = (static_cast<unsigned>(a) > f) ? s : static_cast<int8_t>(-s);
                }
                b_mu[i] = m;
            }
        }
    }
    mu_odd[0] = 0;
    if (half_u >= 1) mu_odd[1] = 1; // value 1 has no prime factors (acc=0 would give -1)
}

/**
 * High-performance parallel segmented sieve for Mobius and Mertens.
 * Uses direct int16_t flat table for M(n) (since |M(n)| < 32768 for all n <= 7.6 * 10^9, Hurst 2018).
 * Takes only 2 bytes per entry with direct LDRH single-cycle memory load.
 * mu array is only kept up to mu_limit (A_max) to save memory bandwidth and RAM.
 */
class SieveTable {
public:
    int64 u;
    int64 mu_limit;
    int64 m_limit; // upper bound actually built in M (defaults to u; windowed path passes mu_limit)
    bool retain_mu_odd;
    std::vector<int8_t> mu;
    std::vector<int16_t> M; // 2 bytes per entry, zero indirection
    std::vector<int8_t> mu_odd_store; // populated only when retain_mu_odd=true

    explicit SieveTable(int64 limit, int num_threads = 0, int64 max_mu = 0,
                        bool retain_mu_odd_ = false, int64 max_M_ = -1) : u(limit), retain_mu_odd(retain_mu_odd_) {
        int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );

        mu_limit = (max_mu > 0 ? std::min(max_mu, u) : u);
        m_limit = (max_M_ >= 0 ? std::min(max_M_, u) : u);
        mu.assign(static_cast<size_t>(mu_limit + 1), 0);
        M.assign(static_cast<size_t>(m_limit + 1), 0);

        build_sieve(threads);
    }

    inline int32_t get_M(int64 t) const noexcept {
        return static_cast<int32_t>(M[t]);
    }

    inline int16_t operator()(int64 t) const noexcept {
        return M[t];
    }

    inline int8_t get_mu(int64 t) const noexcept {
        return (t <= mu_limit ? mu[t] : 0);
    }

    inline const int16_t* data() const noexcept {
        return M.data();
    }

    inline const int8_t* mu_odd_data() const noexcept {
        return mu_odd_store.empty() ? nullptr : mu_odd_store.data();
    }

    inline int64 mu_odd_size() const noexcept {
        return mu_odd_store.empty() ? 0 : static_cast<int64>(mu_odd_store.size()) - 1;
    }

private:
    void build_sieve(int threads) {
        int64 half_u = (u + 1) / 2;

        // 1+2. Shared byte-log segmented sieve (no rem[], no divisions).
        std::vector<int8_t> mu_odd(static_cast<size_t>(half_u + 1), 0);
        sieve_mu_odd_byte_log(mu_odd.data(), half_u, u, threads);

        // 3. Populate small mu table (only up to mu_limit)
        int8_t* __restrict__ mu_ptr = mu.data();
        const int8_t* __restrict mu_odd_ptr = mu_odd.data();
        for (int64 i = 1; i <= mu_limit; ++i) {
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

        // 4. Parallel direct prefix sum for M from mu_odd (zero mu array overhead)
        // Windowed path sets m_limit=mu_limit to skip the 1.2GB full table.
        int16_t* __restrict__ M_ptr = M.data();
        int num_t = threads;
        std::vector<int32_t> block_sums(num_t + 1, 0);

        #pragma omp parallel num_threads(num_t)
        {
            int tid = omp_get_thread_num();
            int64 chunk = (m_limit + num_t - 1) / num_t;
            chunk = (chunk + 3) & ~3LL;
            int64 start = tid * chunk + 1;
            int64 end = std::min(m_limit, (tid + 1) * chunk);

            int32_t local_sum = 0;
            for (int64 i = start; i <= end; ++i) {
                int8_t m;
                if (i & 1) {
                    m = mu_odd_ptr[(i + 1) >> 1];
                } else if ((i >> 1) & 1) {
                    m = -mu_odd_ptr[((i >> 1) + 1) >> 1];
                } else {
                    m = 0;
                }
                local_sum += m;
            }
            block_sums[tid + 1] = local_sum;
        }

        for (int i = 1; i <= num_t; ++i) {
            block_sums[i] += block_sums[i - 1];
        }

        #pragma omp parallel num_threads(num_t)
        {
            int tid = omp_get_thread_num();
            int64 chunk = (m_limit + num_t - 1) / num_t;
            chunk = (chunk + 3) & ~3LL;
            int64 start = tid * chunk + 1;
            int64 end = std::min(m_limit, (tid + 1) * chunk);

            int32_t run_sum = block_sums[tid];
            int64 i = start;
            for (; i + 3 <= end; i += 4) {
                int64 m = (i - 1) / 4;
                int8_t m1 = mu_odd_ptr[2 * m + 1];
                int8_t m2 = -mu_odd_ptr[m + 1];
                int8_t m3 = mu_odd_ptr[2 * m + 2];
                int16_t s0 = static_cast<int16_t>(run_sum += m1);
                int16_t s1 = static_cast<int16_t>(run_sum += m2);
                int16_t s2 = static_cast<int16_t>(run_sum += m3);
                int16_t s3 = static_cast<int16_t>(run_sum);

                uint64_t quad = (static_cast<uint16_t>(s0))
                              | (static_cast<uint64_t>(static_cast<uint16_t>(s1)) << 16)
                              | (static_cast<uint64_t>(static_cast<uint16_t>(s2)) << 32)
                              | (static_cast<uint64_t>(static_cast<uint16_t>(s3)) << 48);
                std::memcpy(&M_ptr[i], &quad, sizeof(quad));
            }
            for (; i <= end; ++i) {
                int8_t m;
                if (i & 1) {
                    m = mu_odd_ptr[(i + 1) >> 1];
                } else if ((i >> 1) & 1) {
                    m = -mu_odd_ptr[((i >> 1) + 1) >> 1];
                } else {
                    m = 0;
                }
                run_sum += m;
                M_ptr[i] = static_cast<int16_t>(run_sum);
            }
        }

        if (retain_mu_odd) {
            mu_odd_store = std::move(mu_odd);
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

struct Piece1 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq - k * 12; return 2 * static_cast<int64>(k) + LUT_P1[r]; } };
struct Piece2 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq - k * 12; return 3 * static_cast<int64>(k) + LUT_P2[r]; } };
struct Piece3 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq - k * 12; return 1 * static_cast<int64>(k) + LUT_P3[r]; } };
struct Piece4 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq - k * 12; return -1 * static_cast<int64>(k) + LUT_P4[r]; } };
struct Piece5 { static inline int64 eval(int64 q) noexcept { return (q >> 2) + (q & 1); } };
struct Piece6 { static inline int64 eval(int64 q) noexcept { return (q & 1); } };
struct Piece7 { static inline int64 eval(int64 q) noexcept { return (q + 1) >> 1; } };
struct Piece8 { static inline int64 eval(int64 q) noexcept { return q; } };

struct SinglePiece1 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq - k * 12; return 4 * static_cast<int64>(k) + LUT_S1[r]; } };
struct SinglePiece2 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq - k * 12; return 2 * static_cast<int64>(k) + LUT_S2[r]; } };
struct SinglePiece3 { static inline int64 eval(int64 q) noexcept { return (q + 1) >> 1; } };
struct SinglePiece4 { static inline int64 eval(int64 q) noexcept { return q; } };

/**
 * Branchless, Modulo-Free, NEON 4-Way Pipelined S2 Interval Runner with LUT Summand.
 * Merged: upstream tail-clamp + r=uq-k*12, local fast_div (>2^53 exact) + large_y
 * NEON bypass. For y>2^53 double division is inexact, so we fall back to
 * scalar integer division (correct for X>=1e17).
 */
template <typename PieceType>
inline int64 run_s2_fast(int64 j_start, int64 j_end, int64 y, double dy, const int8_t* __restrict mu_ptr) noexcept {
    if (j_start > j_end) return 0;
    int64 sum = 0;

    int64 m_start = (j_start + 5) / 6;
    int64 m_end = j_end / 6;

    // Head Boundary — exact via fast_div guard (>2^53 uses integer division)
    int64 head_end = std::min(j_end, m_start * 6);
    for (int64 j = j_start; j <= head_end; ++j) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = mu_ptr[j];
            if (m != 0) {
                int64 q = fast_div(y, dy, j);
                sum += static_cast<int64>(m) * PieceType::eval(q);
            }
        }
    }

    const bool large_y = (y > 9007199254740992LL);
#if defined(__ARM_NEON)
    if (!large_y) {
    // 4-way NEON pipelined division loop (processes 8 coprime-6 values per step)
    float64x2_t v_dy = vdupq_n_f64(dy);
    float64x2_t v_step24 = {24.0, 24.0};
    int64 m = m_start;

    float64x2_t v_j12 = {static_cast<double>(6 * m + 1),  static_cast<double>(6 * m + 5)};
    float64x2_t v_j34 = {static_cast<double>(6 * m + 7),  static_cast<double>(6 * m + 11)};
    float64x2_t v_j56 = {static_cast<double>(6 * m + 13), static_cast<double>(6 * m + 17)};
    float64x2_t v_j78 = {static_cast<double>(6 * m + 19), static_cast<double>(6 * m + 23)};

    for (; m + 3 < m_end; m += 4) {
        float64x2_t v_q12 = vdivq_f64(v_dy, v_j12);
        float64x2_t v_q34 = vdivq_f64(v_dy, v_j34);
        float64x2_t v_q56 = vdivq_f64(v_dy, v_j56);
        float64x2_t v_q78 = vdivq_f64(v_dy, v_j78);

        v_j12 = vaddq_f64(v_j12, v_step24);
        v_j34 = vaddq_f64(v_j34, v_step24);
        v_j56 = vaddq_f64(v_j56, v_step24);
        v_j78 = vaddq_f64(v_j78, v_step24);

        int64x2_t v_qi12 = vcvtq_s64_f64(v_q12);
        int64x2_t v_qi34 = vcvtq_s64_f64(v_q34);
        int64x2_t v_qi56 = vcvtq_s64_f64(v_q56);
        int64x2_t v_qi78 = vcvtq_s64_f64(v_q78);

        int64 j_base = 6 * m;
        int8_t m1 = mu_ptr[j_base + 1];  int8_t m2 = mu_ptr[j_base + 5];
        int8_t m3 = mu_ptr[j_base + 7];  int8_t m4 = mu_ptr[j_base + 11];
        int8_t m5 = mu_ptr[j_base + 13]; int8_t m6 = mu_ptr[j_base + 17];
        int8_t m7 = mu_ptr[j_base + 19]; int8_t m8 = mu_ptr[j_base + 23];

        if (m1) sum += static_cast<int64>(m1) * PieceType::eval(vgetq_lane_s64(v_qi12, 0));
        if (m2) sum += static_cast<int64>(m2) * PieceType::eval(vgetq_lane_s64(v_qi12, 1));
        if (m3) sum += static_cast<int64>(m3) * PieceType::eval(vgetq_lane_s64(v_qi34, 0));
        if (m4) sum += static_cast<int64>(m4) * PieceType::eval(vgetq_lane_s64(v_qi34, 1));
        if (m5) sum += static_cast<int64>(m5) * PieceType::eval(vgetq_lane_s64(v_qi56, 0));
        if (m6) sum += static_cast<int64>(m6) * PieceType::eval(vgetq_lane_s64(v_qi56, 1));
        if (m7) sum += static_cast<int64>(m7) * PieceType::eval(vgetq_lane_s64(v_qi78, 0));
        if (m8) sum += static_cast<int64>(m8) * PieceType::eval(vgetq_lane_s64(v_qi78, 1));
    }
    for (; m < m_end; ++m) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        if (m1) sum += static_cast<int64>(m1) * PieceType::eval(fast_div(y, dy, j1));
        if (m2) sum += static_cast<int64>(m2) * PieceType::eval(fast_div(y, dy, j2));
    }
    } else {
    // large_y: double division inexact, use scalar integer division
    for (int64 m = m_start; m < m_end; ++m) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        if (m1) sum += static_cast<int64>(m1) * PieceType::eval(y / j1);
        if (m2) sum += static_cast<int64>(m2) * PieceType::eval(y / j2);
    }
    }
#else
    for (int64 m = m_start; m < m_end; ++m) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        if (m1) sum += static_cast<int64>(m1) * PieceType::eval(fast_div(y, dy, j1));
        if (m2) sum += static_cast<int64>(m2) * PieceType::eval(fast_div(y, dy, j2));
    }
#endif

    // Tail Boundary.
    // NOTE: must start past head_end: when the whole piece fits inside one
    // mod-6 block (m_start > m_end, i.e. narrow pieces at small y, reached
    // via y/3,y/6 sub-evals when u is small), head already covers
    // [j_start, j_end] and an unclamped tail would count everything twice
    // (found: M(1e8) wrong for u<~90k). Clamp is a no-op otherwise since
    // tail_start = m_end*6+1 >= m_start*6+1 >= head_end+1 when m_start<=m_end.
    int64 tail_start = std::max(j_start, m_end * 6 + 1);
    tail_start = std::max(tail_start, head_end + 1);
    for (int64 j = tail_start; j <= j_end; ++j) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = mu_ptr[j];
            if (m != 0) {
                int64 q = fast_div(y, dy, j);
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
    sum += run_s2_fast<Piece1>(1, b6, y, dy, mu_ptr);
    sum += run_s2_fast<Piece2>(b6 + 1, a6, y, dy, mu_ptr);
    sum += run_s2_fast<Piece3>(a6 + 1, b3, y, dy, mu_ptr);
    sum += run_s2_fast<Piece4>(b3 + 1, a3, y, dy, mu_ptr);
    sum += run_s2_fast<Piece5>(a3 + 1, b2, y, dy, mu_ptr);
    sum += run_s2_fast<Piece6>(b2 + 1, a2, y, dy, mu_ptr);
    sum += run_s2_fast<Piece7>(a2 + 1, B, y, dy, mu_ptr);
    sum += run_s2_fast<Piece8>(B + 1, A, y, dy, mu_ptr);

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
    sum += run_s2_fast<SinglePiece1>(1, a6, y, dy, mu_ptr);
    sum += run_s2_fast<SinglePiece2>(a6 + 1, a3, y, dy, mu_ptr);
    sum += run_s2_fast<SinglePiece3>(a3 + 1, a2, y, dy, mu_ptr);
    sum += run_s2_fast<SinglePiece4>(a2 + 1, A, y, dy, mu_ptr);

    return sum;
}

// ---------------------------------------------------------------------------
// Compressed Hierarchical Mertens Table (CHMT) — optional huge-X path (local).
// Rebuilt on top of the shared upstream byte-log sieve (no rem[], no UDIV,
// int64-safe to 8B). Two-tier M(n):
//   Tier 1 (Dense):      int16_t M[0..u_dense] — direct O(1)
//   Tier 2 (Compressed): 2-bit mu deltas + int32 checkpoints — popcount O(1)
// Encoding per n: nonzero=(mu!=0), sign=(mu<0); M(q)=checkpoint+pos-neg.
// Each 256-value chunk = 32B/plane = 64B = 1 cacheline.
// Default engine stays dense+windowed (low RAM); use compute_compressed /
// MERTENS_USE_CHMT=1 only when outer-term reduction (N=X/u) dominates.
// ---------------------------------------------------------------------------

struct alignas(64) BitChunk {
    uint8_t nonzero[32];
    uint8_t sign[32];
};

class CompressedMertensTable {
public:
    int64 u_dense;
    int64 u_total;
    int64 mu_limit;

    std::vector<int16_t> dense_M;
    std::vector<int8_t> mu;

    int64 compressed_size;
    int64 num_chunks;
    std::vector<BitChunk> chunks;
    std::vector<int32_t> checkpoints;

    CompressedMertensTable(int64 total_limit, int64 dense_limit,
                           int num_threads = 0, int64 max_mu = 0)
        : u_dense(std::min(total_limit, dense_limit)), u_total(total_limit)
    {
        int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );

        mu_limit = (max_mu > 0 ? max_mu : u_dense);
        compressed_size = std::max<int64>(0, u_total - u_dense);
        num_chunks = (compressed_size > 0 ? (compressed_size + 255) / 256 : 0);
        build(threads);
    }

    inline int32_t get_M(int64 q) const noexcept {
        if (q <= u_dense) {
            return static_cast<int32_t>(dense_M[static_cast<size_t>(q)]);
        }
        return reconstruct_compressed(q);
    }

    inline int8_t get_mu(int64 j) const noexcept {
        return (j <= mu_limit ? mu[static_cast<size_t>(j)] : 0);
    }

    inline const int16_t* dense_data() const noexcept {
        return dense_M.data();
    }

    inline void prefetch_M(int64 q) const noexcept {
        if (q <= u_dense) {
            __builtin_prefetch(&dense_M[static_cast<size_t>(q)], 0, 0);
        } else if (q <= u_total) {
            int64 ci = (q - u_dense - 1) >> 8;
            __builtin_prefetch(&chunks[static_cast<size_t>(ci)], 0, 0);
            __builtin_prefetch(&checkpoints[static_cast<size_t>(ci)], 0, 0);
        }
    }

    inline int32_t reconstruct_compressed(int64 q) const noexcept {
        int64 offset = q - u_dense;
        int64 ci = (offset - 1) >> 8;
        int r = static_cast<int>((offset - 1) & 255) + 1;

        int32_t base = checkpoints[static_cast<size_t>(ci)];
        const BitChunk& chunk = chunks[static_cast<size_t>(ci)];

        const uint64_t* nz64 = reinterpret_cast<const uint64_t*>(chunk.nonzero);
        const uint64_t* sg64 = reinterpret_cast<const uint64_t*>(chunk.sign);

        int full_words = r >> 6;
        int rem_bits = r & 63;

        int pos = 0, neg = 0;
        for (int w = 0; w < full_words; ++w) {
            uint64_t nz = nz64[w];
            uint64_t sg = sg64[w];
            pos += __builtin_popcountll(nz & ~sg);
            neg += __builtin_popcountll(nz & sg);
        }
        if (rem_bits > 0) {
            uint64_t mask = (rem_bits == 64) ? ~0ULL : ((1ULL << rem_bits) - 1ULL);
            uint64_t nz = nz64[full_words] & mask;
            uint64_t sg = sg64[full_words] & mask;
            pos += __builtin_popcountll(nz & ~sg);
            neg += __builtin_popcountll(nz & sg);
        }

        return base + pos - neg;
    }

private:
    void build(int threads) {
        int64 half_u_total = (u_total + 1) / 2;
        std::vector<int8_t> mu_odd(static_cast<size_t>(half_u_total + 64), 0);
        // Shared upstream sieve: division-free, int64-safe (fixes local
        // int-overflow segfault at u>2.1e9 and lg/is_zero extra traffic).
        sieve_mu_odd_byte_log(mu_odd.data(), half_u_total, u_total, threads);
        const int8_t* __restrict__ mu_odd_ptr = mu_odd.data();

        // Phase 2: dense M prefix
        dense_M.assign(static_cast<size_t>(u_dense + 1), 0);
        {
            int16_t* __restrict__ M_ptr = dense_M.data();
            int num_t = threads;
            std::vector<int32_t> block_sums(static_cast<size_t>(num_t + 1), 0);

            #pragma omp parallel num_threads(num_t)
            {
                int tid = omp_get_thread_num();
                int64 chunk_sz = (u_dense + num_t - 1) / num_t;
                chunk_sz = (chunk_sz + 3) & ~3LL;
                int64 start = tid * chunk_sz + 1;
                int64 end = std::min(u_dense, (tid + 1) * chunk_sz);

                int32_t local_sum = 0;
                for (int64 i = start; i <= end; ++i) {
                    local_sum += mu_from_odd(i, mu_odd_ptr);
                }
                block_sums[static_cast<size_t>(tid + 1)] = local_sum;
            }
            for (int i = 1; i <= num_t; ++i)
                block_sums[static_cast<size_t>(i)] += block_sums[static_cast<size_t>(i - 1)];

            #pragma omp parallel num_threads(num_t)
            {
                int tid = omp_get_thread_num();
                int64 chunk_sz = (u_dense + num_t - 1) / num_t;
                chunk_sz = (chunk_sz + 3) & ~3LL;
                int64 start = tid * chunk_sz + 1;
                int64 end = std::min(u_dense, (tid + 1) * chunk_sz);

                int32_t run_sum = block_sums[static_cast<size_t>(tid)];
                for (int64 i = start; i <= end; ++i) {
                    run_sum += mu_from_odd(i, mu_odd_ptr);
                    M_ptr[i] = static_cast<int16_t>(run_sum);
                }
            }
        }

        // Phase 3: mu for S2
        mu.assign(static_cast<size_t>(mu_limit + 1), 0);
        {
            int8_t* __restrict__ mu_ptr = mu.data();
            for (int64 i = 1; i <= mu_limit; ++i) {
                mu_ptr[i] = mu_from_odd(i, mu_odd_ptr);
            }
        }

        // Phase 4: compressed tier (branchless 8-element packing;
        // mu(4k)=0 so only 6 live values per 8, bits 3,7 skipped)
        if (num_chunks > 0) {
            chunks.resize(static_cast<size_t>(num_chunks));
            checkpoints.resize(static_cast<size_t>(num_chunks));
            std::vector<int32_t> local_sums(static_cast<size_t>(num_chunks), 0);

            #pragma omp parallel for schedule(static, 256) num_threads(threads)
            for (int64 ci = 0; ci < num_chunks; ++ci) {
                BitChunk& ch = chunks[static_cast<size_t>(ci)];
                int32_t lsum = 0;
                int64 k_base = (u_dense + ci * 256) >> 3;

                for (int b = 0; b < 32; ++b) {
                    int64 k = k_base + b;
                    int64 idx4 = (k << 2) + 1;
                    int64 idx2 = (k << 1) + 1;

                    int8_t v1 = mu_odd_ptr[idx4];
                    int8_t v2 = static_cast<int8_t>(-mu_odd_ptr[idx2]);
                    int8_t v3 = mu_odd_ptr[idx4 + 1];
                    int8_t v5 = mu_odd_ptr[idx4 + 2];
                    int8_t v6 = static_cast<int8_t>(-mu_odd_ptr[idx2 + 1]);
                    int8_t v7 = mu_odd_ptr[idx4 + 3];

                    uint8_t nz = static_cast<uint8_t>(
                        (v1 != 0) | ((v2 != 0) << 1) | ((v3 != 0) << 2) |
                        ((v5 != 0) << 4) | ((v6 != 0) << 5) | ((v7 != 0) << 6)
                    );
                    uint8_t sg = static_cast<uint8_t>(
                        (v1 < 0) | ((v2 < 0) << 1) | ((v3 < 0) << 2) |
                        ((v5 < 0) << 4) | ((v6 < 0) << 5) | ((v7 < 0) << 6)
                    );

                    ch.nonzero[b] = nz;
                    ch.sign[b] = sg;
                    lsum += (v1 + v2 + v3 + v5 + v6 + v7);
                }
                local_sums[static_cast<size_t>(ci)] = lsum;
            }

            checkpoints[0] = static_cast<int32_t>(dense_M[static_cast<size_t>(u_dense)]);
            for (int64 ci = 1; ci < num_chunks; ++ci) {
                checkpoints[static_cast<size_t>(ci)] =
                    checkpoints[static_cast<size_t>(ci - 1)] + local_sums[static_cast<size_t>(ci - 1)];
            }
        }
    }
};

/**
 * Master Deléglise-Rivat Mertens Solver.
 */
class DelégliseRivatEngine {
public:
    /**
     * Dynamically chooses optimal balance parameter cx based on hardware SIMD and memory characteristics.
     * MERTENS_CX env override (0.1..5.0) for empirical tuning without rebuild.
     */
    static inline double choose_cx(int64 X) noexcept {
        if (const char* e = std::getenv("MERTENS_CX")) {
            double v = std::atof(e);
            if (v > 0.1 && v < 5.0) return v;
        }
        if (X >= 1000000000000000LL) return 1.35;
        if (X >= 10000000000000LL) return 1.15;
        if (X >= 100000000000LL) return 0.95;
        return 0.70;
    }

    /**
     * Dynamically chooses optimal sieve cutoff u for target X.
     * MERTENS_FX env override (0.1..3.0) scales the balance factor;
     * MERTENS_U_CAP env override (1e8..4e9) hard-caps u. For empirical
     * tuning without rebuild.
     */
    static inline int64 choose_sieve_limit(int64 X, int threads) {
        (void)threads;
        if (X <= 50000000LL) {
            return X; // Direct sieve table for X <= 50M
        }

        double loglogX = std::log(std::max(2.0, std::log(static_cast<double>(X))));
        // Measured optimum on x86-64/MSVC (i7-7700, 17GB): fx ~= 0.3.
        // Smaller u shrinks the random-gather M table (cache wins) until the
        // extra outer terms take over: ~300M best at 1e14, ~1.2G at 1e15
        // (~15-20% faster than the old 0.95/1.25 + 600M cap). Retune per box
        // via MERTENS_FX / MERTENS_U_CAP.
        double fx = 0.32;
        if (const char* e = std::getenv("MERTENS_FX")) {
            double v = std::atof(e);
            if (v >= 0.1 && v <= 3.0) fx = v;
        }

        int64 u = static_cast<int64>(fx * std::pow(static_cast<double>(X) / loglogX, 2.0 / 3.0));
        int64 S = isqrt(X);
        if (u < 3 * S) u = 3 * S;

        // Cap u at 1.5G (~3 GB M table max). Old 600M cap cost ~15-20% at
        // 1e14/1e15 here; beyond ~1.8G returns diminish and fragmentation
        // risk grows (needs contiguous multi-GB vectors).
        int64 cap = 1500000000LL;
        if (const char* e = std::getenv("MERTENS_U_CAP")) {
            int64 v = static_cast<int64>(std::atoll(e));
            if (v >= 100000000LL && v <= 4000000000LL) cap = v;
        }
        return std::min(u, cap);
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

        if (X <= 100000LL) {
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

        // Opt-in huge-X path: MERTENS_USE_CHMT=1 routes large X to the
        // compressed table (local). Default stays dense+windowed (upstream).
        if (X >= COMPRESSED_THRESHOLD && use_chmt_env()) {
            return compute_compressed(X, threads);
        }

        int64 u = choose_sieve_limit(X, threads);
        const double cx = choose_cx(X);
        const int64 N = X / u;

        int64 A_max = static_cast<int64>(cx * std::sqrt(static_cast<double>(X))) + 1000;
        int64 mu_limit = std::max(A_max, N + 1000);

        // Precompute sieve table (with compact mu limit)
        SieveTable table(u, threads, (X <= u ? 0 : mu_limit));

        if (X <= u) {
            return table.get_M(X);
        }

        const int16_t* __restrict M_ptr = table.data();
        const int8_t* __restrict mu_ptr = table.mu.data();

        int64 n6 = N / 6;
        int64 n3 = N / 3;
        int64 n2 = N / 2;

        // ---------------------------------------------------------------------------
        // Build unified term list (merges all 4 ranges into a single dispatch vector)
        // ---------------------------------------------------------------------------
        struct KTerm { int64 k; int8_t mu_k; uint8_t range; };
        std::vector<KTerm> terms;
        terms.reserve(static_cast<size_t>(N * 2 / 3));

        for (int64 k = 1; k <= N; ++k) {
            if (k % 2 == 0 || k % 3 == 0) continue;
            int8_t mk = mu_ptr[k];
            if (mk == 0) continue;
            uint8_t range;
            if (k <= n6) range = 1;
            else if (k <= n3) range = 2;
            else if (k <= n2) range = 3;
            else range = 4;
            terms.push_back({k, mk, range});
        }

        int64 total_M = 0;

        // ---------------------------------------------------------------------------
        // Optimized S1: NEON 8-way unrolled with pure SIMD increments and vcvtq.
        // Used by eval_single_S (Range 4 + parts of Range 2).
        // Merged: NEON fast path for y<=2^53, scalar 8-way unrolled fast_div
        // path for large_y (local) + exact fast_div tails.
        // ---------------------------------------------------------------------------
        auto eval_single_S = [&](int64 y) noexcept -> int64 {
            int64 A = static_cast<int64>(cx * std::sqrt(static_cast<double>(y)));
            if (A >= y) A = y - 1;
            if (A < 1) A = 1;
            int64 kappa_y = y / (A + 1);

            int64 S1 = 0;
            int64 start_n = y / u + 1;
            double dy = static_cast<double>(y);
            int64 n = start_n;
            const bool large_y = (y > 9007199254740992LL);

#if defined(__ARM_NEON)
            if (!large_y) {
            float64x2_t v_dy = vdupq_n_f64(dy);
            float64x2_t v_step8 = {8.0, 8.0};
            float64x2_t v_n1 = {static_cast<double>(n),     static_cast<double>(n + 1)};
            float64x2_t v_n2 = {static_cast<double>(n + 2), static_cast<double>(n + 3)};
            float64x2_t v_n3 = {static_cast<double>(n + 4), static_cast<double>(n + 5)};
            float64x2_t v_n4 = {static_cast<double>(n + 6), static_cast<double>(n + 7)};

            for (; n + 7 <= kappa_y; n += 8) {
                float64x2_t v_q1 = vdivq_f64(v_dy, v_n1);
                float64x2_t v_q2 = vdivq_f64(v_dy, v_n2);
                float64x2_t v_q3 = vdivq_f64(v_dy, v_n3);
                float64x2_t v_q4 = vdivq_f64(v_dy, v_n4);

                v_n1 = vaddq_f64(v_n1, v_step8);
                v_n2 = vaddq_f64(v_n2, v_step8);
                v_n3 = vaddq_f64(v_n3, v_step8);
                v_n4 = vaddq_f64(v_n4, v_step8);

                int64x2_t v_qi1 = vcvtq_s64_f64(v_q1);
                int64x2_t v_qi2 = vcvtq_s64_f64(v_q2);
                int64x2_t v_qi3 = vcvtq_s64_f64(v_q3);
                int64x2_t v_qi4 = vcvtq_s64_f64(v_q4);

                S1 += M_ptr[vgetq_lane_s64(v_qi1, 0)]
                    + M_ptr[vgetq_lane_s64(v_qi1, 1)]
                    + M_ptr[vgetq_lane_s64(v_qi2, 0)]
                    + M_ptr[vgetq_lane_s64(v_qi2, 1)]
                    + M_ptr[vgetq_lane_s64(v_qi3, 0)]
                    + M_ptr[vgetq_lane_s64(v_qi3, 1)]
                    + M_ptr[vgetq_lane_s64(v_qi4, 0)]
                    + M_ptr[vgetq_lane_s64(v_qi4, 1)];
            }
            }
#endif
            // Scalar 8-way unrolled dense reads (local CHMT contribution):
            // helps large_y and non-NEON builds; exact via fast_div.
            for (; n + 7 <= kappa_y; n += 8) {
                S1 += M_ptr[fast_div(y, dy, n)]
                    + M_ptr[fast_div(y, dy, n + 1)]
                    + M_ptr[fast_div(y, dy, n + 2)]
                    + M_ptr[fast_div(y, dy, n + 3)]
                    + M_ptr[fast_div(y, dy, n + 4)]
                    + M_ptr[fast_div(y, dy, n + 5)]
                    + M_ptr[fast_div(y, dy, n + 6)]
                    + M_ptr[fast_div(y, dy, n + 7)];
            }
            for (; n <= kappa_y; ++n) {
                S1 += M_ptr[fast_div(y, dy, n)];
            }

            int64 S2 = eval_s2_single(y, A, mu_ptr);
            return 1 - S1 + kappa_y * static_cast<int64>(M_ptr[A]) - S2;
        };

        // ---------------------------------------------------------------------------
        // Optimized eval_comb2: 8-way NEON S1 and even-n loops with pure SIMD increments.
        // Merged: NEON guarded by !large_y, scalar 8-way unrolled fast_div paths
        // (local) for large_y / non-NEON, exact fast_div tails.
        // ---------------------------------------------------------------------------
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
            const bool large_y = (y > 9007199254740992LL);

#if defined(__ARM_NEON)
            if (!large_y) {
            float64x2_t v_dy = vdupq_n_f64(dy);
            float64x2_t v_step16 = {16.0, 16.0};
            float64x2_t v_n1 = {static_cast<double>(n),      static_cast<double>(n + 2)};
            float64x2_t v_n2 = {static_cast<double>(n + 4),  static_cast<double>(n + 6)};
            float64x2_t v_n3 = {static_cast<double>(n + 8),  static_cast<double>(n + 10)};
            float64x2_t v_n4 = {static_cast<double>(n + 12), static_cast<double>(n + 14)};

            for (; n + 15 <= kappa_y; n += 16) {
                float64x2_t v_q1 = vdivq_f64(v_dy, v_n1);
                float64x2_t v_q2 = vdivq_f64(v_dy, v_n2);
                float64x2_t v_q3 = vdivq_f64(v_dy, v_n3);
                float64x2_t v_q4 = vdivq_f64(v_dy, v_n4);

                v_n1 = vaddq_f64(v_n1, v_step16);
                v_n2 = vaddq_f64(v_n2, v_step16);
                v_n3 = vaddq_f64(v_n3, v_step16);
                v_n4 = vaddq_f64(v_n4, v_step16);

                int64x2_t v_qi1 = vcvtq_s64_f64(v_q1);
                int64x2_t v_qi2 = vcvtq_s64_f64(v_q2);
                int64x2_t v_qi3 = vcvtq_s64_f64(v_q3);
                int64x2_t v_qi4 = vcvtq_s64_f64(v_q4);

                S1_diff += M_ptr[vgetq_lane_s64(v_qi1, 0)]
                         + M_ptr[vgetq_lane_s64(v_qi1, 1)]
                         + M_ptr[vgetq_lane_s64(v_qi2, 0)]
                         + M_ptr[vgetq_lane_s64(v_qi2, 1)]
                         + M_ptr[vgetq_lane_s64(v_qi3, 0)]
                         + M_ptr[vgetq_lane_s64(v_qi3, 1)]
                         + M_ptr[vgetq_lane_s64(v_qi4, 0)]
                         + M_ptr[vgetq_lane_s64(v_qi4, 1)];
            }
            }
#endif
            // Scalar 8-wide unrolled odd loop (local contribution, exact via fast_div)
            for (; n + 15 <= kappa_y; n += 16) {
                S1_diff += M_ptr[fast_div(y, dy, n)]
                         + M_ptr[fast_div(y, dy, n + 2)]
                         + M_ptr[fast_div(y, dy, n + 4)]
                         + M_ptr[fast_div(y, dy, n + 6)]
                         + M_ptr[fast_div(y, dy, n + 8)]
                         + M_ptr[fast_div(y, dy, n + 10)]
                         + M_ptr[fast_div(y, dy, n + 12)]
                         + M_ptr[fast_div(y, dy, n + 14)];
            }
            for (; n <= kappa_y; n += 2) {
                S1_diff += M_ptr[fast_div(y, dy, n)];
            }

            // ---------------------------------------------------------------------------
            // Even-n correction: pure SIMD increments and vcvtq (+scalar unrolled fast_div)
            // ---------------------------------------------------------------------------
            if (two_kappa_y2 > kappa_y) {
                int64 start_even = kappa_y + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
#if defined(__ARM_NEON)
                if (!large_y) {
                float64x2_t v_dy = vdupq_n_f64(dy);
                float64x2_t v_step16 = {16.0, 16.0};
                float64x2_t v_ne1 = {static_cast<double>(ne),      static_cast<double>(ne + 2)};
                float64x2_t v_ne2 = {static_cast<double>(ne + 4),  static_cast<double>(ne + 6)};
                float64x2_t v_ne3 = {static_cast<double>(ne + 8),  static_cast<double>(ne + 10)};
                float64x2_t v_ne4 = {static_cast<double>(ne + 12), static_cast<double>(ne + 14)};

                for (; ne + 15 <= two_kappa_y2; ne += 16) {
                    float64x2_t v_qe1 = vdivq_f64(v_dy, v_ne1);
                    float64x2_t v_qe2 = vdivq_f64(v_dy, v_ne2);
                    float64x2_t v_qe3 = vdivq_f64(v_dy, v_ne3);
                    float64x2_t v_qe4 = vdivq_f64(v_dy, v_ne4);

                    v_ne1 = vaddq_f64(v_ne1, v_step16);
                    v_ne2 = vaddq_f64(v_ne2, v_step16);
                    v_ne3 = vaddq_f64(v_ne3, v_step16);
                    v_ne4 = vaddq_f64(v_ne4, v_step16);

                    int64x2_t v_qei1 = vcvtq_s64_f64(v_qe1);
                    int64x2_t v_qei2 = vcvtq_s64_f64(v_qe2);
                    int64x2_t v_qei3 = vcvtq_s64_f64(v_qe3);
                    int64x2_t v_qei4 = vcvtq_s64_f64(v_qe4);

                    S1_diff -= M_ptr[vgetq_lane_s64(v_qei1, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei1, 1)]
                             + M_ptr[vgetq_lane_s64(v_qei2, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei2, 1)]
                             + M_ptr[vgetq_lane_s64(v_qei3, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei3, 1)]
                             + M_ptr[vgetq_lane_s64(v_qei4, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei4, 1)];
                }
                }
#endif
                for (; ne + 15 <= two_kappa_y2; ne += 16) {
                    S1_diff -= (M_ptr[fast_div(y, dy, ne)]
                              + M_ptr[fast_div(y, dy, ne + 2)]
                              + M_ptr[fast_div(y, dy, ne + 4)]
                              + M_ptr[fast_div(y, dy, ne + 6)]
                              + M_ptr[fast_div(y, dy, ne + 8)]
                              + M_ptr[fast_div(y, dy, ne + 10)]
                              + M_ptr[fast_div(y, dy, ne + 12)]
                              + M_ptr[fast_div(y, dy, ne + 14)]);
                }
                for (; ne <= two_kappa_y2; ne += 2) {
                    S1_diff -= M_ptr[fast_div(y, dy, ne)];
                }
            } else if (two_kappa_y2 < kappa_y) {
                int64 start_even = two_kappa_y2 + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
#if defined(__ARM_NEON)
                if (!large_y) {
                float64x2_t v_dy2 = vdupq_n_f64(dy);
                float64x2_t v_step16_2 = {16.0, 16.0};
                float64x2_t v_ne1 = {static_cast<double>(ne),      static_cast<double>(ne + 2)};
                float64x2_t v_ne2 = {static_cast<double>(ne + 4),  static_cast<double>(ne + 6)};
                float64x2_t v_ne3 = {static_cast<double>(ne + 8),  static_cast<double>(ne + 10)};
                float64x2_t v_ne4 = {static_cast<double>(ne + 12), static_cast<double>(ne + 14)};

                for (; ne + 15 <= kappa_y; ne += 16) {
                    float64x2_t v_qe1 = vdivq_f64(v_dy2, v_ne1);
                    float64x2_t v_qe2 = vdivq_f64(v_dy2, v_ne2);
                    float64x2_t v_qe3 = vdivq_f64(v_dy2, v_ne3);
                    float64x2_t v_qe4 = vdivq_f64(v_dy2, v_ne4);

                    v_ne1 = vaddq_f64(v_ne1, v_step16_2);
                    v_ne2 = vaddq_f64(v_ne2, v_step16_2);
                    v_ne3 = vaddq_f64(v_ne3, v_step16_2);
                    v_ne4 = vaddq_f64(v_ne4, v_step16_2);

                    int64x2_t v_qei1 = vcvtq_s64_f64(v_qe1);
                    int64x2_t v_qei2 = vcvtq_s64_f64(v_qe2);
                    int64x2_t v_qei3 = vcvtq_s64_f64(v_qe3);
                    int64x2_t v_qei4 = vcvtq_s64_f64(v_qe4);

                    S1_diff += M_ptr[vgetq_lane_s64(v_qei1, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei1, 1)]
                             + M_ptr[vgetq_lane_s64(v_qei2, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei2, 1)]
                             + M_ptr[vgetq_lane_s64(v_qei3, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei3, 1)]
                             + M_ptr[vgetq_lane_s64(v_qei4, 0)]
                             + M_ptr[vgetq_lane_s64(v_qei4, 1)];
                }
                }
#endif
                for (; ne + 15 <= kappa_y; ne += 16) {
                    S1_diff += (M_ptr[fast_div(y, dy, ne)]
                              + M_ptr[fast_div(y, dy, ne + 2)]
                              + M_ptr[fast_div(y, dy, ne + 4)]
                              + M_ptr[fast_div(y, dy, ne + 6)]
                              + M_ptr[fast_div(y, dy, ne + 8)]
                              + M_ptr[fast_div(y, dy, ne + 10)]
                              + M_ptr[fast_div(y, dy, ne + 12)]
                              + M_ptr[fast_div(y, dy, ne + 14)]);
                }
                for (; ne <= kappa_y; ne += 2) {
                    S1_diff += M_ptr[fast_div(y, dy, ne)];
                }
            }

            int64 S2_diff = eval_s2_combined(y, A, B, mu_ptr);
            return -S1_diff + (kappa_y * static_cast<int64>(M_ptr[A]) - kappa_y2 * static_cast<int64>(M_ptr[B])) - S2_diff;
        };

        // ---------------------------------------------------------------------------
        // Single merged parallel loop with dynamic scheduling.
        // ---------------------------------------------------------------------------
        #pragma omp parallel for reduction(+:total_M) schedule(dynamic, 64) num_threads(threads)
        for (long long i = 0; i < static_cast<long long>(terms.size()); ++i) {
            int64 k = terms[i].k;
            int8_t mu_k = terms[i].mu_k;
            uint8_t range = terms[i].range;
            int64 y = X / k;
            int64 term;

            switch (range) {
                case 1: term = eval_comb2(y, y / 2) - eval_comb2(y / 3, y / 6); break;
                case 2: term = eval_comb2(y, y / 2) - eval_single_S(y / 3); break;
                case 3: term = eval_comb2(y, y / 2); break;
                default: term = eval_single_S(y); break;
            }

            total_M += static_cast<int64>(mu_k) * term;
        }

        return total_M;
    }

    /**
     * CHMT tuning (local, optional huge-X path).
     * Default engine uses choose_sieve_limit (fx=0.32, cap 1.5G, low RAM).
     * CHMT scales u to 8B via 2-bit compression to cut N=X/u when outer
     * terms dominate. Enable via MERTENS_USE_CHMT=1 or compute_compressed().
     */
    static constexpr int64 COMPRESSED_THRESHOLD = 1000000000000000LL; // 10^15
    static constexpr int64 DENSE_LIMIT = 600000000LL; // 600M

    static inline double choose_cx_compressed(int64 X) noexcept {
        (void)X;
        // Local used flat 1.5 to shift work S1->S2 at huge u; keep upstream
        // env-aware values by default (see choose_cx). This helper preserves
        // the old CHMT default for A/B comparisons.
        return 1.5;
    }

    static inline int64 choose_compressed_limit(int64 X) {
        if (X < 50000000LL) return X;
        double loglogX = std::log(std::max(2.0, std::log(static_cast<double>(X))));
        double fx = 0.95;
        if (X >= 1000000000000000LL) fx = 1.25;
        if (X >= 10000000000000000LL) fx = 1.50;
        int64 u = static_cast<int64>(fx * std::pow(static_cast<double>(X) / loglogX, 2.0 / 3.0));
        int64 S = isqrt(X);
        if (u < 3 * S) u = 3 * S;
        u = ((u + 255) / 256) * 256;
        return std::min(u, 8000000000LL);
    }

    static bool use_chmt_env() noexcept {
        if (const char* e = std::getenv("MERTENS_USE_CHMT")) {
            return (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' || e[0] == 'T');
        }
        return false;
    }

    /**
     * Compressed-table computation path (CHMT, opt-in for X>=1e15).
     * Uses two-tier M table with popcount reconstruction + 8-way unrolled S1.
     * S2 uses the merged run_s2_fast (tail-clamp + fast_div exact).
     */
    static int64 compute_compressed(int64 X, int threads) {
        int64 u_total = choose_compressed_limit(X);
        int64 u_dense = std::min(u_total, DENSE_LIMIT);
        // Upstream cx by default; set MERTENS_CX=1.5 to reproduce local tuning.
        const double cx = choose_cx(X);
        const int64 N = X / u_total;

        int64 A_max = static_cast<int64>(cx * std::sqrt(static_cast<double>(X))) + 1000;
        int64 mu_limit = std::max(A_max, N + 1000);

        CompressedMertensTable table(u_total, u_dense, threads, mu_limit);

        if (X <= u_total) {
            return table.get_M(X);
        }

        const int8_t* __restrict mu_ptr = table.mu.data();
        struct KTerm { int64 k; int8_t mu_k; uint8_t range; };
        std::vector<KTerm> terms;

        int64 n6 = N / 6, n3 = N / 3, n2 = N / 2;
        terms.reserve(static_cast<size_t>(N * 2 / 3 + 64));
        for (int64 k = 1; k <= N; ++k) {
            if (k % 2 == 0 || k % 3 == 0) continue;
            int8_t mk = table.get_mu(k);
            if (mk == 0) continue;
            uint8_t range;
            if (k <= n6) range = 1;
            else if (k <= n3) range = 2;
            else if (k <= n2) range = 3;
            else range = 4;
            terms.push_back({k, mk, range});
        }

        int64 total_M = 0;
        const int16_t* __restrict M_dense = table.dense_data();

        auto eval_single_S = [&](int64 y) noexcept -> int64 {
            int64 A = static_cast<int64>(cx * std::sqrt(static_cast<double>(y)));
            if (A >= y) A = y - 1;
            if (A < 1) A = 1;
            int64 kappa_y = y / (A + 1);
            int64 S1 = 0;
            int64 start_n = y / u_total + 1;
            double dy = static_cast<double>(y);

            int64 split_n = std::min(kappa_y, (u_dense > 0 ? y / u_dense : kappa_y));
            for (int64 n = start_n; n <= split_n; ++n) {
                int64 q = fast_div(y, dy, n);
                S1 += table.reconstruct_compressed(q);
            }
            int64 n = std::max(start_n, split_n + 1);
            for (; n + 7 <= kappa_y; n += 8) {
                S1 += M_dense[fast_div(y, dy, n)]
                    + M_dense[fast_div(y, dy, n + 1)]
                    + M_dense[fast_div(y, dy, n + 2)]
                    + M_dense[fast_div(y, dy, n + 3)]
                    + M_dense[fast_div(y, dy, n + 4)]
                    + M_dense[fast_div(y, dy, n + 5)]
                    + M_dense[fast_div(y, dy, n + 6)]
                    + M_dense[fast_div(y, dy, n + 7)];
            }
            for (; n <= kappa_y; ++n) {
                S1 += M_dense[fast_div(y, dy, n)];
            }
            int64 S2 = eval_s2_single(y, A, mu_ptr);
            return 1 - S1 + kappa_y * static_cast<int64>(M_dense[A]) - S2;
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
            int64 start_odd = (y / u_total + 1);
            if (start_odd % 2 == 0) ++start_odd;
            double dy = static_cast<double>(y);

            int64 split_odd = std::min(kappa_y, (u_dense > 0 ? y / u_dense : kappa_y));
            int64 n = start_odd;
            for (; n <= split_odd; n += 2) {
                int64 q = fast_div(y, dy, n);
                S1_diff += table.reconstruct_compressed(q);
            }
            for (; n + 15 <= kappa_y; n += 16) {
                S1_diff += M_dense[fast_div(y, dy, n)]
                         + M_dense[fast_div(y, dy, n + 2)]
                         + M_dense[fast_div(y, dy, n + 4)]
                         + M_dense[fast_div(y, dy, n + 6)]
                         + M_dense[fast_div(y, dy, n + 8)]
                         + M_dense[fast_div(y, dy, n + 10)]
                         + M_dense[fast_div(y, dy, n + 12)]
                         + M_dense[fast_div(y, dy, n + 14)];
            }
            for (; n <= kappa_y; n += 2) {
                S1_diff += M_dense[fast_div(y, dy, n)];
            }

            if (two_kappa_y2 > kappa_y) {
                int64 start_even = kappa_y + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
                for (; ne + 15 <= two_kappa_y2; ne += 16) {
                    S1_diff -= (M_dense[fast_div(y, dy, ne)]
                              + M_dense[fast_div(y, dy, ne + 2)]
                              + M_dense[fast_div(y, dy, ne + 4)]
                              + M_dense[fast_div(y, dy, ne + 6)]
                              + M_dense[fast_div(y, dy, ne + 8)]
                              + M_dense[fast_div(y, dy, ne + 10)]
                              + M_dense[fast_div(y, dy, ne + 12)]
                              + M_dense[fast_div(y, dy, ne + 14)]);
                }
                for (; ne <= two_kappa_y2; ne += 2) {
                    S1_diff -= M_dense[fast_div(y, dy, ne)];
                }
            } else if (two_kappa_y2 < kappa_y) {
                int64 start_even = two_kappa_y2 + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
                for (; ne + 15 <= kappa_y; ne += 16) {
                    S1_diff += (M_dense[fast_div(y, dy, ne)]
                              + M_dense[fast_div(y, dy, ne + 2)]
                              + M_dense[fast_div(y, dy, ne + 4)]
                              + M_dense[fast_div(y, dy, ne + 6)]
                              + M_dense[fast_div(y, dy, ne + 8)]
                              + M_dense[fast_div(y, dy, ne + 10)]
                              + M_dense[fast_div(y, dy, ne + 12)]
                              + M_dense[fast_div(y, dy, ne + 14)]);
                }
                for (; ne <= kappa_y; ne += 2) {
                    S1_diff += M_dense[fast_div(y, dy, ne)];
                }
            }

            int64 S2_diff = eval_s2_combined(y, A, B, mu_ptr);
            return -S1_diff + (kappa_y * static_cast<int64>(M_dense[A]) - kappa_y2 * static_cast<int64>(M_dense[B])) - S2_diff;
        };

        #pragma omp parallel for reduction(+:total_M) schedule(dynamic, 8) num_threads(threads)
        for (long long i = 0; i < static_cast<long long>(terms.size()); ++i) {
            int64 k = terms[static_cast<size_t>(i)].k;
            int8_t mu_k = terms[static_cast<size_t>(i)].mu_k;
            uint8_t range = terms[static_cast<size_t>(i)].range;
            int64 y = X / k;
            int64 term;

            switch (range) {
                case 1: term = eval_comb2(y, y / 2) - eval_comb2(y / 3, y / 6); break;
                case 2: term = eval_comb2(y, y / 2) - eval_single_S(y / 3); break;
                case 3: term = eval_comb2(y, y / 2); break;
                default: term = eval_single_S(y); break;
            }

            total_M += static_cast<int64>(mu_k) * term;
        }

        return total_M;
    }
};

} // namespace mertens_dr

#endif // MERTENS_DR_HPP
