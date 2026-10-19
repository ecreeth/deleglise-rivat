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

// Quotient-loop backend selection.
// Define MERTENS_USE_NEON_DIV=1 to opt into the NEON 4-way pipelined
// double-division loops. On ARM the default is scalar division: measured ~6%
// faster on Apple M1 for M(1e14) because the scalar divider throughput plus
// out-of-order execution beats explicit NEON vdiv+vcvt once M-table gather
// latency dominates the loop body.
#if defined(__ARM_NEON) && defined(MERTENS_USE_NEON_DIV)
#define MERTENS_NEON_DIV 1
#endif

// SSE2 is part of the x86-64 baseline. Quotient division can be disabled
// independently of the byte-sieve vector kernels.
#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#define MERTENS_HAVE_SSE2 1
#if !defined(MERTENS_USE_SIMD_SIEVE) || MERTENS_USE_SIMD_SIEVE
#define MERTENS_SIMD_SIEVE 1
#endif
#if !defined(MERTENS_USE_SSE2_DIV) || MERTENS_USE_SSE2_DIV
#define MERTENS_SSE2_DIV 1
#endif
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
    const uint64 target = static_cast<uint64>(n);
    uint64 s = static_cast<uint64>(std::sqrt(static_cast<double>(n)));
    // For signed 64-bit n, the candidate is below 2^32, so these unsigned
    // squares fit even when the next square exceeds INT64_MAX.
    while ((s + 1) * (s + 1) <= target) ++s;
    while (s * s > target) --s;
    return static_cast<int64>(s);
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

#if defined(MERTENS_SSE2_DIV)
// Dense quotient lookups have 32-bit indices for the default sieve cap.
// Two packed divides handle four independent indices per iteration.
template <int Stride>
inline int64 sum_dense_quotients(int64 y, int64 first, int64 last,
                                 const int16_t* __restrict__ values) noexcept {
    static_assert(Stride == 1 || Stride == 2);
    if (first > last) return 0;
    const double dy = static_cast<double>(y);
    int64 sum = 0;
    int64 n = first;
    if (y <= 9007199254740992LL && y / first <= INT32_MAX) {
        const __m128d numerator = _mm_set1_pd(dy);
        const __m128d step = _mm_set1_pd(4.0 * Stride);
        __m128d d0 = _mm_set_pd(static_cast<double>(n + Stride), static_cast<double>(n));
        __m128d d1 = _mm_set_pd(static_cast<double>(n + 3 * Stride), static_cast<double>(n + 2 * Stride));
        for (; n + 3 * Stride <= last; n += 4 * Stride) {
            const __m128i q0 = _mm_cvttpd_epi32(_mm_div_pd(numerator, d0));
            const __m128i q1 = _mm_cvttpd_epi32(_mm_div_pd(numerator, d1));
            sum += values[_mm_cvtsi128_si32(q0)]
                 + values[_mm_cvtsi128_si32(_mm_srli_si128(q0, 4))]
                 + values[_mm_cvtsi128_si32(q1)]
                 + values[_mm_cvtsi128_si32(_mm_srli_si128(q1, 4))];
            d0 = _mm_add_pd(d0, step);
            d1 = _mm_add_pd(d1, step);
        }
    }
    for (; n + 7 * Stride <= last; n += 8 * Stride) {
        sum += values[fast_div(y, dy, n)]
             + values[fast_div(y, dy, n + Stride)]
             + values[fast_div(y, dy, n + 2 * Stride)]
             + values[fast_div(y, dy, n + 3 * Stride)]
             + values[fast_div(y, dy, n + 4 * Stride)]
             + values[fast_div(y, dy, n + 5 * Stride)]
             + values[fast_div(y, dy, n + 6 * Stride)]
             + values[fast_div(y, dy, n + 7 * Stride)];
    }
    for (; n <= last; n += Stride) sum += values[fast_div(y, dy, n)];
    return sum;
}
#endif

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

// Entries represent consecutive values separated by Stride (1 or 2).
// All entries between powers of two share one exact logarithmic threshold.
template <int Stride>
inline void finalize_mobius_bytes(uint8_t* values, int64 len, int64 first) noexcept {
    static_assert(Stride == 1 || Stride == 2);
    int64 i = 0;
    while (i < len) {
        const unsigned bits = std::bit_width(static_cast<uint64_t>(first + Stride * i));
        const int64 stop = std::min<int64>(len, ((uint64_t(1) << bits) - first + Stride - 1) / Stride);
#if defined(MERTENS_SIMD_SIEVE)
        const __m128i ones = _mm_set1_epi8(1);
        const __m128i threshold = _mm_set1_epi8(static_cast<char>(bits - 1));
        const __m128i minus_one = _mm_set1_epi8(-1);
        for (; i + 15 < stop; i += 16) {
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + i));
            const __m128i above = _mm_cmpgt_epi8(a, threshold);
            const __m128i parity = _mm_and_si128(_mm_xor_si128(a, above), ones);
            const __m128i value = _mm_add_epi8(minus_one, _mm_slli_epi16(parity, 1));
            const __m128i valid = _mm_cmpgt_epi8(a, minus_one);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(values + i), _mm_and_si128(value, valid));
        }
#endif
        for (; i < stop; ++i) {
            const uint8_t a = values[i];
            const int sign = (a & 1) ? -1 : 1;
            values[i] = static_cast<uint8_t>((a & 0x80) ? 0 : (a > bits - 1 ? sign : -sign));
        }
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

    // Period 17325 = lcm(3,5,7,11,9,25) in odd-table indices.
    // This permits pre-marking the two densest odd square strides as well
    // as weights for 3,5,7,11; every copied phase preserves all residues.
    constexpr int64 STENCIL_SIZE = 17325;
    static const std::vector<int8_t> kOddStencil = []() {
        constexpr int kLen = 17325;
        std::vector<int8_t> st(static_cast<size_t>(kLen), 0);
        const int small_p[4] = {3, 5, 7, 11};
        uint8_t small_w[4];
        for (int k = 0; k < 4; ++k)
            small_w[k] = static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(small_p[k] - 1)) | 1u);
        for (int j = 0; j < kLen; ++j) {
            int v = 2 * j + 1;
            int acc = 0;
            for (int k = 0; k < 4; ++k) {
                if (v % small_p[k] == 0) acc += small_w[k];
            }
            if (v % 9 == 0 || v % 25 == 0) acc |= 0x80;
            st[static_cast<size_t>(j)] = static_cast<int8_t>(acc);
        }
        return st;
    }();
    size_t pi_start = 0;
    while (pi_start < primes.size() && primes[pi_start] < 13) ++pi_start;
    size_t square_start = 0;
    while (square_start < primes.size() && primes[square_start] < 7) ++square_start;

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
            // Stencil fill with phase offset (replaces memset + p<=11 weights).
            {
                int64 off = (low_idx - 1) % STENCIL_SIZE;
                int64 done = 0;
                int64 first = std::min<int64>(len, STENCIL_SIZE - off);
                std::memcpy(b_mu, kOddStencil.data() + off, static_cast<size_t>(first));
                done = first;
                while (done + STENCIL_SIZE <= len) {
                    std::memcpy(b_mu + done, kOddStencil.data(), STENCIL_SIZE);
                    done += STENCIL_SIZE;
                }
                if (done < len) {
                    std::memcpy(b_mu + done, kOddStencil.data(),
                                static_cast<size_t>(len - done));
                }
            }

            // Squares of 3 and 5 are already in the periodic stencil.
            for (size_t pi = square_start; pi < primes.size(); ++pi) {
                int64 p = primes[pi];
                int64 p2 = p * p;
                if (p2 > 2 * high_idx - 1) break;
                int64 low_val = 2 * low_idx - 1;

                int64 start_sq_val = ((low_val + p2 - 1) / p2) * p2;
                if (start_sq_val % 2 == 0) start_sq_val += p2;
                int64 start_sq_i = (start_sq_val - low_val) / 2;

                for (int64 i = start_sq_i; i < len; i += p2) {
                    b_mu[i] |= static_cast<int8_t>(0x80);
                }
            }

            // Weight accumulation for p >= 13 (p <= 11 via stencil).
            for (size_t pi = pi_start; pi < primes.size(); ++pi) {
                int64 p = primes[pi];
                uint8_t w = prime_w[pi];
                int64 low_val = 2 * low_idx - 1;

                int64 start_val = ((low_val + p - 1) / p) * p;
                if (start_val % 2 == 0) start_val += p;
                int64 start_i = (start_val - low_val) / 2;

                for (int64 i = start_i; i < len; i += p) {
                    // The square flag survives addition: all prime weights
                    // together remain below 0x80 for the supported sieve range.
                    b_mu[i] = static_cast<int8_t>(static_cast<uint8_t>(b_mu[i]) + w);
                }
            }

            finalize_mobius_bytes<2>(reinterpret_cast<uint8_t*>(b_mu), len, 2 * low_idx - 1);
        }
    }
    mu_odd[0] = 0;
    if (half_u >= 1) mu_odd[1] = 1; // value 1 has no prime factors (acc=0 would give -1)
}

// Fixed logical partitions keep prefix offsets correct even when OpenMP
// supplies fewer workers than requested (including dynamic/nested teams).
inline void build_dense_prefix(int16_t* __restrict__ M_ptr,
                               const int8_t* __restrict__ mu_odd_ptr,
                               int64 m_limit, int threads) {
    int num_t = threads;
    std::vector<int32_t> block_sums(num_t + 1, 0);

    #pragma omp parallel for schedule(static) num_threads(num_t)
    for (int tid = 0; tid < num_t; ++tid) {
        int64 chunk = (m_limit + num_t - 1) / num_t;
        chunk = (chunk + 3) & ~3LL;
        int64 start = tid * chunk + 1;
        int64 end = std::min(m_limit, (tid + 1) * chunk);

        int32_t local_sum = 0;
        int64 i = start;
        // mu(4k) = 0; sum each aligned group with three loads.
        for (; i + 3 <= end; i += 4) {
            int64 k = (i - 1) / 4;
            local_sum += static_cast<int32_t>(mu_odd_ptr[2 * k + 1])
                       - static_cast<int32_t>(mu_odd_ptr[k + 1])
                       + static_cast<int32_t>(mu_odd_ptr[2 * k + 2]);
        }
        for (; i <= end; ++i) local_sum += mu_from_odd(i, mu_odd_ptr);
        block_sums[tid + 1] = local_sum;
    }

    for (int i = 1; i <= num_t; ++i) {
        block_sums[i] += block_sums[i - 1];
    }

    #pragma omp parallel for schedule(static) num_threads(num_t)
    for (int tid = 0; tid < num_t; ++tid) {
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

        build_dense_prefix(M.data(), mu_odd_ptr, m_limit, threads);

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
#if defined(MERTENS_NEON_DIV)
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
#if defined(MERTENS_SSE2_DIV)
    if (!large_y) {
        const __m128d numerator = _mm_set1_pd(dy);
        const __m128d step = _mm_set1_pd(6.0);
        __m128d divisor = _mm_set_pd(static_cast<double>(6 * m_start + 5),
                                    static_cast<double>(6 * m_start + 1));
        for (int64 m = m_start; m < m_end; ++m) {
            const __m128d q = _mm_div_pd(numerator, divisor);
            const int64 q1 = _mm_cvttsd_si64(q);
            const int64 q2 = _mm_cvttsd_si64(_mm_unpackhi_pd(q, q));
            // Evaluating both lanes avoids a branch for each Mobius byte.
            sum += static_cast<int64>(mu_ptr[6 * m + 1]) * PieceType::eval(q1)
                 + static_cast<int64>(mu_ptr[6 * m + 5]) * PieceType::eval(q2);
            divisor = _mm_add_pd(divisor, step);
        }
    } else
#endif
    {
        for (int64 m = m_start; m < m_end; ++m) {
            int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
            int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
            if (m1) sum += static_cast<int64>(m1) * PieceType::eval(fast_div(y, dy, j1));
            if (m2) sum += static_cast<int64>(m2) * PieceType::eval(fast_div(y, dy, j2));
        }
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
// Streaming compressed Mertens table. Each 256-value chunk stores two
// 32-byte Mobius bit planes plus four int32 checkpoints (one per 64 values).
// A query needs one checkpoint, two word loads, and two popcounts. The
// compressed tier uses 80/256 = 0.3125 bytes per integer, plus a dense prefix.
// Construction uses cache-sized per-worker buffers, with no full odd sieve.
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
    std::vector<int32_t> checkpoints; // M immediately before each 64-value word

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

        mu_limit = std::min(u_total, (max_mu > 0 ? max_mu : u_dense));
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
        auto prefetch = [](const void* address) noexcept {
#if defined(MERTENS_HAVE_SSE2)
            _mm_prefetch(static_cast<const char*>(address), _MM_HINT_NTA);
#elif !defined(_MSC_VER)
            __builtin_prefetch(address, 0, 0);
#else
            (void)address;
#endif
        };
        if (q <= u_dense) {
            prefetch(&dense_M[static_cast<size_t>(q)]);
        } else if (q <= u_total) {
            const int64 offset = q - u_dense - 1;
            prefetch(&chunks[static_cast<size_t>(offset >> 8)]);
            prefetch(&checkpoints[static_cast<size_t>(offset >> 6)]);
        }
    }

    inline int32_t reconstruct_compressed(int64 q) const noexcept {
        const int64 offset = q - u_dense - 1;
        const int64 word = offset >> 6;
        const BitChunk& chunk = chunks[static_cast<size_t>(word >> 2)];
        const size_t byte = static_cast<size_t>((word & 3) * 8);
        uint64_t nz, negative;
        std::memcpy(&nz, chunk.nonzero + byte, sizeof(nz));
        std::memcpy(&negative, chunk.sign + byte, sizeof(negative));
        const uint64_t mask = ~uint64_t(0) >> (63 - (offset & 63));
        return checkpoints[static_cast<size_t>(word)]
             + std::popcount(nz & mask) - 2 * std::popcount(negative & mask);
    }

private:
    void build(int threads) {
        if (num_chunks == 0) {
            SieveTable table(u_total, threads, mu_limit);
            dense_M = std::move(table.M);
            mu = std::move(table.mu);
            return;
        }
        // Dense and compressed runs have separate logical partitions so an
        // arbitrary dense boundary never splits a compressed 256-value chunk.
        constexpr int64 BLOCK = 131072;
        const int64 dense_blocks = (u_dense + BLOCK - 1) / BLOCK;
        const int64 compressed_blocks = (compressed_size + BLOCK - 1) / BLOCK;
        const int64 num_blocks = dense_blocks + compressed_blocks;
        dense_M.assign(static_cast<size_t>(u_dense + 1), 0);
        mu.assign(static_cast<size_t>(mu_limit + 1), 0);
        chunks.resize(static_cast<size_t>(num_chunks));
        checkpoints.resize(static_cast<size_t>(num_chunks * 4));
        std::vector<int32_t> block_sums(static_cast<size_t>(num_blocks + 1));

        const int64 root = isqrt(u_total);
        std::vector<uint8_t> prime_flags(static_cast<size_t>(root + 1), 1);
        std::vector<int64> primes;
        std::vector<uint8_t> weights;
        for (int64 p = 2; p <= root; ++p) {
            if (!prime_flags[static_cast<size_t>(p)]) continue;
            primes.push_back(p);
            weights.push_back(static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(p - 1)) | 1u));
            for (int64 n = p * p; n <= root; n += p) prime_flags[static_cast<size_t>(n)] = 0;
        }
        // lcm(2,3,5,7,11,4,9) preserves both weights and square flags.
        static const auto stencil = [] {
            std::vector<uint8_t> values(13860);
            for (int n = 1; n <= 13860; ++n) {
                for (unsigned p : {2u, 3u, 5u, 7u, 11u})
                    if (n % p == 0) values[n - 1] += static_cast<uint8_t>(std::bit_width(p - 1) | 1u);
                if (n % 4 == 0 || n % 9 == 0) values[n - 1] |= 0x80;
            }
            return values;
        }();
        size_t first_square = 0;
        while (first_square < primes.size() && primes[first_square] < 5) ++first_square;
        size_t first_weight = 0;
        while (first_weight < primes.size() && primes[first_weight] < 13) ++first_weight;

        // Allocate scratch on the calling thread so allocation failures can
        // propagate normally instead of escaping an OpenMP worker region.
        std::vector<uint8_t> scratch(static_cast<size_t>(threads) * static_cast<size_t>(BLOCK));
        #pragma omp parallel num_threads(threads)
        {
#ifdef _OPENMP
            const int worker = omp_get_thread_num();
#else
            const int worker = 0;
#endif
            uint8_t* buffer = scratch.data() + static_cast<size_t>(worker) * static_cast<size_t>(BLOCK);
            #pragma omp for schedule(dynamic, 1)
            for (int64 block = 0; block < num_blocks; ++block) {
                const bool dense = block < dense_blocks;
                const int64 base = dense ? block * BLOCK : u_dense + (block - dense_blocks) * BLOCK;
                const int64 end = std::min(base + BLOCK, dense ? u_dense : u_total);
                const int64 len = end - base;
                const int64 low = base + 1;
                int64 done = 0;
                while (done < len) {
                    const size_t phase = static_cast<size_t>((base + done) % 13860);
                    const size_t count = static_cast<size_t>(std::min<int64>(len - done, 13860 - phase));
                    std::memcpy(buffer + done, stencil.data() + phase, count);
                    done += static_cast<int64>(count);
                }
                for (size_t pi = first_square; pi < primes.size(); ++pi) {
                    const int64 square = primes[pi] * primes[pi];
                    if (square > end) break;
                    const int64 first = ((low + square - 1) / square) * square - low;
                    for (int64 i = first; i < len; i += square) buffer[i] |= 0x80;
                }
                for (size_t pi = first_weight; pi < primes.size(); ++pi) {
                    const int64 p = primes[pi];
                    const uint8_t weight = weights[pi];
                    const int64 first = ((low + p - 1) / p) * p - low;
                    for (int64 i = first; i < len; i += p) buffer[i] += weight;
                }
                finalize_mobius_bytes<1>(buffer, len, low);
                // The log threshold needs explicit values for 1 and 2.
                if (base == 0) buffer[0] = 1;
                if (low <= 2 && end >= 2) buffer[static_cast<size_t>(2 - low)] = 255;
                if (base < mu_limit) {
                    const size_t count = static_cast<size_t>(std::min(end, mu_limit) - base);
                    std::memcpy(mu.data() + low, buffer, count);
                }
                const int8_t* values = reinterpret_cast<const int8_t*>(buffer);
                int32_t sum = 0;
                if (dense) {
                    for (int64 n = 0; n < len; ++n) {
                        sum += values[n];
                        dense_M[static_cast<size_t>(low + n)] = static_cast<int16_t>(sum);
                    }
                } else {
                    for (int64 off = 0; off < len; off += 256) {
                        const int64 ci = (base - u_dense + off) / 256;
                        BitChunk& chunk = chunks[static_cast<size_t>(ci)];
                        std::memset(&chunk, 0, sizeof(chunk));
                        const int64 count = std::min<int64>(256, len - off);
                        int64 n = 0;
#if defined(MERTENS_SIMD_SIEVE)
                        const __m128i zero = _mm_setzero_si128();
                        const __m128i ones = _mm_set1_epi8(1);
                        for (; n + 15 < count; n += 16) {
                            if ((n & 63) == 0) checkpoints[static_cast<size_t>(ci * 4 + n / 64)] = sum;
                            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + off + n));
                            const uint16_t nonzero = static_cast<uint16_t>(~_mm_movemask_epi8(_mm_cmpeq_epi8(v, zero)));
                            const uint16_t negative = static_cast<uint16_t>(_mm_movemask_epi8(v));
                            std::memcpy(chunk.nonzero + n / 8, &nonzero, sizeof(nonzero));
                            std::memcpy(chunk.sign + n / 8, &negative, sizeof(negative));
                            const __m128i totals = _mm_sad_epu8(_mm_add_epi8(v, ones), zero);
                            sum += _mm_cvtsi128_si32(totals) + _mm_cvtsi128_si32(_mm_srli_si128(totals, 8)) - 16;
                        }
#endif
                        for (; n < count; ++n) {
                            if ((n & 63) == 0) checkpoints[static_cast<size_t>(ci * 4 + n / 64)] = sum;
                            const int8_t v = values[off + n];
                            chunk.nonzero[n / 8] |= static_cast<uint8_t>((v != 0) << (n & 7));
                            chunk.sign[n / 8] |= static_cast<uint8_t>((v < 0) << (n & 7));
                            sum += v;
                        }
                    }
                }
                block_sums[static_cast<size_t>(block + 1)] = sum;
            }
        }
        // Convert local prefixes/checkpoints to global Mertens values.
        for (int64 b = 1; b <= num_blocks; ++b)
            block_sums[static_cast<size_t>(b)] += block_sums[static_cast<size_t>(b - 1)];
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int64 b = 0; b < num_blocks; ++b) {
            const int32_t offset = block_sums[static_cast<size_t>(b)];
            if (b < dense_blocks) {
                const int64 base = b * BLOCK;
                const int64 end = std::min(base + BLOCK, u_dense);
                for (int64 n = base + 1; n <= end; ++n)
                    dense_M[static_cast<size_t>(n)] = static_cast<int16_t>(dense_M[static_cast<size_t>(n)] + offset);
            } else {
                const int64 first_word = (b - dense_blocks) * (BLOCK / 64);
                const int64 last_word = std::min(first_word + BLOCK / 64, (compressed_size + 63) / 64);
                for (int64 w = first_word; w < last_word; ++w) checkpoints[static_cast<size_t>(w)] += offset;
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

        if (X <= 50000000LL) {
            const int64 half = (X + 1) / 2;
            std::vector<int8_t> mu_odd(static_cast<size_t>(half + 1));
            sieve_mu_odd_byte_log(mu_odd.data(), half, X, threads);
            // M(X) = sum_{odd n <= X} mu(n) - sum_{odd n <= X/2} mu(n).
            // The common prefix cancels, leaving a contiguous suffix.
            const int64 first = (X / 2 + 1) / 2 + 1;
            int64 sum = 0;
            #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(threads)
            for (int64 i = first; i <= half; ++i) sum += mu_odd[i];
            return sum;
        }

        // Use the streaming table for large x86 inputs. Explicit dense
        // tuning preserves the dense path; MERTENS_USE_CHMT overrides it.
        const bool compressed = std::getenv("MERTENS_USE_CHMT")
                              ? use_chmt_env() : use_compressed_default(X);
        if (X >= COMPRESSED_THRESHOLD && compressed) {
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

#if defined(MERTENS_NEON_DIV)
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
            // Packed x86 dense reads, with the scalar loop retained elsewhere.
            // The helper preserves integer division for large y.
#if defined(MERTENS_SSE2_DIV)
            S1 += sum_dense_quotients<1>(y, n, kappa_y, M_ptr);
#else
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
#endif

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

#if defined(MERTENS_NEON_DIV)
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
            // Packed x86 odd-quotient reads, or the scalar unrolled fallback.
#if defined(MERTENS_SSE2_DIV)
            S1_diff += sum_dense_quotients<2>(y, n, kappa_y, M_ptr);
#else
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
#endif

            // ---------------------------------------------------------------------------
            // Even-n correction: pure SIMD increments and vcvtq (+scalar unrolled fast_div)
            // ---------------------------------------------------------------------------
            if (two_kappa_y2 > kappa_y) {
                int64 start_even = kappa_y + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
#if defined(MERTENS_NEON_DIV)
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
#if defined(MERTENS_SSE2_DIV)
                S1_diff -= sum_dense_quotients<2>(y, ne, two_kappa_y2, M_ptr);
#else
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
#endif
            } else if (two_kappa_y2 < kappa_y) {
                int64 start_even = two_kappa_y2 + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
#if defined(MERTENS_NEON_DIV)
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
#if defined(MERTENS_SSE2_DIV)
                S1_diff += sum_dense_quotients<2>(y, ne, kappa_y, M_ptr);
#else
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
#endif
            }

            int64 S2_diff = eval_s2_combined(y, A, B, mu_ptr);
            return -S1_diff + (kappa_y * static_cast<int64>(M_ptr[A]) - kappa_y2 * static_cast<int64>(M_ptr[B])) - S2_diff;
        };

        // ---------------------------------------------------------------------------
        // Single merged parallel loop with dynamic scheduling.
        // Chunk is tunable at compile time (default 64). Smaller chunks
        // balance the heavy head terms better; larger chunks cut dispatch.
        // ---------------------------------------------------------------------------
#ifndef MERTENS_DYN_CHUNK
#define MERTENS_DYN_CHUNK 64
#endif
        #pragma omp parallel for reduction(+:total_M) schedule(dynamic, MERTENS_DYN_CHUNK) num_threads(threads)
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
     * Large-input streaming strategy. The dense prefix stays small while
     * the compressed tier raises u, reducing the outer summation X/u.
     * x86 defaults to this path from 1e11; MERTENS_USE_CHMT=0 forces dense.
     */
    static constexpr int64 COMPRESSED_THRESHOLD = 100000000000LL; // 10^11
    static constexpr int64 DENSE_LIMIT = 128000000LL; // dense prefix target

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
        int64 cap = 8000000000LL;
        if (const char* value = std::getenv("MERTENS_CHMT_U_CAP")) {
            const int64 requested = std::atoll(value);
            if (requested >= 100000000LL && requested <= 10000000000LL) cap = requested;
        }
        const int64 endpoints = static_cast<int64>(choose_cx(X) * std::sqrt(static_cast<double>(X))) + 1000;
        cap = std::max(cap, std::max(S + 1000, endpoints));
        return std::min(u, cap);
    }

    static bool use_compressed_default(int64 X) noexcept {
#if defined(MERTENS_HAVE_SSE2)
        return X >= COMPRESSED_THRESHOLD && !std::getenv("MERTENS_FX") && !std::getenv("MERTENS_U_CAP");
#else
        (void)X;
        return false;
#endif
    }

    static bool use_chmt_env() noexcept {
        if (const char* e = std::getenv("MERTENS_USE_CHMT")) {
            return (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' || e[0] == 'T');
        }
        return false;
    }

    /**
     * Streamed compressed-table computation, also callable explicitly.
     * The dense prefix includes every A/B endpoint; larger quotients use
     * two-popcount reconstruction. S2 keeps the exact SIMD/scalar kernels.
     */
    static int64 compute_compressed(int64 X, int threads) {
        if (X <= 100000LL) return compute_mertens(X, threads);
        if (threads <= 0) {
#ifdef _OPENMP
            threads = omp_get_max_threads();
#else
            threads = 1;
#endif
        }
        int64 u_total = choose_compressed_limit(X);
        int64 endpoint_limit = static_cast<int64>(choose_cx(X) * std::sqrt(static_cast<double>(X))) + 1000;
        endpoint_limit = (endpoint_limit + 255) & ~255LL;
        const int64 dense_target = std::min(DENSE_LIMIT, ((u_total / 4 + 255) & ~255LL));
        int64 u_dense = std::min(u_total, std::max(dense_target, endpoint_limit));
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

            // floor(y/n) > u_dense iff n <= floor(y/(u_dense+1)).
            int64 split_n = std::min(kappa_y, y / (u_dense + 1));
            for (int64 n = start_n; n <= split_n; ++n) {
                int64 q = fast_div(y, dy, n);
                S1 += table.reconstruct_compressed(q);
            }
            int64 n = std::max(start_n, split_n + 1);
#if defined(MERTENS_SSE2_DIV)
            S1 += sum_dense_quotients<1>(y, n, kappa_y, M_dense);
#else
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
#endif
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

            int64 split_odd = std::min(kappa_y, y / (u_dense + 1));
            int64 n = start_odd;
            for (; n <= split_odd; n += 2) {
                int64 q = fast_div(y, dy, n);
                S1_diff += table.reconstruct_compressed(q);
            }
#if defined(MERTENS_SSE2_DIV)
            S1_diff += sum_dense_quotients<2>(y, n, kappa_y, M_dense);
#else
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
#endif

            if (two_kappa_y2 > kappa_y) {
                int64 start_even = kappa_y + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
#if defined(MERTENS_SSE2_DIV)
                S1_diff -= sum_dense_quotients<2>(y, ne, two_kappa_y2, M_dense);
#else
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
#endif
            } else if (two_kappa_y2 < kappa_y) {
                int64 start_even = two_kappa_y2 + 1;
                if (start_even % 2 != 0) ++start_even;
                int64 ne = start_even;
#if defined(MERTENS_SSE2_DIV)
                S1_diff += sum_dense_quotients<2>(y, ne, kappa_y, M_dense);
#else
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
#endif
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
