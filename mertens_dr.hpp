#ifndef MERTENS_DR_HPP
#define MERTENS_DR_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
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

// ---------------------------------------------------------------------------
// Standalone parallel segmented Mobius sieve (extracted for reuse)
// ---------------------------------------------------------------------------

/**
 * Sieves mu_odd[1..half_u] where mu_odd[k] = mu(2k-1) for odd numbers.
 * mu_odd must be pre-allocated with size >= half_u + 1.
 */
template <uint64_t P>
inline void sieve_prime_const(int8_t* __restrict__ b_mu, uint64_t* __restrict__ rem,
                              int64 len, int64 low_val) noexcept {
    constexpr uint64_t P2 = P * P;
    int64 start_val = ((low_val + P - 1) / P) * P;
    if (start_val % 2 == 0) start_val += P;
    int64 start_i = (start_val - low_val) / 2;

    int64 start_sq_val = ((low_val + P2 - 1) / P2) * P2;
    if (start_sq_val % 2 == 0) start_sq_val += P2;
    int64 start_sq_i = (start_sq_val - low_val) / 2;

    for (int64 i = start_sq_i; i < len; i += P2) {
        b_mu[i] = 0;
    }

    for (int64 i = start_i; i < len; i += P) {
        if (b_mu[i] != 0) {
            b_mu[i] = -b_mu[i];
            rem[i] /= P;
            while (rem[i] % P == 0) {
                rem[i] /= P;
                b_mu[i] = 0;
            }
        }
    }
}

/**
 * Sieves mu_odd[1..half_u] where mu_odd[k] = mu(2k-1) for odd numbers.
 * mu_odd must be pre-allocated with size >= half_u + 1.
 */
inline void sieve_mu_odd(int8_t* __restrict__ mu_odd, int64 half_u, int64 u,
                         int threads) {
    int sqrt_u = static_cast<int>(std::sqrt(static_cast<double>(u))) + 1;
    std::vector<int> primes;
    std::vector<uint8_t> is_p(sqrt_u + 1, 1);
    for (int64 i = 2; i <= sqrt_u; ++i) {
        if (is_p[i]) {
            if (i > 71) primes.push_back(static_cast<int>(i));
            for (int64 j = i * i; j <= sqrt_u; j += i) is_p[j] = 0;
        }
    }

    mu_odd[0] = 0;
    for (int64 i = 1; i <= half_u; ++i) mu_odd[i] = 1;

    const int64 BLOCK_SIZE = 131072;
    int64 num_blocks = (half_u + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<uint64_t> rem(BLOCK_SIZE);
        #pragma omp for schedule(dynamic, 4)
        for (int64 b = 0; b < num_blocks; ++b) {
            int64 low_idx = b * BLOCK_SIZE + 1;
            int64 high_idx = std::min(half_u, (b + 1) * BLOCK_SIZE);
            int64 len = high_idx - low_idx + 1;
            if (len <= 0) continue;

            for (int64 i = 0; i < len; ++i) {
                rem[i] = static_cast<uint64_t>(2 * (low_idx + i) - 1);
            }

            int8_t* __restrict__ b_mu = &mu_odd[low_idx];
            uint64_t* __restrict__ b_rem = rem.data();
            int64 low_val = 2 * low_idx - 1;

            // Small primes with 1-cycle compile-time constant division
            sieve_prime_const<3>(b_mu, b_rem, len, low_val);
            sieve_prime_const<5>(b_mu, b_rem, len, low_val);
            sieve_prime_const<7>(b_mu, b_rem, len, low_val);
            sieve_prime_const<11>(b_mu, b_rem, len, low_val);
            sieve_prime_const<13>(b_mu, b_rem, len, low_val);
            sieve_prime_const<17>(b_mu, b_rem, len, low_val);
            sieve_prime_const<19>(b_mu, b_rem, len, low_val);
            sieve_prime_const<23>(b_mu, b_rem, len, low_val);
            sieve_prime_const<29>(b_mu, b_rem, len, low_val);
            sieve_prime_const<31>(b_mu, b_rem, len, low_val);
            sieve_prime_const<37>(b_mu, b_rem, len, low_val);
            sieve_prime_const<41>(b_mu, b_rem, len, low_val);
            sieve_prime_const<43>(b_mu, b_rem, len, low_val);
            sieve_prime_const<47>(b_mu, b_rem, len, low_val);
            sieve_prime_const<53>(b_mu, b_rem, len, low_val);
            sieve_prime_const<59>(b_mu, b_rem, len, low_val);
            sieve_prime_const<61>(b_mu, b_rem, len, low_val);
            sieve_prime_const<67>(b_mu, b_rem, len, low_val);
            sieve_prime_const<71>(b_mu, b_rem, len, low_val);

            // Remaining larger primes
            for (int p : primes) {
                int64 p2 = static_cast<int64>(p) * p;

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
                        b_rem[i] /= p;
                        while (b_rem[i] % p == 0) {
                            b_rem[i] /= p;
                            b_mu[i] = 0;
                        }
                    }
                }
            }

            for (int64 i = 0; i < len; ++i) {
                if (b_mu[i] != 0 && b_rem[i] > 1) {
                    b_mu[i] = -b_mu[i];
                }
            }
        }
    }
}

/**
 * Derives mu(n) from mu_odd array.
 */
inline int8_t mu_from_odd(int64 n, const int8_t* __restrict__ mu_odd) noexcept {
    if (n & 1) {
        return mu_odd[(n + 1) >> 1];
    } else if ((n >> 1) & 1) {
        return -mu_odd[((n >> 1) + 1) >> 1];
    } else {
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Log-weight segmented Mobius sieve (division-free inner loops, Hurst-style)
// Stores sum of ceil(log2 p)|1 per hit + square flag; finalizes via
// comparison with floor(log2 n). Eliminates rem[] divisions and 1MB/block
// rem traffic. Max weight sum for u<=8B is <50, fits uint8.
// ---------------------------------------------------------------------------
inline int clog2_int(int p) noexcept {
    return 64 - __builtin_clzll((unsigned long long)(p - 1));
}
inline int flog2_64(int64 n) noexcept {
    return 63 - __builtin_clzll((unsigned long long)n);
}
inline void sieve_mu_odd_log(int8_t* __restrict__ mu_odd, int64 half_u, int64 u,
                             int threads) {
    int sqrt_u = static_cast<int>(std::sqrt(static_cast<double>(u))) + 1;
    std::vector<int> primes;
    std::vector<uint8_t> is_p(static_cast<size_t>(sqrt_u) + 1, 1);
    for (int64 i = 2; i <= sqrt_u; ++i) {
        if (is_p[static_cast<size_t>(i)]) {
            if (i > 71) primes.push_back(static_cast<int>(i));
            if (i * i <= sqrt_u) for (int64 j = i * i; j <= sqrt_u; j += i) is_p[static_cast<size_t>(j)] = 0;
        }
    }
    mu_odd[0] = 0;
    const int64 BLOCK = 131072;
    int64 num_blocks = (half_u + BLOCK - 1) / BLOCK;
    const int small_arr[19] = {3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71};
    int wsmall[19];
    for (int i = 0; i < 19; ++i) wsmall[i] = clog2_int(small_arr[i]) | 1;
    std::vector<int> wprime(primes.size());
    for (size_t i = 0; i < primes.size(); ++i) wprime[i] = clog2_int(primes[i]) | 1;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<uint8_t> lg(static_cast<size_t>(BLOCK));
        std::vector<uint8_t> is_zero(static_cast<size_t>(BLOCK));
        #pragma omp for schedule(dynamic, 4)
        for (int64 b = 0; b < num_blocks; ++b) {
            int64 low_idx = b * BLOCK + 1;
            int64 high_idx = std::min(half_u, (b + 1) * BLOCK);
            int64 len = high_idx - low_idx + 1;
            if (len <= 0) continue;
            memset(lg.data(), 0, static_cast<size_t>(len));
            memset(is_zero.data(), 0, static_cast<size_t>(len));
            int64 low_val = 2 * low_idx - 1;
            for (int s = 0; s < 19; ++s) {
                int64 P = small_arr[s];
                int64 P2 = P * P;
                int w = wsmall[s];
                int64 start = ((low_val + P - 1) / P) * P;
                if ((start & 1) == 0) start += P;
                int64 si = (start - low_val) / 2;
                for (int64 i = si; i < len; i += P) lg[static_cast<size_t>(i)] += static_cast<uint8_t>(w);
                int64 start2 = ((low_val + P2 - 1) / P2) * P2;
                if ((start2 & 1) == 0) start2 += P2;
                int64 si2 = (start2 - low_val) / 2;
                for (int64 i = si2; i < len; i += P2) is_zero[static_cast<size_t>(i)] = 1;
            }
            for (size_t pi = 0; pi < primes.size(); ++pi) {
                int64 P = primes[pi];
                int64 P2 = P * P;
                int w = wprime[pi];
                int64 start = ((low_val + P - 1) / P) * P;
                if ((start & 1) == 0) start += P;
                int64 si = (start - low_val) / 2;
                if (si >= len) continue;
                for (int64 i = si; i < len; i += P) lg[static_cast<size_t>(i)] += static_cast<uint8_t>(w);
                int64 start2 = ((low_val + P2 - 1) / P2) * P2;
                if ((start2 & 1) == 0) start2 += P2;
                int64 si2 = (start2 - low_val) / 2;
                for (int64 i = si2; i < len; i += P2) is_zero[static_cast<size_t>(i)] = 1;
            }
            int8_t* out = &mu_odd[low_idx];
            for (int64 i = 0; i < len; ++i) {
                if (is_zero[static_cast<size_t>(i)]) { out[i] = 0; continue; }
                int64 n = low_val + 2 * i;
                if (n == 1) { out[i] = 1; continue; }
                uint8_t S = lg[static_cast<size_t>(i)];
                int fl = flog2_64(n);
                if (S > static_cast<uint8_t>(fl)) out[i] = (S & 1) ? (int8_t)-1 : (int8_t)1;
                else out[i] = (S & 1) ? (int8_t)1 : (int8_t)-1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Original SieveTable (for moderate X values, u <= 600M)
// ---------------------------------------------------------------------------

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
    std::vector<int8_t> mu;
    std::vector<int16_t> M; // 2 bytes per entry, zero indirection

    explicit SieveTable(int64 limit, int num_threads = 0, int64 max_mu = 0) : u(limit) {
        int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );

        mu_limit = (max_mu > 0 ? std::min(max_mu, u) : u);
        mu.assign(static_cast<size_t>(mu_limit + 1), 0);
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
        return (t <= mu_limit ? mu[t] : 0);
    }

    inline const int16_t* data() const noexcept {
        return M.data();
    }

private:
    void build_sieve(int threads) {
        int64 half_u = (u + 1) / 2;

        // 1. Sieve mu_odd (log-weight, division-free)
        std::vector<int8_t> mu_odd(static_cast<size_t>(half_u + 1), 0);
        sieve_mu_odd_log(mu_odd.data(), half_u, u, threads);
        const int8_t* __restrict__ mu_odd_ptr = mu_odd.data();

        // 2. Populate small mu table (only up to mu_limit)
        int8_t* __restrict__ mu_ptr = mu.data();
        for (int64 i = 1; i <= mu_limit; ++i) {
            mu_ptr[i] = mu_from_odd(i, mu_odd_ptr);
        }

        // 3. Parallel direct prefix sum for M from mu_odd
        int16_t* __restrict__ M_ptr = M.data();
        int num_t = threads;
        std::vector<int32_t> block_sums(num_t + 1, 0);

        #pragma omp parallel num_threads(num_t)
        {
            int tid = omp_get_thread_num();
            int64 chunk = (u + num_t - 1) / num_t;
            chunk = (chunk + 3) & ~3LL;
            int64 start = tid * chunk + 1;
            int64 end = std::min(u, (tid + 1) * chunk);

            int32_t local_sum = 0;
            for (int64 i = start; i <= end; ++i) {
                local_sum += mu_from_odd(i, mu_odd_ptr);
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
            chunk = (chunk + 3) & ~3LL;
            int64 start = tid * chunk + 1;
            int64 end = std::min(u, (tid + 1) * chunk);

            int32_t run_sum = block_sums[tid];
            for (int64 i = start; i <= end; ++i) {
                run_sum += mu_from_odd(i, mu_odd_ptr);
                M_ptr[i] = static_cast<int16_t>(run_sum);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Compressed Hierarchical Mertens Table (CHMT) — The New Algorithm
// ---------------------------------------------------------------------------
//
// Two-tier M(n) storage:
//   Tier 1 (Dense):      int16_t M[0..u_dense]          — direct O(1) lookup
//   Tier 2 (Compressed): 2-bit μ deltas + int32 checkpoints — NEON popcount O(1) lookup
//
// Encoding: For each n in the compressed range, μ(n) ∈ {-1,0,+1} is stored as:
//   nonzero bit = (μ(n) != 0)
//   sign bit    = (μ(n) < 0)
//
// Reconstruction: M(q) = checkpoint[block] + popcount(positive bits) - popcount(negative bits)
//   where positive = nonzero AND NOT sign, negative = nonzero AND sign.
//
// Each checkpoint block covers 256 consecutive values = 32 bytes per bit-plane = 64 bytes total.
// The BitChunk struct fits exactly in one 64-byte cache line.

struct alignas(64) BitChunk {
    uint8_t nonzero[32]; // nonzero bit-plane: bit i = (μ(base + i) != 0)
    uint8_t sign[32];    // sign bit-plane: bit i = (μ(base + i) < 0)
};

class CompressedMertensTable {
public:
    int64 u_dense;     // Dense tier limit
    int64 u_total;     // Total effective sieve limit
    int64 mu_limit;    // Limit for the mu array (for S2)

    std::vector<int16_t> dense_M;     // M[0..u_dense], 2 bytes/entry
    std::vector<int8_t> mu;           // mu[0..mu_limit], 1 byte/entry

    int64 compressed_size;  // = u_total - u_dense
    int64 num_chunks;       // = ceil(compressed_size / 256)
    std::vector<BitChunk> chunks;
    std::vector<int32_t> checkpoints; // M value at start of each 256-value block

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
        compressed_size = std::max(0LL, u_total - u_dense);
        num_chunks = (compressed_size > 0 ? (compressed_size + 255) / 256 : 0);
#ifdef DR_VERBOSE
        std::cerr << "[CHMT] Initializing with u_total=" << u_total << ", u_dense=" << u_dense << ", threads=" << threads << std::endl;
#endif
        build(threads);
#ifdef DR_VERBOSE
        std::cerr << "[CHMT] Table built successfully." << std::endl;
#endif
    }

    /**
     * Look up M(q) — the core of the CHMT algorithm.
     * For q <= u_dense: direct int16_t lookup (Tier 1).
     * For q > u_dense:  NEON popcount reconstruction (Tier 2).
     */
    inline int32_t get_M(int64 q) const noexcept {
        if (q <= u_dense) {
            return static_cast<int32_t>(dense_M[q]);
        }
        return reconstruct_compressed(q);
    }

    inline int8_t get_mu(int64 j) const noexcept {
        return (j <= mu_limit ? mu[j] : 0);
    }

    inline const int16_t* dense_data() const noexcept {
        return dense_M.data();
    }

    /**
     * Prefetch the cache line(s) needed for a future M(q) lookup.
     */
    inline void prefetch_M(int64 q) const noexcept {
        if (q <= u_dense) {
            __builtin_prefetch(&dense_M[q], 0, 0);
        } else if (q <= u_total) {
            int64 ci = (q - u_dense - 1) >> 8;
            __builtin_prefetch(&chunks[ci], 0, 0);
            __builtin_prefetch(&checkpoints[ci], 0, 0);
        }
    }

    inline int32_t reconstruct_compressed(int64 q) const noexcept {
        int64 offset = q - u_dense;          // 1-indexed into compressed range
        int64 ci = (offset - 1) >> 8;        // chunk index (/ 256)
        int r = static_cast<int>((offset - 1) & 255) + 1; // bits to count (1..256)

        int32_t base = checkpoints[ci];
        const BitChunk& chunk = chunks[ci];

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
            uint64_t mask = (1ULL << rem_bits) - 1ULL;
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
        sieve_mu_odd_log(mu_odd.data(), half_u_total, u_total, threads);
        const int8_t* __restrict__ mu_odd_ptr = mu_odd.data();

        // -------------------------------------------------------------------
        // Phase 2: Build dense M table for n <= u_dense (parallel prefix sum)
        // -------------------------------------------------------------------
        dense_M.assign(static_cast<size_t>(u_dense + 1), 0);
        {
            int16_t* __restrict__ M_ptr = dense_M.data();
            int num_t = threads;
            std::vector<int32_t> block_sums(num_t + 1, 0);

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
                block_sums[tid + 1] = local_sum;
            }
            for (int i = 1; i <= num_t; ++i)
                block_sums[i] += block_sums[i - 1];

            #pragma omp parallel num_threads(num_t)
            {
                int tid = omp_get_thread_num();
                int64 chunk_sz = (u_dense + num_t - 1) / num_t;
                chunk_sz = (chunk_sz + 3) & ~3LL;
                int64 start = tid * chunk_sz + 1;
                int64 end = std::min(u_dense, (tid + 1) * chunk_sz);

                int32_t run_sum = block_sums[tid];
                for (int64 i = start; i <= end; ++i) {
                    run_sum += mu_from_odd(i, mu_odd_ptr);
                    M_ptr[i] = static_cast<int16_t>(run_sum);
                }
            }
        }

        // -------------------------------------------------------------------
        // Phase 3: Build mu table for S2 (up to mu_limit)
        // -------------------------------------------------------------------
        mu.assign(static_cast<size_t>(mu_limit + 1), 0);
        {
            int8_t* __restrict__ mu_ptr = mu.data();
            for (int64 i = 1; i <= mu_limit; ++i) {
                mu_ptr[i] = mu_from_odd(i, mu_odd_ptr);
            }
        }

        // -------------------------------------------------------------------
        // Phase 4: Build compressed tier (2-bit chunks + checkpoints)
        // -------------------------------------------------------------------
        if (num_chunks > 0) {
            chunks.resize(static_cast<size_t>(num_chunks));
            checkpoints.resize(static_cast<size_t>(num_chunks));

            // 4a: Parallel branchless exact 8-element bit-plane packing
            std::vector<int32_t> local_sums(static_cast<size_t>(num_chunks), 0);

            #pragma omp parallel for schedule(static, 256) num_threads(threads)
            for (int64 ci = 0; ci < num_chunks; ++ci) {
                BitChunk& ch = chunks[ci];
                int32_t lsum = 0;
                int64 k_base = (u_dense + ci * 256) >> 3;

                for (int b = 0; b < 32; ++b) {
                    int64 k = k_base + b;
                    int64 idx4 = (k << 2) + 1;
                    int64 idx2 = (k << 1) + 1;

                    int8_t v1 = mu_odd_ptr[idx4];
                    int8_t v2 = -mu_odd_ptr[idx2];
                    int8_t v3 = mu_odd_ptr[idx4 + 1];
                    int8_t v5 = mu_odd_ptr[idx4 + 2];
                    int8_t v6 = -mu_odd_ptr[idx2 + 1];
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
                local_sums[ci] = lsum;
            }

            // 4b: Parallel prefix sum for checkpoints
            checkpoints[0] = static_cast<int32_t>(dense_M[u_dense]);
            for (int64 ci = 1; ci < num_chunks; ++ci) {
                checkpoints[ci] = checkpoints[ci - 1] + local_sums[ci - 1];
            }
        }
#ifdef DR_VERBOSE
        auto tp4_1 = std::chrono::high_resolution_clock::now();
        std::cerr << "[CHMT] Phase 4 done in " << std::chrono::duration<double>(tp4_1 - tp4_0).count() << " s" << std::endl;
        std::cerr << "[CHMT] Total build time: " << std::chrono::duration<double>(tp4_1 - t_start).count() << " s" << std::endl;
#endif

        // mu_odd freed automatically when it goes out of scope
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

struct Piece1 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq % 12; return 2 * static_cast<int64>(k) + LUT_P1[r]; } };
struct Piece2 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq % 12; return 3 * static_cast<int64>(k) + LUT_P2[r]; } };
struct Piece3 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq % 12; return 1 * static_cast<int64>(k) + LUT_P3[r]; } };
struct Piece4 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq % 12; return -1 * static_cast<int64>(k) + LUT_P4[r]; } };
struct Piece5 { static inline int64 eval(int64 q) noexcept { return (q >> 2) + (q & 1); } };
struct Piece6 { static inline int64 eval(int64 q) noexcept { return (q & 1); } };
struct Piece7 { static inline int64 eval(int64 q) noexcept { return (q + 1) >> 1; } };
struct Piece8 { static inline int64 eval(int64 q) noexcept { return q; } };

struct SinglePiece1 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq % 12; return 4 * static_cast<int64>(k) + LUT_S1[r]; } };
struct SinglePiece2 { static inline int64 eval(int64 q) noexcept { uint64 uq = static_cast<uint64>(q); uint64 k = uq / 12; uint64 r = uq % 12; return 2 * static_cast<int64>(k) + LUT_S2[r]; } };
struct SinglePiece3 { static inline int64 eval(int64 q) noexcept { return (q + 1) >> 1; } };
struct SinglePiece4 { static inline int64 eval(int64 q) noexcept { return q; } };

/**
 * Branchless, Modulo-Free, NEON 4-Way Pipelined S2 Interval Runner with LUT Summand.
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

    // Tail Boundary
    int64 tail_start = std::max(j_start, m_end * 6 + 1);
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

/**
 * Master Deléglise-Rivat Mertens Solver.
 */
class DelégliseRivatEngine {
public:
    /**
     * Dynamically chooses optimal balance parameter cx.
     * For large X with compressed table, higher cx shifts work from
     * random-access S1 to sequential S2 (favorable on modern hardware).
     */
    static inline double choose_cx(int64 X, bool compressed = false) noexcept {
        (void)compressed;
        if (X >= 1000000000000000LL) return 1.5;
        if (X >= 10000000000000LL) return 1.5;
        if (X >= 100000000000LL) return 1.5;
        return 0.70;
    }

    /**
     * Threshold for using the compressed hierarchical table.
     * For X < 10^15, u <= 600M fits completely in the dense table.
     */
    static constexpr int64 COMPRESSED_THRESHOLD = 1000000000000000LL; // 10^15

    /**
     * Dense tier limit for compressed table.
     */
    static constexpr int64 DENSE_LIMIT = 600000000LL; // 600M



    /**
     * Choose total sieve limit for CHMT algorithm across all X.
     * Scales from small X up to 8B for massive X.
     */
    static inline int64 choose_compressed_limit(int64 X) {
        if (X < 50000000LL) return X;

        double loglogX = std::log(std::max(2.0, std::log(static_cast<double>(X))));
        double fx = 0.95;
        if (X >= 1000000000000000LL) fx = 1.25;  // 10^15
        if (X >= 10000000000000000LL) fx = 1.50; // 10^16+

        int64 u = static_cast<int64>(fx * std::pow(static_cast<double>(X) / loglogX, 2.0 / 3.0));
        int64 S = isqrt(X);
        if (u < 3 * S) u = 3 * S;

        u = ((u + 255) / 256) * 256;
        return std::min(u, 8000000000LL);
    }

    /**
     * Computes the Mertens Function M(X) = sum_{n=1}^X mu(n) in O(X^(2/3))
     * using the Compressed Hierarchical Mertens Table for large X,
     * or the original dense table for moderate X.
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

        // ---------------------------------------------------------------
        // Unified Compressed Hierarchical Mertens Table Engine (CHMT)
        // ---------------------------------------------------------------
        return compute_compressed(X, threads);
    }



    /**
     * Compressed-table computation path (CHMT algorithm, for X >= 10^14).
     * Uses the two-tier M table with deep prefetching in S1.
     */
    static int64 compute_compressed(int64 X, int threads) {
        int64 u_total = choose_compressed_limit(X);
        int64 u_dense = std::min(u_total, DENSE_LIMIT);
        const double cx = choose_cx(X, (u_total > DENSE_LIMIT));
        const int64 N = X / u_total;

        int64 A_max = static_cast<int64>(cx * std::sqrt(static_cast<double>(X))) + 1000;
        int64 mu_limit = std::max(A_max, N + 1000);

        CompressedMertensTable table(u_total, u_dense, threads, mu_limit);

        if (X <= u_total) {
            return table.get_M(X);
        }

        const int8_t* __restrict mu_ptr = table.mu.data();

        // Build unified term list
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

        // ---------------------------------------------------------------
        // Deep-Prefetch S1 with CompressedMertensTable lookups
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

            // Compressed range (q > u_dense) — exact via fast_div guard (>2^53 uses integer)
            for (int64 n = start_n; n <= split_n; ++n) {
                int64 q = fast_div(y, dy, n);
                S1 += table.reconstruct_compressed(q);
            }

            // Dense range (q <= u_dense): fast 8-way unrolled direct array reads
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

            // S1 odd loop: compressed tier (q > u_dense)
            int64 n = start_odd;
            for (; n <= split_odd; n += 2) {
                int64 q = fast_div(y, dy, n);
                S1_diff += table.reconstruct_compressed(q);
            }

            // S1 odd loop: dense tier (q <= u_dense) with 8-way unrolling
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

            // Even-n correction: all quotients <= A < u_dense, direct dense M reads!
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

        // Parallel dispatch over terms with small dynamic chunks for optimal small-k load balance
        #pragma omp parallel for reduction(+:total_M) schedule(dynamic, 8) num_threads(threads)
        for (size_t i = 0; i < terms.size(); ++i) {
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
};

} // namespace mertens_dr

#endif // MERTENS_DR_HPP
