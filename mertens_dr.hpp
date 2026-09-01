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
 * High-performance odd-only linear sieve for Mobius and Mertens.
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
        std::vector<int8_t> mu_odd(static_cast<size_t>(half_u + 1), 0);
        std::vector<int> primes;
        primes.reserve(static_cast<size_t>(u / 12));
        std::vector<uint8_t> is_prime(static_cast<size_t>(half_u + 1), 1);
        
        int8_t* __restrict__ mu_odd_ptr = mu_odd.data();
        uint8_t* __restrict__ is_prime_ptr = is_prime.data();
        
        mu_odd_ptr[0] = 0;
        mu_odd_ptr[1] = 1;

        for (int i = 2; 2 * i - 1 <= u; ++i) {
            int num = 2 * i - 1;
            if (is_prime_ptr[i]) {
                primes.push_back(num);
                mu_odd_ptr[i] = -1;
            }
            int8_t mu_i = mu_odd_ptr[i];
            const size_t num_primes = primes.size();
            const int* __restrict__ p_ptr = primes.data();
            for (size_t j = 0; j < num_primes; ++j) {
                int p = p_ptr[j];
                int64 prod = static_cast<int64>(num) * p;
                if (prod > u) break;
                int idx = static_cast<int>((prod + 1) / 2);
                is_prime_ptr[idx] = 0;
                if (num % p == 0) {
                    mu_odd_ptr[idx] = 0;
                    break;
                } else {
                    mu_odd_ptr[idx] = -mu_i;
                }
            }
        }

        // Expand mu table for all n <= u in parallel
        int8_t* __restrict__ mu_full_ptr = mu.data();
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int64 i = 1; i <= u; ++i) {
            int8_t m;
            if (i & 1) {
                m = mu_odd_ptr[(i + 1) >> 1];
            } else if ((i >> 1) & 1) {
                m = -mu_odd_ptr[((i >> 1) + 1) >> 1];
            } else {
                m = 0;
            }
            mu_full_ptr[i] = m;
        }

        // Prefix sum for M table
        int16_t* __restrict__ M_ptr = M.data();
        int32_t run_sum = 0;
        for (int64 i = 1; i <= u; ++i) {
            run_sum += mu_full_ptr[i];
            M_ptr[i] = static_cast<int16_t>(run_sum);
        }
    }
};

/**
 * Branchless, Modulo-Free, NEON-Vectorized S2 Interval Runner.
 * Processes 2 mod-6 pairs per iteration (4 values: j1, j2, j3, j4) with parallel double division.
 */
template <typename F>
inline int64 run_s2_fast(int64 j_start, int64 j_end, double dy, const int8_t* __restrict mu_ptr, F&& summand_fn) noexcept {
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
                sum += static_cast<int64>(m) * summand_fn(q);
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

            if (m1) sum += static_cast<int64>(m1) * summand_fn(q1);
            if (m2) sum += static_cast<int64>(m2) * summand_fn(q2);
            if (m3) sum += static_cast<int64>(m3) * summand_fn(q3);
            if (m4) sum += static_cast<int64>(m4) * summand_fn(q4);
        }
    }
    for (; m < m_end; ++m) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        if (m1) sum += static_cast<int64>(m1) * summand_fn(static_cast<int64>(dy / static_cast<double>(j1)));
        if (m2) sum += static_cast<int64>(m2) * summand_fn(static_cast<int64>(dy / static_cast<double>(j2)));
    }
#else
    for (int64 m = m_start; m < m_end; ++m) {
        int64 j1 = 6 * m + 1; int64 j2 = 6 * m + 5;
        int8_t m1 = mu_ptr[j1]; int8_t m2 = mu_ptr[j2];
        if (m1) sum += static_cast<int64>(m1) * summand_fn(static_cast<int64>(dy / static_cast<double>(j1)));
        if (m2) sum += static_cast<int64>(m2) * summand_fn(static_cast<int64>(dy / static_cast<double>(j2)));
    }
#endif

    // Tail Boundary
    int64 tail_start = std::max(j_start, m_end * 6 + 1);
    for (int64 j = tail_start; j <= j_end; ++j) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = mu_ptr[j];
            if (m != 0) {
                int64 q = static_cast<int64>(dy / static_cast<double>(j));
                sum += static_cast<int64>(m) * summand_fn(q);
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

    // Piece 1: 1 <= j <= b6
    sum += run_s2_fast(1, b6, dy, mu_ptr, [](int64 q) noexcept {
        return (q / 4) + (q & 1) - (q / 12) - ((q / 3) & 1);
    });

    // Piece 2: b6 < j <= a6
    sum += run_s2_fast(b6 + 1, a6, dy, mu_ptr, [](int64 q) noexcept {
        return (q / 4) + (q & 1) - ((q % 6 >= 3) ? 1 : 0);
    });

    // Piece 3: a6 < j <= b3
    sum += run_s2_fast(a6 + 1, b3, dy, mu_ptr, [](int64 q) noexcept {
        return (q / 4) + (q & 1) - ((q + 3) / 6);
    });

    // Piece 4: b3 < j <= a3
    sum += run_s2_fast(b3 + 1, a3, dy, mu_ptr, [](int64 q) noexcept {
        return (q / 4) + (q & 1) - (q / 3);
    });

    // Piece 5: a3 < j <= b2
    sum += run_s2_fast(a3 + 1, b2, dy, mu_ptr, [](int64 q) noexcept {
        return (q / 4) + (q & 1);
    });

    // Piece 6: b2 < j <= a2
    sum += run_s2_fast(b2 + 1, a2, dy, mu_ptr, [](int64 q) noexcept {
        return (q & 1);
    });

    // Piece 7: a2 < j <= B
    sum += run_s2_fast(a2 + 1, B, dy, mu_ptr, [](int64 q) noexcept {
        return (q + 1) / 2;
    });

    // Piece 8: B < j <= A
    sum += run_s2_fast(B + 1, A, dy, mu_ptr, [](int64 q) noexcept {
        return q;
    });

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

    sum += run_s2_fast(1, a6, dy, mu_ptr, [](int64 q) noexcept {
        return (((q + 1) / 2) - (((q / 3) + 1) / 2));
    });
    sum += run_s2_fast(a6 + 1, a3, dy, mu_ptr, [](int64 q) noexcept {
        return (((q + 1) / 2) - (q / 3));
    });
    sum += run_s2_fast(a3 + 1, a2, dy, mu_ptr, [](int64 q) noexcept {
        return ((q + 1) / 2);
    });
    sum += run_s2_fast(a2 + 1, A, dy, mu_ptr, [](int64 q) noexcept {
        return q;
    });

    return sum;
}

/**
 * Master Deléglise-Rivat Mertens Solver.
 */
class DelégliseRivatEngine {
public:
    /**
     * Dynamically chooses optimal sieve cutoff u for target X.
     */
    static inline int64 choose_sieve_limit(int64 X, int threads) {
        (void)threads;
        if (X <= 50000000LL) {
            return X; // Direct linear sieve for X <= 50M
        }

        double loglogX = std::log(std::max(2.0, std::log(static_cast<double>(X))));
        double fx = 0.85;
        if (X >= 1000000000000000LL) {
            fx = 1.05;
        }

        int64 u = static_cast<int64>(fx * std::pow(static_cast<double>(X) / loglogX, 2.0 / 3.0));
        int64 S = isqrt(X);
        if (u < 3 * S) u = 3 * S;
        
        // Cap u at 450 Million (~900 MB RAM) for optimal multi-core cache throughput
        return std::min(u, 450000000LL);
    }

    /**
     * Computes the Mertens Function M(X) = sum_{n=1}^X mu(n) in O(X^(2/3)).
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

        const double cx = 0.70; // Optimal balance for superscalar SIMD pipelines
        const int64 N = X / u;
        const int64 N_half = N / 2;
        const int16_t* __restrict M_ptr = table.data();
        const int8_t* __restrict mu_ptr = table.mu.data();

        // Collect all odd square-free k <= N
        std::vector<int64> odd_k_comb;
        std::vector<int64> odd_k_single;
        odd_k_comb.reserve(static_cast<size_t>(N_half / 2));
        odd_k_single.reserve(static_cast<size_t>((N - N_half) / 2));

        for (int64 k = 1; k <= N; k += 2) {
            if (mu_ptr[k] != 0) {
                if (k <= N_half) {
                    odd_k_comb.push_back(k);
                } else {
                    odd_k_single.push_back(k);
                }
            }
        }

        int64 total_M = 0;

        // Process Combined Range: k <= N/2
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_M) schedule(guided) num_threads(threads)
        #endif
        for (size_t idx = 0; idx < odd_k_comb.size(); ++idx) {
            int64 k = odd_k_comb[idx];
            int8_t mu_k = mu_ptr[k];
            int64 y = X / k;
            int64 y2 = y / 2;

            int64 A = static_cast<int64>(cx * std::sqrt(static_cast<double>(y)));
            int64 B = static_cast<int64>(cx * std::sqrt(static_cast<double>(y2)));
            if (A >= y) A = y - 1;
            if (B >= y2) B = y2 - 1;
            if (A < 1) A = 1;
            if (B < 1) B = 1;

            int64 kappa_y = y / (A + 1);
            int64 kappa_y2 = y2 / (B + 1);
            int64 two_kappa_y2 = 2 * kappa_y2;

            // 1. S1(y, u) - S1(y/2, u) via formula (6)
            int64 S1_diff = 0;
            int64 start_odd = (y / u + 1);
            if (start_odd % 2 == 0) ++start_odd;

            double dy = static_cast<double>(y);
            int64 n = start_odd;

#if defined(__ARM_NEON)
            float64x2_t v_dy = vdupq_n_f64(dy);
            for (; n + 7 <= kappa_y; n += 8) {
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
#else
            for (; n + 7 <= kappa_y; n += 8) {
                int64 q0 = static_cast<int64>(dy / static_cast<double>(n));
                int64 q1 = static_cast<int64>(dy / static_cast<double>(n + 2));
                int64 q2 = static_cast<int64>(dy / static_cast<double>(n + 4));
                int64 q3 = static_cast<int64>(dy / static_cast<double>(n + 6));
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

            // 2. S2(y) - S2(y/2) mod 6 piecewise evaluation
            int64 S2_diff = eval_s2_combined(y, A, B, mu_ptr);

            // S(y, u) - S(y/2, u)
            int64 term = -S1_diff + (kappa_y * static_cast<int64>(M_ptr[A]) - kappa_y2 * static_cast<int64>(M_ptr[B])) - S2_diff;
            total_M += static_cast<int64>(mu_k) * term;
        }

        // Process Single Range: N/2 < k <= N
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_M) schedule(guided) num_threads(threads)
        #endif
        for (size_t idx = 0; idx < odd_k_single.size(); ++idx) {
            int64 k = odd_k_single[idx];
            int8_t mu_k = mu_ptr[k];
            int64 y = X / k;

            int64 A = static_cast<int64>(cx * std::sqrt(static_cast<double>(y)));
            if (A >= y) A = y - 1;
            if (A < 1) A = 1;
            int64 kappa_y = y / (A + 1);

            int64 S1 = 0;
            int64 start_n = y / u + 1;
            double dy = static_cast<double>(y);
            int64 n = start_n;

            for (; n + 7 <= kappa_y; n += 8) {
                int64 q0 = static_cast<int64>(dy / static_cast<double>(n));
                int64 q1 = static_cast<int64>(dy / static_cast<double>(n + 1));
                int64 q2 = static_cast<int64>(dy / static_cast<double>(n + 2));
                int64 q3 = static_cast<int64>(dy / static_cast<double>(n + 3));
                int64 q4 = static_cast<int64>(dy / static_cast<double>(n + 4));
                int64 q5 = static_cast<int64>(dy / static_cast<double>(n + 5));
                int64 q6 = static_cast<int64>(dy / static_cast<double>(n + 6));
                int64 q7 = static_cast<int64>(dy / static_cast<double>(n + 7));

                S1 += M_ptr[q0] + M_ptr[q1] + M_ptr[q2] + M_ptr[q3] +
                      M_ptr[q4] + M_ptr[q5] + M_ptr[q6] + M_ptr[q7];
            }
            for (; n <= kappa_y; ++n) {
                S1 += M_ptr[static_cast<int64>(dy / static_cast<double>(n))];
            }

            int64 S2 = eval_s2_single(y, A, mu_ptr);

            int64 S_val = 1 - S1 + kappa_y * static_cast<int64>(M_ptr[A]) - S2;
            total_M += static_cast<int64>(mu_k) * S_val;
        }

        return total_M;
    }
};

} // namespace mertens_dr

#endif // MERTENS_DR_HPP
