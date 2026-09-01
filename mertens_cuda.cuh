#ifndef MERTENS_CUDA_CUH
#define MERTENS_CUDA_CUH

#include "mertens_dr.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace mertens_cuda {

using int64 = long long;
using uint64 = unsigned long long;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

/**
 * Exact integer floor square root on device.
 */
__device__ inline int64 d_isqrt(int64 n) {
    if (n <= 0) return 0;
    int64 s = static_cast<int64>(sqrt(static_cast<double>(n)));
    while ((s + 1) * (s + 1) <= n) ++s;
    while (s * s > n) --s;
    return s;
}

/**
 * Fast 64-bit integer warp-level parallel reduction.
 */
__device__ inline int64 warp_reduce_sum(int64 val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level parallel reduction for 64-bit integers using shared memory.
 */
__device__ inline int64 block_reduce_sum(int64 val) {
    __shared__ int64 shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

namespace functors {
    struct S2Comb1 { __device__ inline int64 operator()(int64 q) const { return (q / 4) + (q & 1) - (q / 12) - ((q / 3) & 1); } };
    struct S2Comb2 { __device__ inline int64 operator()(int64 q) const { return (q / 4) + (q & 1) - ((q % 6 >= 3) ? 1 : 0); } };
    struct S2Comb3 { __device__ inline int64 operator()(int64 q) const { return (q / 4) + (q & 1) - ((q + 3) / 6); } };
    struct S2Comb4 { __device__ inline int64 operator()(int64 q) const { return (q / 4) + (q & 1) - (q / 3); } };
    struct S2Comb5 { __device__ inline int64 operator()(int64 q) const { return (q / 4) + (q & 1); } };
    struct S2Comb6 { __device__ inline int64 operator()(int64 q) const { return (q & 1); } };
    struct S2Comb7 { __device__ inline int64 operator()(int64 q) const { return (q + 1) / 2; } };
    struct S2Comb8 { __device__ inline int64 operator()(int64 q) const { return q; } };

    struct S2Single1 { __device__ inline int64 operator()(int64 q) const { return (((q + 1) / 2) - (((q / 3) + 1) / 2)); } };
    struct S2Single2 { __device__ inline int64 operator()(int64 q) const { return (((q + 1) / 2) - (q / 3)); } };
    struct S2Single3 { __device__ inline int64 operator()(int64 q) const { return ((q + 1) / 2); } };
    struct S2Single4 { __device__ inline int64 operator()(int64 q) const { return q; } };
}

/**
 * Exact floor division helper: Uses fast double precision when y <= 2^53,
 * and falls back to exact 64-bit integer division when y > 2^53 (e.g. X >= 10^17).
 */
__device__ inline int64 fast_div(int64 y, double dy, int64 d) {
    if (y > 9007199254740992LL) {
        return y / d;
    } else {
        return static_cast<int64>(__ddiv_rn(dy, static_cast<double>(d)));
    }
}

// ---------------------------------------------------------------------------
// 1. Warp-Collaborative S2 Evaluators (32 threads in lockstep, zero divergence)
// ---------------------------------------------------------------------------
template <typename F>
__device__ inline int64 run_s2_fast_warp(int64 j_start, int64 j_end, int64 y, double dy, const int8_t* __restrict__ mu_ptr, F summand_fn, int lane_id) {
    if (j_start > j_end) return 0;
    int64 sum = 0;

    int64 m_start = (j_start + 5) / 6;
    int64 m_end = j_end / 6;

    // Head Boundary
    int64 head_end = (j_end < m_start * 6) ? j_end : (m_start * 6);
    for (int64 j = j_start + lane_id; j <= head_end; j += 32) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = __ldg(mu_ptr + j);
            if (m != 0) {
                int64 q = fast_div(y, dy, j);
                sum += static_cast<int64>(m) * summand_fn(q);
            }
        }
    }

    // Main Loop over (6m+1, 6m+5) strided across warp lanes
    for (int64 m = m_start + lane_id; m < m_end; m += 32) {
        int64 j1 = 6 * m + 1;
        int64 j2 = 6 * m + 5;
        int8_t m1 = __ldg(mu_ptr + j1);
        int8_t m2 = __ldg(mu_ptr + j2);

        if (m1) {
            int64 q1 = fast_div(y, dy, j1);
            sum += static_cast<int64>(m1) * summand_fn(q1);
        }
        if (m2) {
            int64 q2 = fast_div(y, dy, j2);
            sum += static_cast<int64>(m2) * summand_fn(q2);
        }
    }

    // Tail Boundary
    int64 tail_start = (j_start > m_end * 6 + 1) ? j_start : (m_end * 6 + 1);
    for (int64 j = tail_start + lane_id; j <= j_end; j += 32) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = __ldg(mu_ptr + j);
            if (m != 0) {
                int64 q = fast_div(y, dy, j);
                sum += static_cast<int64>(m) * summand_fn(q);
            }
        }
    }

    return sum;
}

__device__ inline int64 eval_s2_combined_warp(int64 y, int64 A, int64 B, const int8_t* __restrict__ mu_ptr, int lane_id) {
    double dy = static_cast<double>(y);
    int64 b6 = B / 6; int64 a6 = A / 6;
    int64 b3 = B / 3; int64 a3 = A / 3;
    int64 b2 = B / 2; int64 a2 = A / 2;

    int64 sum = 0;
    sum += run_s2_fast_warp(1, b6, y, dy, mu_ptr, functors::S2Comb1{}, lane_id);
    sum += run_s2_fast_warp(b6 + 1, a6, y, dy, mu_ptr, functors::S2Comb2{}, lane_id);
    sum += run_s2_fast_warp(a6 + 1, b3, y, dy, mu_ptr, functors::S2Comb3{}, lane_id);
    sum += run_s2_fast_warp(b3 + 1, a3, y, dy, mu_ptr, functors::S2Comb4{}, lane_id);
    sum += run_s2_fast_warp(a3 + 1, b2, y, dy, mu_ptr, functors::S2Comb5{}, lane_id);
    sum += run_s2_fast_warp(b2 + 1, a2, y, dy, mu_ptr, functors::S2Comb6{}, lane_id);
    sum += run_s2_fast_warp(a2 + 1, B, y, dy, mu_ptr, functors::S2Comb7{}, lane_id);
    sum += run_s2_fast_warp(B + 1, A, y, dy, mu_ptr, functors::S2Comb8{}, lane_id);
    return sum;
}

__device__ inline int64 eval_s2_single_warp(int64 y, int64 A, const int8_t* __restrict__ mu_ptr, int lane_id) {
    double dy = static_cast<double>(y);
    int64 a6 = A / 6;
    int64 a3 = A / 3;
    int64 a2 = A / 2;

    int64 sum = 0;
    sum += run_s2_fast_warp(1, a6, y, dy, mu_ptr, functors::S2Single1{}, lane_id);
    sum += run_s2_fast_warp(a6 + 1, a3, y, dy, mu_ptr, functors::S2Single2{}, lane_id);
    sum += run_s2_fast_warp(a3 + 1, a2, y, dy, mu_ptr, functors::S2Single3{}, lane_id);
    sum += run_s2_fast_warp(a2 + 1, A, y, dy, mu_ptr, functors::S2Single4{}, lane_id);
    return sum;
}

// ---------------------------------------------------------------------------
// 2. Block-Collaborative S2 Evaluators (512 threads per block for top heavy k)
// ---------------------------------------------------------------------------
template <typename F>
__device__ inline int64 run_s2_fast_block(int64 j_start, int64 j_end, int64 y, double dy, const int8_t* __restrict__ mu_ptr, F summand_fn) {
    if (j_start > j_end) return 0;
    int64 sum = 0;

    int64 m_start = (j_start + 5) / 6;
    int64 m_end = j_end / 6;

    // Head Boundary
    int64 head_end = (j_end < m_start * 6) ? j_end : (m_start * 6);
    for (int64 j = j_start + threadIdx.x; j <= head_end; j += blockDim.x) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = __ldg(mu_ptr + j);
            if (m != 0) {
                int64 q = fast_div(y, dy, j);
                sum += static_cast<int64>(m) * summand_fn(q);
            }
        }
    }

    for (int64 m = m_start + threadIdx.x; m < m_end; m += blockDim.x) {
        int64 j1 = 6 * m + 1;
        int64 j2 = 6 * m + 5;
        int8_t m1 = __ldg(mu_ptr + j1);
        int8_t m2 = __ldg(mu_ptr + j2);

        if (m1) {
            int64 q1 = fast_div(y, dy, j1);
            sum += static_cast<int64>(m1) * summand_fn(q1);
        }
        if (m2) {
            int64 q2 = fast_div(y, dy, j2);
            sum += static_cast<int64>(m2) * summand_fn(q2);
        }
    }

    int64 tail_start = (j_start > m_end * 6 + 1) ? j_start : (m_end * 6 + 1);
    for (int64 j = tail_start + threadIdx.x; j <= j_end; j += blockDim.x) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = __ldg(mu_ptr + j);
            if (m != 0) {
                int64 q = fast_div(y, dy, j);
                sum += static_cast<int64>(m) * summand_fn(q);
            }
        }
    }

    return sum;
}

__device__ inline int64 eval_s2_combined_block(int64 y, int64 A, int64 B, const int8_t* __restrict__ mu_ptr) {
    double dy = static_cast<double>(y);
    int64 b6 = B / 6; int64 a6 = A / 6;
    int64 b3 = B / 3; int64 a3 = A / 3;
    int64 b2 = B / 2; int64 a2 = A / 2;

    int64 sum = 0;
    sum += run_s2_fast_block(1, b6, y, dy, mu_ptr, functors::S2Comb1{});
    sum += run_s2_fast_block(b6 + 1, a6, y, dy, mu_ptr, functors::S2Comb2{});
    sum += run_s2_fast_block(a6 + 1, b3, y, dy, mu_ptr, functors::S2Comb3{});
    sum += run_s2_fast_block(b3 + 1, a3, y, dy, mu_ptr, functors::S2Comb4{});
    sum += run_s2_fast_block(a3 + 1, b2, y, dy, mu_ptr, functors::S2Comb5{});
    sum += run_s2_fast_block(b2 + 1, a2, y, dy, mu_ptr, functors::S2Comb6{});
    sum += run_s2_fast_block(a2 + 1, B, y, dy, mu_ptr, functors::S2Comb7{});
    sum += run_s2_fast_block(B + 1, A, y, dy, mu_ptr, functors::S2Comb8{});
    return sum;
}

__device__ inline int64 eval_s2_single_block(int64 y, int64 A, const int8_t* __restrict__ mu_ptr) {
    double dy = static_cast<double>(y);
    int64 a6 = A / 6;
    int64 a3 = A / 3;
    int64 a2 = A / 2;

    int64 sum = 0;
    sum += run_s2_fast_block(1, a6, y, dy, mu_ptr, functors::S2Single1{});
    sum += run_s2_fast_block(a6 + 1, a3, y, dy, mu_ptr, functors::S2Single2{});
    sum += run_s2_fast_block(a3 + 1, a2, y, dy, mu_ptr, functors::S2Single3{});
    sum += run_s2_fast_block(a2 + 1, A, y, dy, mu_ptr, functors::S2Single4{});
    return sum;
}

// ---------------------------------------------------------------------------
// 3. Heavy Kernels: 1 Dedicated Block per ultra-heavy k (512 threads)
// ---------------------------------------------------------------------------
__global__ void eval_dr_combined_heavy_kernel(
    int64 X,
    int64 u,
    double cx,
    const int64* __restrict__ d_odd_k,
    int64 num_heavy_k,
    const int16_t* __restrict__ d_M,
    const int8_t* __restrict__ d_mu,
    int64* __restrict__ d_total_sum
) {
    for (int64 k_idx = blockIdx.x; k_idx < num_heavy_k; k_idx += gridDim.x) {
        int64 k = d_odd_k[k_idx];
        int8_t mu_k = __ldg(d_mu + k);
        if (mu_k == 0) continue;

        int64 y = X / k;
        int64 y2 = y / 2;

        int64 A = static_cast<int64>(cx * sqrt(static_cast<double>(y)));
        int64 B = static_cast<int64>(cx * sqrt(static_cast<double>(y2)));
        if (A >= y) A = y - 1;
        if (B >= y2) B = y2 - 1;
        if (A < 1) A = 1;
        if (B < 1) B = 1;

        int64 kappa_y = y / (A + 1);
        int64 kappa_y2 = y2 / (B + 1);
        int64 two_kappa_y2 = 2 * kappa_y2;

        int64 my_s1 = 0;
        int64 start_odd = (y / u + 1);
        if (start_odd % 2 == 0) ++start_odd;

        double dy = static_cast<double>(y);
        int64 n_start = start_odd + 2 * threadIdx.x;
        int64 n_stride = 2 * blockDim.x;

        for (int64 n = n_start; n <= kappa_y; n += n_stride) {
            int64 q = fast_div(y, dy, n);
            my_s1 += __ldg(d_M + q);
        }

        if (two_kappa_y2 > kappa_y) {
            int64 start_even = kappa_y + 1;
            if (start_even % 2 != 0) ++start_even;
            int64 ne_start = start_even + 2 * threadIdx.x;
            for (int64 ne = ne_start; ne <= two_kappa_y2; ne += n_stride) {
                int64 q = fast_div(y, dy, ne);
                my_s1 -= __ldg(d_M + q);
            }
        } else if (two_kappa_y2 < kappa_y) {
            int64 start_even = two_kappa_y2 + 1;
            if (start_even % 2 != 0) ++start_even;
            int64 ne_start = start_even + 2 * threadIdx.x;
            for (int64 ne = ne_start; ne <= kappa_y; ne += n_stride) {
                int64 q = fast_div(y, dy, ne);
                my_s1 += __ldg(d_M + q);
            }
        }

        int64 my_s2 = eval_s2_combined_block(y, A, B, d_mu);

        int64 block_s1 = block_reduce_sum(my_s1);
        int64 block_s2 = block_reduce_sum(my_s2);

        if (threadIdx.x == 0) {
            int64 term = -block_s1 + (kappa_y * static_cast<int64>(__ldg(d_M + A)) - kappa_y2 * static_cast<int64>(__ldg(d_M + B))) - block_s2;
            atomicAdd(reinterpret_cast<unsigned long long*>(d_total_sum), static_cast<unsigned long long>(mu_k * term));
        }
        __syncthreads();
    }
}

__global__ void eval_dr_single_heavy_kernel(
    int64 X,
    int64 u,
    double cx,
    const int64* __restrict__ d_odd_k,
    int64 num_heavy_k,
    const int16_t* __restrict__ d_M,
    const int8_t* __restrict__ d_mu,
    int64* __restrict__ d_total_sum
) {
    for (int64 k_idx = blockIdx.x; k_idx < num_heavy_k; k_idx += gridDim.x) {
        int64 k = d_odd_k[k_idx];
        int8_t mu_k = __ldg(d_mu + k);
        if (mu_k == 0) continue;

        int64 y = X / k;
        int64 A = static_cast<int64>(cx * sqrt(static_cast<double>(y)));
        if (A >= y) A = y - 1;
        if (A < 1) A = 1;
        int64 kappa_y = y / (A + 1);

        int64 my_s1 = 0;
        int64 start_n = y / u + 1;
        double dy = static_cast<double>(y);

        for (int64 n = start_n + threadIdx.x; n <= kappa_y; n += blockDim.x) {
            int64 q = fast_div(y, dy, n);
            my_s1 += __ldg(d_M + q);
        }

        int64 my_s2 = eval_s2_single_block(y, A, d_mu);

        int64 block_s1 = block_reduce_sum(my_s1);
        int64 block_s2 = block_reduce_sum(my_s2);

        if (threadIdx.x == 0) {
            int64 S_val = 1 - block_s1 + kappa_y * static_cast<int64>(__ldg(d_M + A)) - block_s2;
            atomicAdd(reinterpret_cast<unsigned long long*>(d_total_sum), static_cast<unsigned long long>(mu_k * S_val));
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// 4. Warp-Collaborative Kernels: 1 Warp (32 threads) per k (Zero divergence!)
// ---------------------------------------------------------------------------
__global__ void eval_dr_combined_warp_kernel(
    int64 X,
    int64 u,
    double cx,
    const int64* __restrict__ d_odd_k,
    int64 num_k,
    const int16_t* __restrict__ d_M,
    const int8_t* __restrict__ d_mu,
    int64* __restrict__ d_total_sum
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int num_warps = gridDim.x * (blockDim.x / 32);

    for (int64 i = warp_id; i < num_k; i += num_warps) {
        int64 k = d_odd_k[i];
        int8_t mu_k = __ldg(d_mu + k);
        if (mu_k == 0) continue;

        int64 y = X / k;
        int64 y2 = y / 2;

        int64 A = static_cast<int64>(cx * sqrt(static_cast<double>(y)));
        int64 B = static_cast<int64>(cx * sqrt(static_cast<double>(y2)));
        if (A >= y) A = y - 1;
        if (B >= y2) B = y2 - 1;
        if (A < 1) A = 1;
        if (B < 1) B = 1;

        int64 kappa_y = y / (A + 1);
        int64 kappa_y2 = y2 / (B + 1);
        int64 two_kappa_y2 = 2 * kappa_y2;

        int64 my_s1 = 0;
        int64 start_odd = (y / u + 1);
        if (start_odd % 2 == 0) ++start_odd;

        double dy = static_cast<double>(y);
        int64 n_start = start_odd + 2 * lane_id;

        for (int64 n = n_start; n <= kappa_y; n += 64) {
            int64 q = fast_div(y, dy, n);
            my_s1 += __ldg(d_M + q);
        }

        if (two_kappa_y2 > kappa_y) {
            int64 start_even = kappa_y + 1;
            if (start_even % 2 != 0) ++start_even;
            int64 ne_start = start_even + 2 * lane_id;
            for (int64 ne = ne_start; ne <= two_kappa_y2; ne += 64) {
                int64 q = fast_div(y, dy, ne);
                my_s1 -= __ldg(d_M + q);
            }
        } else if (two_kappa_y2 < kappa_y) {
            int64 start_even = two_kappa_y2 + 1;
            if (start_even % 2 != 0) ++start_even;
            int64 ne_start = start_even + 2 * lane_id;
            for (int64 ne = ne_start; ne <= kappa_y; ne += 64) {
                int64 q = fast_div(y, dy, ne);
                my_s1 += __ldg(d_M + q);
            }
        }

        int64 my_s2 = eval_s2_combined_warp(y, A, B, d_mu, lane_id);

        int64 warp_s1 = warp_reduce_sum(my_s1);
        int64 warp_s2 = warp_reduce_sum(my_s2);

        if (lane_id == 0) {
            int64 term = -warp_s1 + (kappa_y * static_cast<int64>(__ldg(d_M + A)) - kappa_y2 * static_cast<int64>(__ldg(d_M + B))) - warp_s2;
            atomicAdd(reinterpret_cast<unsigned long long*>(d_total_sum), static_cast<unsigned long long>(mu_k * term));
        }
    }
}

__global__ void eval_dr_single_warp_kernel(
    int64 X,
    int64 u,
    double cx,
    const int64* __restrict__ d_odd_k,
    int64 num_k,
    const int16_t* __restrict__ d_M,
    const int8_t* __restrict__ d_mu,
    int64* __restrict__ d_total_sum
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int num_warps = gridDim.x * (blockDim.x / 32);

    for (int64 i = warp_id; i < num_k; i += num_warps) {
        int64 k = d_odd_k[i];
        int8_t mu_k = __ldg(d_mu + k);
        if (mu_k == 0) continue;

        int64 y = X / k;
        int64 A = static_cast<int64>(cx * sqrt(static_cast<double>(y)));
        if (A >= y) A = y - 1;
        if (A < 1) A = 1;
        int64 kappa_y = y / (A + 1);

        int64 my_s1 = 0;
        int64 start_n = y / u + 1;
        double dy = static_cast<double>(y);

        for (int64 n = start_n + lane_id; n <= kappa_y; n += 32) {
            int64 q = fast_div(y, dy, n);
            my_s1 += __ldg(d_M + q);
        }

        int64 my_s2 = eval_s2_single_warp(y, A, d_mu, lane_id);

        int64 warp_s1 = warp_reduce_sum(my_s1);
        int64 warp_s2 = warp_reduce_sum(my_s2);

        if (lane_id == 0) {
            int64 S_val = 1 - warp_s1 + kappa_y * static_cast<int64>(__ldg(d_M + A)) - warp_s2;
            atomicAdd(reinterpret_cast<unsigned long long*>(d_total_sum), static_cast<unsigned long long>(mu_k * S_val));
        }
    }
}

/**
 * Master CUDA Deléglise-Rivat Mertens Engine.
 */
class CudaDelégliseRivatEngine {
public:
    struct TimingStats {
        double host_sieve_time = 0.0;
        double h2d_copy_time = 0.0;
        double kernel_time = 0.0;
        double d2h_copy_time = 0.0;
        double total_time = 0.0;
    };

    /**
     * Dynamically chooses optimal sieve cutoff u for target X on GPU.
     */
    static inline int64 choose_gpu_sieve_limit(int64 X, int threads) {
        (void)threads;
        if (X <= 50000000LL) return X;

        double loglogX = std::log(std::max(2.0, std::log(static_cast<double>(X))));
        double fx = 1.10;
        if (X >= 1000000000000000LL) fx = 1.45;
        if (X >= 10000000000000000LL) fx = 2.20;

        int64 u = static_cast<int64>(fx * std::pow(static_cast<double>(X) / loglogX, 2.0 / 3.0));
        int64 S = mertens_dr::isqrt(X);
        if (u < 3 * S) u = 3 * S;

        // Cap u safely at 1.4 Billion (~2.8 GB table) to stay well within 12.7 GB Host System RAM
        return std::min(u, 1400000000LL);
    }

    /**
     * Computes M(X) = sum_{n=1}^X mu(n) using GPU acceleration.
     */
    static int64 compute_mertens(int64 X, int host_threads = 0, TimingStats* stats = nullptr) {
        if (X < 1) return 0;
        if (X == 1) return 1;
        if (X == 2) return 0;
        if (X == 3) return -1;

        auto t_start = std::chrono::high_resolution_clock::now();

        if (X <= 50000000LL) {
            return mertens_dr::DelégliseRivatEngine::compute_mertens(X, host_threads);
        }

        int64 u = choose_gpu_sieve_limit(X, host_threads);

        // 1. Precompute sieve table on host
        auto t0 = std::chrono::high_resolution_clock::now();
        mertens_dr::SieveTable table(u, host_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (stats) stats->host_sieve_time = std::chrono::duration<double>(t1 - t0).count();

        if (X <= u) {
            return table.get_M(X);
        }

        const double cx = 0.70;
        const int64 N = X / u;
        const int64 N_half = N / 2;
        const int8_t* mu_ptr = table.mu.data();

        // Collect odd square-free k
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

        // 2. Allocate and copy to GPU
        int64 mu_limit = std::min(u, std::max(N, static_cast<int64>(cx * std::sqrt(static_cast<double>(X)))) + 65536LL);

        int16_t* d_M = nullptr;
        int8_t* d_mu = nullptr;
        int64* d_odd_k_comb = nullptr;
        int64* d_odd_k_single = nullptr;
        int64* d_total_sum = nullptr;

        CUDA_CHECK(cudaMalloc(&d_M, (u + 1) * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&d_mu, (mu_limit + 1) * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_total_sum, sizeof(int64)));

        if (!odd_k_comb.empty()) {
            CUDA_CHECK(cudaMalloc(&d_odd_k_comb, odd_k_comb.size() * sizeof(int64)));
        }
        if (!odd_k_single.empty()) {
            CUDA_CHECK(cudaMalloc(&d_odd_k_single, odd_k_single.size() * sizeof(int64)));
        }

        int64 zero_val = 0;
        auto t2 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(d_total_sum, &zero_val, sizeof(int64), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_M, table.data(), (u + 1) * sizeof(int16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mu, table.mu.data(), (mu_limit + 1) * sizeof(int8_t), cudaMemcpyHostToDevice));

        if (!odd_k_comb.empty()) {
            CUDA_CHECK(cudaMemcpy(d_odd_k_comb, odd_k_comb.data(), odd_k_comb.size() * sizeof(int64), cudaMemcpyHostToDevice));
        }
        if (!odd_k_single.empty()) {
            CUDA_CHECK(cudaMemcpy(d_odd_k_single, odd_k_single.data(), odd_k_single.size() * sizeof(int64), cudaMemcpyHostToDevice));
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (stats) stats->h2d_copy_time = std::chrono::duration<double>(t3 - t2).count();

        // 3. Launch GPU Kernels
        cudaEvent_t k_start, k_stop;
        CUDA_CHECK(cudaEventCreate(&k_start));
        CUDA_CHECK(cudaEventCreate(&k_stop));
        CUDA_CHECK(cudaEventRecord(k_start));

        int num_sms = 40; // Tesla T4 has 40 SMs
        int dev_id = 0;
        if (cudaGetDevice(&dev_id) == cudaSuccess) {
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
        }

        const int block_size = 256;
        const int64 heavy_k_limit = 256; // Top heavy terms run on 512-thread dedicated blocks

        // Combined Range Execution
        if (!odd_k_comb.empty()) {
            int64 num_comb = static_cast<int64>(odd_k_comb.size());
            int64 num_heavy = 0;
            while (num_heavy < num_comb && odd_k_comb[num_heavy] <= heavy_k_limit) {
                ++num_heavy;
            }

            if (num_heavy > 0) {
                int heavy_grid = std::min(static_cast<int>(num_heavy), num_sms * 8);
                eval_dr_combined_heavy_kernel<<<heavy_grid, 512>>>(
                    X, u, cx, d_odd_k_comb, num_heavy, d_M, d_mu, d_total_sum
                );
                CUDA_CHECK(cudaGetLastError());
            }

            int64 num_light = num_comb - num_heavy;
            if (num_light > 0) {
                int warp_grid = num_sms * 16;
                eval_dr_combined_warp_kernel<<<warp_grid, block_size>>>(
                    X, u, cx, d_odd_k_comb + num_heavy, num_light, d_M, d_mu, d_total_sum
                );
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // Single Range Execution
        if (!odd_k_single.empty()) {
            int64 num_single = static_cast<int64>(odd_k_single.size());
            int64 num_heavy = 0;
            while (num_heavy < num_single && odd_k_single[num_heavy] <= heavy_k_limit) {
                ++num_heavy;
            }

            if (num_heavy > 0) {
                int heavy_grid = std::min(static_cast<int>(num_heavy), num_sms * 8);
                eval_dr_single_heavy_kernel<<<heavy_grid, 512>>>(
                    X, u, cx, d_odd_k_single, num_heavy, d_M, d_mu, d_total_sum
                );
                CUDA_CHECK(cudaGetLastError());
            }

            int64 num_light = num_single - num_heavy;
            if (num_light > 0) {
                int warp_grid = num_sms * 16;
                eval_dr_single_warp_kernel<<<warp_grid, block_size>>>(
                    X, u, cx, d_odd_k_single + num_heavy, num_light, d_M, d_mu, d_total_sum
                );
                CUDA_CHECK(cudaGetLastError());
            }
        }

        CUDA_CHECK(cudaEventRecord(k_stop));
        CUDA_CHECK(cudaEventSynchronize(k_stop));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k_start, k_stop));
        if (stats) stats->kernel_time = kernel_ms / 1000.0;

        CUDA_CHECK(cudaEventDestroy(k_start));
        CUDA_CHECK(cudaEventDestroy(k_stop));

        // 4. Retrieve result
        int64 total_gpu_M = 0;
        auto t4 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(&total_gpu_M, d_total_sum, sizeof(int64), cudaMemcpyDeviceToHost));
        auto t5 = std::chrono::high_resolution_clock::now();
        if (stats) stats->d2h_copy_time = std::chrono::duration<double>(t5 - t4).count();

        // 5. Cleanup
        CUDA_CHECK(cudaFree(d_M));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFree(d_total_sum));
        if (d_odd_k_comb) CUDA_CHECK(cudaFree(d_odd_k_comb));
        if (d_odd_k_single) CUDA_CHECK(cudaFree(d_odd_k_single));

        auto t_end = std::chrono::high_resolution_clock::now();
        if (stats) stats->total_time = std::chrono::duration<double>(t_end - t_start).count();

        return total_gpu_M;
    }
};

} // namespace mertens_cuda

#endif // MERTENS_CUDA_CUH
