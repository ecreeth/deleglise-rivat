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
 * Warp-level parallel reduction for 64-bit integers.
 */
__device__ inline int64 warp_reduce_sum(int64 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level parallel reduction for 64-bit integers using shared memory.
 */
__device__ inline int64 block_reduce_sum(int64 val) {
    __shared__ int64 shared[32]; // Max 32 warps per block (1024 threads)
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

/**
 * Fast S2 Modulo-6 stream evaluator on GPU device.
 */
template <typename F>
__device__ inline int64 run_s2_fast_device(int64 j_start, int64 j_end, double dy, const int8_t* __restrict__ mu_ptr, F summand_fn) {
    if (j_start > j_end) return 0;
    int64 sum = 0;

    int64 m_start = (j_start + 5) / 6;
    int64 m_end = j_end / 6;

    // Head Boundary
    int64 head_end = (j_end < m_start * 6) ? j_end : (m_start * 6);
    for (int64 j = j_start; j <= head_end; ++j) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = __ldg(mu_ptr + j);
            if (m != 0) {
                int64 q = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(j)));
                sum += static_cast<int64>(m) * summand_fn(q);
            }
        }
    }

    // Main Loop over (6m+1, 6m+5)
    for (int64 m = m_start; m < m_end; ++m) {
        int64 j1 = 6 * m + 1;
        int64 j2 = 6 * m + 5;
        int8_t m1 = __ldg(mu_ptr + j1);
        int8_t m2 = __ldg(mu_ptr + j2);

        if (m1) {
            int64 q1 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(j1)));
            sum += static_cast<int64>(m1) * summand_fn(q1);
        }
        if (m2) {
            int64 q2 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(j2)));
            sum += static_cast<int64>(m2) * summand_fn(q2);
        }
    }

    // Tail Boundary
    int64 tail_start = (j_start > m_end * 6 + 1) ? j_start : (m_end * 6 + 1);
    for (int64 j = tail_start; j <= j_end; ++j) {
        if (j % 2 != 0 && j % 3 != 0) {
            int8_t m = __ldg(mu_ptr + j);
            if (m != 0) {
                int64 q = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(j)));
                sum += static_cast<int64>(m) * summand_fn(q);
            }
        }
    }

    return sum;
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
 * Device S2 Combined Range Evaluator: S2(y) - S2(y/2).
 */
__device__ inline int64 eval_s2_combined_device(int64 y, int64 A, int64 B, const int8_t* __restrict__ mu_ptr) {
    double dy = static_cast<double>(y);
    int64 b6 = B / 6; int64 a6 = A / 6;
    int64 b3 = B / 3; int64 a3 = A / 3;
    int64 b2 = B / 2; int64 a2 = A / 2;

    int64 sum = 0;

    // Piece 1: 1 <= j <= b6
    sum += run_s2_fast_device(1, b6, dy, mu_ptr, functors::S2Comb1{});

    // Piece 2: b6 < j <= a6
    sum += run_s2_fast_device(b6 + 1, a6, dy, mu_ptr, functors::S2Comb2{});

    // Piece 3: a6 < j <= b3
    sum += run_s2_fast_device(a6 + 1, b3, dy, mu_ptr, functors::S2Comb3{});

    // Piece 4: b3 < j <= a3
    sum += run_s2_fast_device(b3 + 1, a3, dy, mu_ptr, functors::S2Comb4{});

    // Piece 5: a3 < j <= b2
    sum += run_s2_fast_device(a3 + 1, b2, dy, mu_ptr, functors::S2Comb5{});

    // Piece 6: b2 < j <= a2
    sum += run_s2_fast_device(b2 + 1, a2, dy, mu_ptr, functors::S2Comb6{});

    // Piece 7: a2 < j <= B
    sum += run_s2_fast_device(a2 + 1, B, dy, mu_ptr, functors::S2Comb7{});

    // Piece 8: B < j <= A
    sum += run_s2_fast_device(B + 1, A, dy, mu_ptr, functors::S2Comb8{});

    return sum;
}

/**
 * Device S2 Single Range Evaluator: S2(y).
 */
__device__ inline int64 eval_s2_single_device(int64 y, int64 A, const int8_t* __restrict__ mu_ptr) {
    double dy = static_cast<double>(y);
    int64 a6 = A / 6;
    int64 a3 = A / 3;
    int64 a2 = A / 2;

    int64 sum = 0;

    sum += run_s2_fast_device(1, a6, dy, mu_ptr, functors::S2Single1{});
    sum += run_s2_fast_device(a6 + 1, a3, dy, mu_ptr, functors::S2Single2{});
    sum += run_s2_fast_device(a3 + 1, a2, dy, mu_ptr, functors::S2Single3{});
    sum += run_s2_fast_device(a2 + 1, A, dy, mu_ptr, functors::S2Single4{});

    return sum;
}

/**
 * CUDA Kernel: Evaluates Combined Range (k <= N/2)
 */
__global__ void eval_dr_combined_kernel(
    int64 X,
    int64 u,
    double cx,
    const int64* __restrict__ d_odd_k,
    int64 num_k,
    const int16_t* __restrict__ d_M,
    const int8_t* __restrict__ d_mu,
    int64* __restrict__ d_total_sum
) {
    int64 thread_sum = 0;
    int64 global_idx = static_cast<int64>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64 stride = static_cast<int64>(gridDim.x) * blockDim.x;

    for (int64 i = global_idx; i < num_k; i += stride) {
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

        // 1. S1(y, u) - S1(y/2, u)
        int64 S1_diff = 0;
        int64 start_odd = (y / u + 1);
        if (start_odd % 2 == 0) ++start_odd;

        double dy = static_cast<double>(y);

        // Stride-2 unrolled loop
        int64 n = start_odd;
        for (; n + 6 <= kappa_y; n += 8) {
            int64 q0 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n)));
            int64 q1 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 2)));
            int64 q2 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 4)));
            int64 q3 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 6)));
            S1_diff += __ldg(d_M + q0) + __ldg(d_M + q1) + __ldg(d_M + q2) + __ldg(d_M + q3);
        }
        for (; n <= kappa_y; n += 2) {
            int64 q = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n)));
            S1_diff += __ldg(d_M + q);
        }

        if (two_kappa_y2 > kappa_y) {
            int64 start_even = kappa_y + 1;
            if (start_even % 2 != 0) ++start_even;
            for (int64 ne = start_even; ne <= two_kappa_y2; ne += 2) {
                int64 q = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(ne)));
                S1_diff -= __ldg(d_M + q);
            }
        } else if (two_kappa_y2 < kappa_y) {
            int64 start_even = two_kappa_y2 + 1;
            if (start_even % 2 != 0) ++start_even;
            for (int64 ne = start_even; ne <= kappa_y; ne += 2) {
                int64 q = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(ne)));
                S1_diff += __ldg(d_M + q);
            }
        }

        // 2. S2(y) - S2(y/2) mod 6 piecewise evaluation
        int64 S2_diff = eval_s2_combined_device(y, A, B, d_mu);

        // S(y, u) - S(y/2, u)
        int64 term = -S1_diff + (kappa_y * static_cast<int64>(__ldg(d_M + A)) - kappa_y2 * static_cast<int64>(__ldg(d_M + B))) - S2_diff;
        thread_sum += static_cast<int64>(mu_k) * term;
    }

    int64 block_sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0 && block_sum != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(d_total_sum), static_cast<unsigned long long>(block_sum));
    }
}

/**
 * CUDA Kernel: Evaluates Single Range (N/2 < k <= N)
 */
__global__ void eval_dr_single_kernel(
    int64 X,
    int64 u,
    double cx,
    const int64* __restrict__ d_odd_k,
    int64 num_k,
    const int16_t* __restrict__ d_M,
    const int8_t* __restrict__ d_mu,
    int64* __restrict__ d_total_sum
) {
    int64 thread_sum = 0;
    int64 global_idx = static_cast<int64>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64 stride = static_cast<int64>(gridDim.x) * blockDim.x;

    for (int64 i = global_idx; i < num_k; i += stride) {
        int64 k = d_odd_k[i];
        int8_t mu_k = __ldg(d_mu + k);
        if (mu_k == 0) continue;

        int64 y = X / k;
        int64 A = static_cast<int64>(cx * sqrt(static_cast<double>(y)));
        if (A >= y) A = y - 1;
        if (A < 1) A = 1;
        int64 kappa_y = y / (A + 1);

        int64 S1 = 0;
        int64 start_n = y / u + 1;
        double dy = static_cast<double>(y);
        int64 n = start_n;

        for (; n + 7 <= kappa_y; n += 8) {
            int64 q0 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n)));
            int64 q1 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 1)));
            int64 q2 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 2)));
            int64 q3 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 3)));
            int64 q4 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 4)));
            int64 q5 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 5)));
            int64 q6 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 6)));
            int64 q7 = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n + 7)));

            S1 += __ldg(d_M + q0) + __ldg(d_M + q1) + __ldg(d_M + q2) + __ldg(d_M + q3) +
                  __ldg(d_M + q4) + __ldg(d_M + q5) + __ldg(d_M + q6) + __ldg(d_M + q7);
        }
        for (; n <= kappa_y; ++n) {
            int64 q = static_cast<int64>(__ddiv_rn(dy, static_cast<double>(n)));
            S1 += __ldg(d_M + q);
        }

        int64 S2 = eval_s2_single_device(y, A, d_mu);

        int64 S_val = 1 - S1 + kappa_y * static_cast<int64>(__ldg(d_M + A)) - S2;
        thread_sum += static_cast<int64>(mu_k) * S_val;
    }

    int64 block_sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0 && block_sum != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(d_total_sum), static_cast<unsigned long long>(block_sum));
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
     * Computes M(X) = sum_{n=1}^X mu(n) using GPU acceleration.
     */
    static int64 compute_mertens(int64 X, int host_threads = 0, TimingStats* stats = nullptr) {
        if (X < 1) return 0;
        if (X == 1) return 1;
        if (X == 2) return 0;
        if (X == 3) return -1;

        auto t_start = std::chrono::high_resolution_clock::now();

        // For very small X, CPU direct linear sieve is faster
        if (X <= 50000000LL) {
            return mertens_dr::DelégliseRivatEngine::compute_mertens(X, host_threads);
        }

        int64 u = mertens_dr::DelégliseRivatEngine::choose_sieve_limit(X, host_threads);

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
        int16_t* d_M = nullptr;
        int8_t* d_mu = nullptr;
        int64* d_odd_k_comb = nullptr;
        int64* d_odd_k_single = nullptr;
        int64* d_total_sum = nullptr;

        CUDA_CHECK(cudaMalloc(&d_M, (u + 1) * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&d_mu, (u + 1) * sizeof(int8_t)));
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
        CUDA_CHECK(cudaMemcpy(d_mu, table.mu.data(), (u + 1) * sizeof(int8_t), cudaMemcpyHostToDevice));

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

        const int block_size = 256;
        int num_sms = 40; // Default estimate (e.g. Tesla T4 has 40 SMs)
        int dev_id = 0;
        if (cudaGetDevice(&dev_id) == cudaSuccess) {
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
        }
        int num_blocks = num_sms * 16; // 640 blocks for high occupancy

        if (!odd_k_comb.empty()) {
            eval_dr_combined_kernel<<<num_blocks, block_size>>>(
                X, u, cx, d_odd_k_comb, static_cast<int64>(odd_k_comb.size()), d_M, d_mu, d_total_sum
            );
            CUDA_CHECK(cudaGetLastError());
        }

        if (!odd_k_single.empty()) {
            eval_dr_single_kernel<<<num_blocks, block_size>>>(
                X, u, cx, d_odd_k_single, static_cast<int64>(odd_k_single.size()), d_M, d_mu, d_total_sum
            );
            CUDA_CHECK(cudaGetLastError());
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
