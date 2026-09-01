#include "mertens_cuda.cuh"
#include "mertens_dr.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cassert>

void print_usage(const char* prog) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << prog << " <X> [host_threads]                   Evaluate M(X) on CUDA GPU" << std::endl;
    std::cout << "  " << prog << " --benchmark [max_exp] [host_threads]  Run CUDA benchmark sweep up to 10^max_exp" << std::endl;
    std::cout << "  " << prog << " --compare [max_exp] [host_threads]    Compare CPU Deléglise-Rivat vs CUDA GPU" << std::endl;
}

int main(int argc, char* argv[]) {
    using namespace mertens_cuda;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable device found or CUDA driver error: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << " | Compute Capability: " 
              << prop.major << "." << prop.minor << " | Global Memory: " 
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;

    std::string arg1 = argv[1];

    if (arg1 == "--benchmark") {
        int max_exp = (argc >= 3) ? std::stoi(argv[2]) : 16;
        int host_threads = (argc >= 4) ? std::stoi(argv[3]) : 0;

        std::cout << "==========================================================================================================" << std::endl;
        std::cout << "          Deléglise-Rivat CUDA Mertens Benchmark Sweep (10^1..10^" << max_exp << ")                      " << std::endl;
        std::cout << "==========================================================================================================" << std::endl;
        std::cout << "| Power | Target X            | Exact M(X)         | Sieve (s) | H2D (s)   | GPU Kernel (s) | Total Time (s) |" << std::endl;
        std::cout << "| :---  | :---                | :---               | :---      | :---      | :---           | :---           |" << std::endl;

        int64 cur_X = 10;
        for (int exp = 1; exp <= max_exp; ++exp) {
            CudaDelégliseRivatEngine::TimingStats stats;
            int64 res = CudaDelégliseRivatEngine::compute_mertens(cur_X, host_threads, &stats);

            std::cout << "| 10^" << std::setw(2) << std::left << exp << " | "
                      << std::setw(19) << std::right << cur_X << " | "
                      << std::setw(18) << res << " | "
                      << std::setw(9) << std::fixed << std::setprecision(4) << stats.host_sieve_time << " | "
                      << std::setw(9) << std::fixed << std::setprecision(4) << stats.h2d_copy_time << " | "
                      << std::setw(14) << std::fixed << std::setprecision(5) << stats.kernel_time << " | "
                      << std::setw(14) << std::fixed << std::setprecision(5) << stats.total_time << " |" << std::endl;

            if (exp < 18) cur_X *= 10;
        }
        return 0;
    }

    if (arg1 == "--compare") {
        int max_exp = (argc >= 3) ? std::stoi(argv[2]) : 16;
        int host_threads = (argc >= 4) ? std::stoi(argv[3]) : 0;

        std::cout << "==========================================================================================================" << std::endl;
        std::cout << "  Comparative Performance Benchmark: CPU Deléglise-Rivat vs. CUDA GPU Deléglise-Rivat Engine              " << std::endl;
        std::cout << "==========================================================================================================" << std::endl;
        std::cout << "| Target X    | Exact M(X)        | CPU Time (s)      | GPU Kernel (s)    | GPU Total (s)     | Kernel Speedup |" << std::endl;
        std::cout << "| :---        | :---              | :---              | :---              | :---              | :---           |" << std::endl;

        int64 cur_X = 10000000000LL; // Start at 10^10
        for (int exp = 10; exp <= max_exp; ++exp) {
            // CPU time
            auto t0 = std::chrono::high_resolution_clock::now();
            int64 cpu_val = mertens_dr::DelégliseRivatEngine::compute_mertens(cur_X, host_threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpu_time = std::chrono::duration<double>(t1 - t0).count();

            // CUDA GPU time
            CudaDelégliseRivatEngine::TimingStats stats;
            int64 gpu_val = CudaDelégliseRivatEngine::compute_mertens(cur_X, host_threads, &stats);
            assert(cpu_val == gpu_val);

            double kernel_speedup = (stats.kernel_time > 0.0) ? (cpu_time / stats.kernel_time) : 1.0;

            std::cout << "| 10^" << std::setw(2) << std::left << exp << "      | "
                      << std::setw(17) << std::right << gpu_val << " | "
                      << std::setw(17) << std::fixed << std::setprecision(4) << cpu_time << " | "
                      << std::setw(17) << std::fixed << std::setprecision(4) << stats.kernel_time << " | "
                      << std::setw(17) << std::fixed << std::setprecision(4) << stats.total_time << " | "
                      << std::setw(14) << std::fixed << std::setprecision(2) << kernel_speedup << "x |" << std::endl;

            if (exp < 18) cur_X *= 10;
        }
        return 0;
    }

    // Single evaluation
    int64 X = std::stoll(arg1);
    int host_threads = (argc >= 3) ? std::stoi(argv[2]) : 0;

    std::cout << "Computing M(" << X << ") with Deléglise-Rivat CUDA engine..." << std::endl;
    CudaDelégliseRivatEngine::TimingStats stats;
    int64 result = CudaDelégliseRivatEngine::compute_mertens(X, host_threads, &stats);

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "M(" << X << ") = " << result << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Timing Breakdown:" << std::endl;
    std::cout << "  Host Sieve Precomputation: " << std::fixed << std::setprecision(6) << stats.host_sieve_time << " s" << std::endl;
    std::cout << "  Host-to-Device Memory Copy:" << std::fixed << std::setprecision(6) << stats.h2d_copy_time << " s" << std::endl;
    std::cout << "  GPU Kernel Execution:      " << std::fixed << std::setprecision(6) << stats.kernel_time << " s" << std::endl;
    std::cout << "  Device-to-Host Memory Copy:" << std::fixed << std::setprecision(6) << stats.d2h_copy_time << " s" << std::endl;
    std::cout << "  Total Execution Time:      " << std::fixed << std::setprecision(6) << stats.total_time << " s" << std::endl;

    return 0;
}
