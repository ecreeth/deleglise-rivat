#include "mertens_cuda.cuh"
#include "mertens_dr.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <chrono>

struct TestCase {
    mertens_cuda::int64 X;
    mertens_cuda::int64 expected;
    const char* label;
};

int main() {
    using namespace mertens_cuda;

    std::cout << "============================================================" << std::endl;
    std::cout << "      Deléglise-Rivat CUDA Mertens Unit Test Suite          " << std::endl;
    std::cout << "============================================================" << std::endl;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable device found or CUDA driver error: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Target Device: " << prop.name 
              << " (" << prop.multiProcessorCount << " SMs, "
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB VRAM, CC " 
              << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::vector<TestCase> cases = {
        {1LL, 1LL, "1"},
        {2LL, 0LL, "2"},
        {3LL, -1LL, "3"},
        {4LL, -1LL, "4"},
        {5LL, -2LL, "5"},
        {10LL, -1LL, "10^1"},
        {100LL, 1LL, "10^2"},
        {1000LL, 2LL, "10^3"},
        {10000LL, -23LL, "10^4"},
        {100000LL, -48LL, "10^5"},
        {1000000LL, 212LL, "10^6"},
        {10000000LL, 1037LL, "10^7"},
        {100000000LL, 1928LL, "10^8"},
        {1000000000LL, -222LL, "10^9"},
        {10000000000LL, -33722LL, "10^10"},
        {100000000000LL, -87856LL, "10^11"},
        {1000000000000LL, 62366LL, "10^12"},
        {10000000000000LL, 599582LL, "10^13"},
        {100000000000000LL, -875575LL, "10^14"},
        {1000000000000000LL, -3216373LL, "10^15"}
    };

    int passed = 0;
    int failed = 0;

    for (const auto& tc : cases) {
        CudaDelégliseRivatEngine::TimingStats stats;
        int64 actual = CudaDelégliseRivatEngine::compute_mertens(tc.X, 0, &stats);

        bool ok = (actual == tc.expected);
        if (ok) {
            ++passed;
            std::cout << " [PASS] M(" << std::setw(17) << tc.X << ") [" << std::setw(6) << tc.label << "] = " 
                      << std::setw(15) << actual << "  (Total: " << std::fixed << std::setprecision(5) << stats.total_time 
                      << " s, GPU Kernel: " << std::setprecision(5) << stats.kernel_time << " s)" << std::endl;
        } else {
            ++failed;
            std::cerr << " [FAIL] M(" << tc.X << ") Expected: " << tc.expected << ", Got: " << actual << std::endl;
        }
    }

    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << " Cross-Validation Against CPU Deléglise-Rivat Engine:       " << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::vector<int64> cross_checks = {
        54321LL, 987654LL, 12345678LL, 987654321LL, 5000000000LL, 123456789012LL, 1000000000000LL
    };

    for (int64 x : cross_checks) {
        int64 cuda_val = CudaDelégliseRivatEngine::compute_mertens(x);
        int64 cpu_val = mertens_dr::DelégliseRivatEngine::compute_mertens(x);
        if (cuda_val == cpu_val) {
            std::cout << " [MATCH] M(" << x << ") = " << cuda_val << " (CUDA matches CPU bit-identically)" << std::endl;
            ++passed;
        } else {
            std::cerr << " [MISMATCH] M(" << x << ") CUDA: " << cuda_val << " vs CPU: " << cpu_val << std::endl;
            ++failed;
        }
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << " Test Summary: " << passed << " Passed, " << failed << " Failed." << std::endl;
    std::cout << "============================================================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
