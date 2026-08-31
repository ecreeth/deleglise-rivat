#include "mertens_dr.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cassert>

#if __has_include("../dirichlet_engine.hpp")
#include "../dirichlet_engine.hpp"
#define HAS_DIRICHLET_ENGINE 1
#endif

void print_usage(const char* prog) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << prog << " <X> [threads]                   Evaluate M(X) at a single value" << std::endl;
    std::cout << "  " << prog << " --benchmark [max_exp] [threads]  Run benchmark sweep up to 10^max_exp" << std::endl;
    std::cout << "  " << prog << " --compare [max_exp] [threads]    Compare DP vs Deléglise-Rivat" << std::endl;
}

int main(int argc, char* argv[]) {
    using namespace mertens_dr;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string arg1 = argv[1];

    if (arg1 == "--benchmark") {
        int max_exp = (argc >= 3) ? std::stoi(argv[2]) : 15;
        int threads = (argc >= 4) ? std::stoi(argv[3]) : 0;

        std::cout << "========================================================================" << std::endl;
        std::cout << "      Deléglise-Rivat O(X^(2/3)) Mertens Benchmark Sweep (10^1..10^" << max_exp << ")  " << std::endl;
        std::cout << "========================================================================" << std::endl;
        std::cout << "| Power | Target X            | Exact M(X)         | Time (seconds)    |" << std::endl;
        std::cout << "| :---  | :---                | :---               | :---              |" << std::endl;

        int64 cur_X = 10;
        for (int exp = 1; exp <= max_exp; ++exp) {
            auto t0 = std::chrono::high_resolution_clock::now();
            int64 res = DelégliseRivatEngine::compute_mertens(cur_X, threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();

            std::cout << "| 10^" << std::setw(2) << std::left << exp << " | "
                      << std::setw(19) << std::right << cur_X << " | "
                      << std::setw(18) << res << " | "
                      << std::setw(14) << std::fixed << std::setprecision(6) << elapsed << " s |" << std::endl;

            if (exp < 18) cur_X *= 10;
        }
        return 0;
    }

    if (arg1 == "--compare") {
        int max_exp = (argc >= 3) ? std::stoi(argv[2]) : 14;
        int threads = (argc >= 4) ? std::stoi(argv[3]) : 0;

        std::cout << "==========================================================================================================" << std::endl;
        std::cout << "  Comparative Performance Benchmark: O(X^(3/4)) Dirichlet DP vs. O(X^(2/3)) Deléglise-Rivat Engine        " << std::endl;
        std::cout << "==========================================================================================================" << std::endl;
        std::cout << "| Target X    | Exact M(X)        | Dirichlet DP Time | Deléglise-Rivat Time | Speedup (vs DP) | arXiv:2607.07566 |" << std::endl;
        std::cout << "| :---        | :---              | :---              | :---                 | :---            | :---             |" << std::endl;

        int64 cur_X = 10000000000LL; // Start at 10^10
        for (int exp = 10; exp <= max_exp; ++exp) {
            // DR time
            auto t0 = std::chrono::high_resolution_clock::now();
            int64 dr_val = DelégliseRivatEngine::compute_mertens(cur_X, threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dr_time = std::chrono::duration<double>(t1 - t0).count();

            double dp_time = 0.0;
#ifdef HAS_DIRICHLET_ENGINE
            auto t2 = std::chrono::high_resolution_clock::now();
            int64 dp_val = dirichlet::DirichletEngine::compute_mertens(cur_X, threads);
            auto t3 = std::chrono::high_resolution_clock::now();
            dp_time = std::chrono::duration<double>(t3 - t2).count();
            assert(dr_val == dp_val);
#endif

            double speedup = (dp_time > 0.0) ? (dp_time / dr_time) : 1.0;
            std::string hurst_ref = "-";
            if (exp == 13) hurst_ref = "0.12s (M2 Max)";
            if (exp == 14) hurst_ref = "0.51s (M2 Max)";
            if (exp == 15) hurst_ref = "2.10s (M2 Max)";
            if (exp == 16) hurst_ref = "2.39s (M3 Ultra)";

            std::cout << "| 10^" << std::setw(2) << std::left << exp << "      | "
                      << std::setw(17) << std::right << dr_val << " | "
                      << std::setw(14) << std::fixed << std::setprecision(4) << dp_time << " s | "
                      << std::setw(17) << std::fixed << std::setprecision(4) << dr_time << " s | "
                      << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x | "
                      << std::setw(16) << hurst_ref << " |" << std::endl;

            if (exp < 18) cur_X *= 10;
        }
        return 0;
    }

    // Single evaluation
    int64 X = std::stoll(arg1);
    int threads = (argc >= 3) ? std::stoi(argv[2]) : 0;

    std::cout << "Computing M(" << X << ") with Deléglise-Rivat O(X^(2/3)) engine..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    int64 result = DelégliseRivatEngine::compute_mertens(X, threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "M(" << X << ") = " << result << std::endl;
    std::cout << "Elapsed time: " << std::fixed << std::setprecision(6) << elapsed << " seconds" << std::endl;

    return 0;
}
