#include "mertens_dr.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <chrono>

#if __has_include("../dirichlet_engine.hpp")
#include "../dirichlet_engine.hpp"
#define HAS_DIRICHLET_ENGINE 1
#endif

struct TestCase {
    mertens_dr::int64 X;
    mertens_dr::int64 expected;
    const char* label;
};

int main() {
    using namespace mertens_dr;

    std::cout << "============================================================" << std::endl;
    std::cout << "      Deléglise-Rivat O(X^(2/3)) Mertens Unit Test Suite     " << std::endl;
    std::cout << "============================================================" << std::endl;

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
        auto t0 = std::chrono::high_resolution_clock::now();
        int64 actual = DelégliseRivatEngine::compute_mertens(tc.X);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        bool ok = (actual == tc.expected);
        if (ok) {
            ++passed;
            std::cout << " [PASS] M(" << std::setw(17) << tc.X << ") [" << std::setw(6) << tc.label << "] = " 
                      << std::setw(15) << actual << "  (time: " << std::fixed << std::setprecision(5) << elapsed << " s)" << std::endl;
        } else {
            ++failed;
            std::cerr << " [FAIL] M(" << tc.X << ") Expected: " << tc.expected << ", Got: " << actual << std::endl;
        }
    }

#ifdef HAS_DIRICHLET_ENGINE
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << " Cross-Validation Against Dirichlet Hyperbola DP Engine:" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::vector<int64> cross_checks = {
        54321LL, 987654LL, 12345678LL, 987654321LL, 5000000000LL, 123456789012LL, 1000000000000LL
    };

    for (int64 x : cross_checks) {
        int64 dr_val = DelégliseRivatEngine::compute_mertens(x);
        int64 dp_val = dirichlet::DirichletEngine::compute_mertens(x);
        if (dr_val == dp_val) {
            std::cout << " [MATCH] M(" << x << ") = " << dr_val << " (DR matches DP bit-identically)" << std::endl;
            ++passed;
        } else {
            std::cerr << " [MISMATCH] M(" << x << ") DR: " << dr_val << " vs DP: " << dp_val << std::endl;
            ++failed;
        }
    }
#endif

    std::cout << "\n============================================================" << std::endl;
    std::cout << " Test Summary: " << passed << " Passed, " << failed << " Failed." << std::endl;
    std::cout << "============================================================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
