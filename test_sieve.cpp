#include "mertens_dr.hpp"
#include <stdexcept>
#include <unordered_map>

namespace {
using namespace mertens_dr;
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Independent Eratosthenes Mobius sieve and quotient-grouped recurrence.
struct Reference {
    std::vector<int32_t> prefix;
    std::vector<int8_t> mu;
    std::unordered_map<int64, int64> cache;
    explicit Reference(int limit) : prefix(limit + 1), mu(limit + 1, 1) {
        mu[0] = 0;
        std::vector<bool> composite(limit + 1);
        for (int p = 2; p <= limit; ++p) {
            if (composite[p]) continue;
            for (int n = p; n <= limit; n += p) {
                composite[n] = true;
                mu[n] = static_cast<int8_t>(-mu[n]);
            }
            const int64 square = static_cast<int64>(p) * p;
            for (int64 n = square; n <= limit; n += square) mu[n] = 0;
        }
        for (int n = 1; n <= limit; ++n) prefix[n] = prefix[n - 1] + mu[n];
    }
    int64 mertens(int64 x) {
        if (x < static_cast<int64>(prefix.size())) return prefix[x];
        auto it = cache.find(x);
        if (it != cache.end()) return it->second;
        int64 sum = 1;
        for (int64 l = 2, r; l <= x; l = r + 1) {
            const int64 q = x / l;
            r = x / q;
            sum -= (r - l + 1) * mertens(q);
        }
        return cache[x] = sum;
    }
};

template <typename Piece>
void check_s2_intervals(const std::vector<int8_t>& mu) {
    for (int64 y : {37LL, 65535LL, 999999937LL, 9007199254740991LL,
                    9007199254740992LL, 9007199254740993LL,
                    10000000000000000LL, 100000000000000000LL}) {
        for (int64 first : {1LL, 2LL, 5LL, 6LL, 7LL, 11LL, 12LL, 13LL, 97LL, 99991LL}) {
            for (int64 count : {0LL, 1LL, 2LL, 5LL, 6LL, 7LL, 23LL, 24LL, 25LL, 57LL}) {
                const int64 last = first + count - 1;
                int64 expected = 0;
                for (int64 j = first; j <= last; ++j) {
                    if (j % 2 && j % 3)
                        expected += static_cast<int64>(mu[j]) * Piece::eval(y / j);
                }
                check(run_s2_fast<Piece>(first, last, y, static_cast<double>(y), mu.data()) == expected,
                      "S2 interval mismatch against integer division");
            }
        }
    }
}

#if defined(MERTENS_SSE2_DIV)
template <int Stride>
void check_dense_quotients() {
    std::vector<int16_t> values(4096);
    for (int i = 0; i < 4096; ++i) values[i] = static_cast<int16_t>(i % 511 - 255);
    for (int64 y : {1LL, 2LL, 999LL, 65536LL, 4294967296LL, 9007199254740991LL,
                    9007199254740992LL, 9007199254740993LL,
                    10000000000000000LL, 100000000000000000LL}) {
        for (int64 offset = 0; offset < 8; ++offset) {
            const int64 first = y / 4095 + 1 + offset;
            for (int64 count = 0; count <= 21; ++count) {
                const int64 last = first + count * Stride - 1;
                int64 expected = 0;
                for (int64 n = first; n <= last; n += Stride) expected += values[y / n];
                check(sum_dense_quotients<Stride>(y, first, last, values.data()) == expected,
                      "Dense quotient mismatch against integer division");
            }
        }
    }
}
#endif

}

int main() {
    using namespace mertens_dr;
    Reference ref(1048576);
    check_s2_intervals<Piece1>(ref.mu);
    check_s2_intervals<Piece2>(ref.mu);
    check_s2_intervals<Piece3>(ref.mu);
    check_s2_intervals<Piece4>(ref.mu);
    check_s2_intervals<Piece5>(ref.mu);
    check_s2_intervals<Piece6>(ref.mu);
    check_s2_intervals<Piece7>(ref.mu);
    check_s2_intervals<Piece8>(ref.mu);
    check_s2_intervals<SinglePiece1>(ref.mu);
    check_s2_intervals<SinglePiece2>(ref.mu);
    check_s2_intervals<SinglePiece3>(ref.mu);
    check_s2_intervals<SinglePiece4>(ref.mu);
#if defined(MERTENS_SSE2_DIV)
    check_dense_quotients<1>();
    check_dense_quotients<2>();
#endif
    for (int threads : {1, 3, 8}) {
        for (int limit : {0, 1, 2, 3, 4, 7, 31, 256, 257, 262145, 1000033}) {
            SieveTable table(limit, threads);
            for (int n = 0; n <= limit; ++n) {
                check(table.get_mu(n) == ref.mu[n], "Mobius table mismatch");
                check(table.get_M(n) == ref.prefix[n], "Dense prefix mismatch");
            }
        }
        SieveTable partial(262145, threads, 20000, true, 12345);
        for (int n = 0; n <= 12345; ++n)
            check(partial.get_M(n) == ref.prefix[n], "Partial prefix mismatch");
        for (int n = 1; n <= 262145; ++n)
            check(mu_from_odd(n, partial.mu_odd_data()) == ref.mu[n], "Retained odd table mismatch");
        CompressedMertensTable compressed(1048576, 1024, threads, 1048576);
        for (int n = 0; n <= 1048576; ++n) {
            check(compressed.get_M(n) == ref.prefix[n], "Compressed prefix mismatch");
            check(compressed.get_mu(n) == ref.mu[n], "Compressed Mobius mismatch");
        }
        for (int64 x : {100000LL, 100001LL, 262143LL, 262144LL, 262145LL,
                         999983LL, 1000000LL, 49999999LL, 50000000LL, 50000001LL,
                         50000003LL, 987654321LL, 10000000000LL})
            check(DelégliseRivatEngine::compute_mertens(x, threads) == ref.mertens(x),
                  "End-to-end recurrence mismatch");
    }
#ifdef _OPENMP
    // Nested parallelism is normally disabled, forcing the inner requested
    // eight-worker table build to run with one actual worker.
    omp_set_nested(0);
    int failures = 0;
    #pragma omp parallel num_threads(2) reduction(+:failures)
    {
        SieveTable table(262145, 8);
        CompressedMertensTable compressed(1048576, 1024, 8, 1024);
        for (int n = 0; n <= 262145; ++n) {
            if (table.get_M(n) != ref.prefix[n]) ++failures;
            if (compressed.get_M(n) != ref.prefix[n]) ++failures;
        }
    }
    check(failures == 0, "Reduced OpenMP team mismatch");
#endif
    check(DelégliseRivatEngine::compute_compressed(100000000LL, 3) == 1928,
          "Compressed engine M(10^8) mismatch");
    check(DelégliseRivatEngine::compute_compressed(10000000000LL, 3) == -33722,
          "Compressed engine M(10^10) mismatch");
    check(DelégliseRivatEngine::compute_mertens(1000000000000LL, 4) == 62366,
          "M(10^12) mismatch");
    std::cout << "Sieve, prefix, compressed, quotient, boundary, and recurrence checks passed.\n";
}
