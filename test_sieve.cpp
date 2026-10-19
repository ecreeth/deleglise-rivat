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

void check_square_roots() {
    constexpr int64 large = 3037000499LL;
    const int64 cases[][2] = {{-1, 0}, {0, 0}, {1, 1}, {2, 1}, {3, 1}, {4, 2},
        {15, 3}, {16, 4}, {17, 4}, {9007199254740992LL, 94906265LL},
        {10000000000000000LL, 100000000LL}, {large * large - 1, large - 1},
        {large * large, large}, {large * large + 1, large}, {9223372036854775807LL, large}};
    for (const auto& item : cases) check(isqrt(item[0]) == item[1], "Integer square root mismatch");
}

template <int Stride>
void check_byte_finalization() {
    for (int64 first : {1LL, 2LL, 3LL, 31LL, 32LL, 33LL, 127LL, 128LL, 129LL,
                        (1LL << 31) - 31, (1LL << 31) - 1, (1LL << 31) + 1,
                        (1LL << 33) - 31, (1LL << 33) + 1}) {
        for (int64 len : {0LL, 1LL, 15LL, 16LL, 17LL, 31LL, 32LL, 33LL, 63LL, 64LL, 65LL}) {
            for (unsigned seed = 0; seed < 8; ++seed) {
                std::vector<uint8_t> actual(static_cast<size_t>(len + 16), 0x5a);
                std::vector<uint8_t> expected(static_cast<size_t>(len));
                for (int64 i = 0; i < len; ++i) {
                    const uint8_t a = static_cast<uint8_t>(i * 37 + seed * 31);
                    actual[i] = a;
                    const unsigned threshold = std::bit_width(static_cast<uint64_t>(first + Stride * i)) - 1;
                    const int sign = (a & 1) ? -1 : 1;
                    expected[i] = static_cast<uint8_t>((a & 0x80) ? 0 : (a > threshold ? sign : -sign));
                }
                finalize_mobius_bytes<Stride>(actual.data(), len, first);
                for (int64 i = 0; i < len; ++i)
                    check(actual[i] == expected[i], "Byte finalization mismatch");
                for (int64 i = len; i < len + 16; ++i)
                    check(actual[i] == 0x5a, "Byte finalization wrote past its interval");
            }
        }
    }
}

void check_streamed_tables(const Reference& ref) {
    for (int threads : {1, 3}) {
        for (int total : {0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257,
                          13859, 13860, 13861, 17324, 17325, 17326,
                          131071, 131072, 131073, 262147, 1000033}) {
            for (int dense : {0, 1, 7, 63, 64, 65, 255, 256, 257, 1000, 131071, total}) {
                if (dense > total) continue;
                CompressedMertensTable table(total, dense, threads, total + 17);
                for (int n = 0; n <= total; ++n) {
                    check(table.get_M(n) == ref.prefix[n], "Streamed prefix mismatch");
                    check(table.get_mu(n) == ref.mu[n], "Streamed Mobius mismatch");
                }
                check(table.mu_limit == total, "Streamed Mobius limit was not clamped");
                check(table.get_mu(total + 1) == 0, "Streamed Mobius lookup passed its limit");
            }
        }
    }
}

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

int main() try {
    using namespace mertens_dr;
    check_square_roots();
    Reference ref(1048576);
    check_byte_finalization<1>();
    check_byte_finalization<2>();
    check_streamed_tables(ref);
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
    // Here floor(X/3985) equals the dense-prefix limit (25088).
    // It must use the dense table, rather than a negative compressed offset.
    check(DelégliseRivatEngine::compute_compressed(99975680LL, 3) == ref.mertens(99975680LL),
          "Compressed/dense quotient boundary mismatch");
    check(DelégliseRivatEngine::compute_compressed(0, 0) == 0,
          "Compressed engine M(0) mismatch");
    check(DelégliseRivatEngine::compute_compressed(100000000LL, 3) == 1928,
          "Compressed engine M(10^8) mismatch");
    check(DelégliseRivatEngine::compute_compressed(10000000000LL, 3) == -33722,
          "Compressed engine M(10^10) mismatch");
    check(DelégliseRivatEngine::compute_mertens(1000000000000LL, 4) == 62366,
          "M(10^12) mismatch");
    std::cout << "Sieve, streamed table, quotient, boundary, and recurrence checks passed.\n";
}

catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
