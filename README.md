# Deléglise–Rivat $O(X^{2/3})$ Mertens Engine

A self-contained, header-only C++20 implementation of the **Deléglise–Rivat combinatorial method with inclusion–exclusion reductions and micro-architectural optimizations** for evaluating the Mertens function:
$$ M(X) = \sum_{n=1}^X \mu(n) $$

Based on and extending the mathematical framework described in Greg Hurst's preprint (*"Practical Computations of the Mertens Function: $M(10^{24})$ and $M(10^{25})$"*, [arXiv:2607.07566](https://arxiv.org/abs/2607.07566)).

---

## CPU Strategies and Optimizations

- **Small inputs:** a linear odd sieve handles `X <= 100000`. Through 50 million,
  the segmented odd sieve returns the required suffix sum without allocating a
  full Mertens prefix table.
- **Dense Deléglise-Rivat:** medium inputs use the mod-6 outer reduction,
  piecewise S2 sums, parity cancellation in S1, and direct 16-bit Mertens lookups.
- **Streaming compressed table:** x86-64 selects this strategy from `10^11`.
  It raises the sieve cutoff to as much as **8 billion**, reducing the number
  of outer terms. The dense prefix targets the smaller of 128 million entries
  and one quarter of the cutoff, expanding to include every A/B endpoint. Each compressed 256-value chunk has
  two 32-byte Mobius bit planes and four 32-bit checkpoints: **0.3125 bytes per
  compressed entry**, with two popcounts per lookup. Construction uses 128 KiB
  buffers per worker, without retaining a full temporary odd sieve.
- **Vectorized sieving and quotient loops:** x86-64 finalizes 16 byte-log sieve
  entries at once. Phase-correct stencils pre-mark the densest prime weights
  and square factors. SSE2 quotient loops retain exact integer division above
  `2^53`; other architectures retain scalar/optional NEON quotient backends.
- **Parallel prefixes:** logical blocks keep prefix offsets correct when
  OpenMP supplies fewer workers than requested, including nested teams.

Strategy and tuning controls:

| Setting | Behavior |
|---|---|
| `MERTENS_USE_CHMT` unset | Automatic streaming strategy from `10^11` on x86-64; dense elsewhere. |
| `MERTENS_USE_CHMT=0` | Force the dense strategy. |
| `MERTENS_USE_CHMT=1` | Select streaming compression for inputs from `10^11`, including other architectures. |
| `MERTENS_CHMT_U_CAP` | Set the compressed cutoff target (100 million to 10 billion); default is 8 billion. For example, `4000000000` reduces memory. The cutoff must still cover the algorithm's endpoints. |
| `MERTENS_FX`, `MERTENS_U_CAP` | Preserve explicit dense tuning and disable automatic compression, unless CHMT is explicitly enabled. |
| `MERTENS_CX` | Override the S1/S2 split parameter for either strategy. |

Define `MERTENS_USE_SSE2_DIV=0` to compare scalar quotient loops, or
`MERTENS_USE_SIMD_SIEVE=0` to compare scalar sieve finalization and packing.
These compiler controls leave the numerical result unchanged.

---

## Build & Test

### Prerequisites
- Clang / GCC supporting C++20
- OpenMP (`libomp` via Homebrew on macOS: `brew install libomp`)
- Native Windows builds can use MSVC 2022 with C++20 and OpenMP.

### Compilation
```bash
make
```

From an x64 Visual Studio Developer Command Prompt on Windows:

```bat
cl /O2 /std:c++20 /utf-8 /openmp /EHsc bench_dr.cpp /Fe:bench_dr.exe
cl /O2 /std:c++20 /utf-8 /openmp /EHsc test_dr.cpp /Fe:test_dr.exe
cl /O2 /std:c++20 /utf-8 /openmp /EHsc test_sieve.cpp /Fe:test_sieve.exe
```

### Running Unit Tests
```bash
make test
# Or:
./test_dr
```

### Running Benchmarks
```bash
# Evaluate a single value:
./bench_dr 1000000000000000 8

# Benchmark sweep from 10^1 to 10^15:
./bench_dr --benchmark 15 8

# Median of three runs per input (reduces timing noise):
./bench_dr --benchmark 15 8 3

# Comparative benchmark vs O(X^(3/4)) Dirichlet DP Engine:
./bench_dr --compare 14 8
```

Speedup varies with input size, table footprint, and the selected strategy.
Compare repeated runs with the same compiler, thread count, and tuning settings.
The [streaming-table comparison](benchmarks/streaming-2026-10-19.md) measures
this update against commit `edeb00d`. The [earlier SSE2 comparison](benchmarks/cpu-2026-10-19.md)
records the preceding optimization.

---

## CUDA GPU Acceleration (NVIDIA T4 16GB / `sm_75`)

The engine provides CUDA backend support with GPU-accelerated $S_1$ and $S_2$ evaluations:
- **`__ldg` Read-Only Caching:** Directly streams lookups of $M(n)$ from VRAM (~900MB).
- **`__constant__` Memory LUT Evaluators:** Fast broadcast caching for $S_2$ piecewise summands with zero register pressure.
- **Branchless Piecewise $S_2$ Modulo-6:** Evaluates arithmetic ranges without branch divergence.
- **Warp-Level Parallel Reduction:** Uses `__shfl_down_sync` and shared memory tree reductions for high-occupancy 64-bit atomic accumulation.

### CUDA Compilation
```bash
# Build CUDA binaries targeting NVIDIA T4 (sm_75):
make cuda

# Or specify a custom architecture (e.g. sm_80 for A100, sm_89 for RTX 4090, sm_90 for H100):
make cuda CUDA_ARCH=sm_80
```

### Running CUDA Tests & Benchmarks
```bash
# Run CUDA unit test suite:
make test-cuda

# Evaluate a single value on GPU with detailed timing breakdown:
./bench_cuda 1000000000000000

# Run CUDA benchmark sweep:
./bench_cuda --benchmark 16

# Compare CPU vs CUDA GPU performance:
./bench_cuda --compare 16
```

---

## Benchmark Results

### CPU Performance & Comparison vs. Greg Hurst (arXiv:2607.07566)

Historical measurements from 2026-09-20 on Apple M1 Mac mini (8T, 16GB), before the streaming-table update, used the engine
(scalar double-division quotient loops by default with opt-in NEON via
`MERTENS_USE_NEON_DIV`, small-prime `{3,5,7,11}` sieve stencil, tunable
OpenMP chunk `MERTENS_DYN_CHUNK=64`). `MERTENS_FX`/`MERTENS_CX` env overrides
select the sieve balance `fx` and split `cx`; defaults are `fx=0.32`
(cap 1.5G) and `cx=1.35` ($\ge 10^{15}$) / `1.15` ($\ge 10^{13}$) /
`0.95` ($\ge 10^{11}$) / `0.70` (below). Best 2026-09-20 tuning on this box
for $10^{15}$ is `MERTENS_FX=0.2 MERTENS_CX=1.15`. Expect $\pm 10$–$15\%$
run-to-run variance (frequency ramp / heat soak on Apple Silicon); figures
below are single-sweep runs with `./bench_dr --benchmark 15 8`.

| Target $X$ | Exact $M(X)$ | Baseline (Apple M1, 8T) | Current Engine, default config (Apple M1, 8T) | Tuned `FX=0.2 CX=1.15` (Apple M1, 8T) | Greg Hurst Preprint (Reported) | Speedup (best M1 vs Baseline) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^8$** | $1,928$ | $0.0021$ s | **$0.0004$ s** | **$0.0007$ s** | — | **$3.0\times$** |
| **$10^9$** | $-222$ | $0.0062$ s | **$0.0008$ s** | **$0.0011$ s** | — | **$5.6\times$** |
| **$10^{10}$** | $-33,722$ | $0.0151$ s | **$0.0040$ s** | **$0.0044$ s** | — | **$3.4\times$** |
| **$10^{11}$** | $-87,856$ | $0.0548$ s | **$0.0125$ s** | **$0.0111$ s** | — | **$4.9\times$** |
| **$10^{12}$** | $62,366$ | $0.1030$ s | **$0.0487$ s** | **$0.0572$ s** | $\sim 0.03$ s (M2 Max) | **$1.8\times$** |
| **$10^{13}$** | $599,582$ | $0.4500$ s | **$0.2212$ s** | **$0.2526$ s** | $0.12$ s (M2 Max) | **$1.8\times$** |
| **$10^{14}$** | $-875,575$ | $2.3370$ s | **$1.1768$ s** | **$1.2620$ s** | $0.51$ s (M2 Max) | **$1.9\times$** |
| **$10^{15}$** | $-3,216,373$ | $14.8000$ s | **$7.7940$ s** | **$7.0379$ s** (best observed $5.88$ s cold) | $2.10$ s (M2 Max) | **$2.1\times$** |
| **$10^{16}$** | $-3,195,437$ | $69.6000$ s | **$50.4758$ s** | — (not re-run tuned) | $2.39$ s (M3 Ultra) | **$1.4\times$** |

Small inputs (same 2026-09-20 sweep, default vs tuned — tuning is a no-op
below $u = X$, shown for completeness):

| Target $X$ | Exact $M(X)$ | Default config | Tuned `FX=0.2 CX=1.15` |
| :--- | :--- | :--- | :--- |
| **$10^1$** | $-1$ | $0.0006$ s | $0.0006$ s |
| **$10^2$** | $1$ | $<0.0001$ s | $<0.0001$ s |
| **$10^3$** | $2$ | $<0.0001$ s | $<0.0001$ s |
| **$10^4$** | $-23$ | $<0.0001$ s | $<0.0001$ s |
| **$10^5$** | $-48$ | $0.0003$ s | $0.0003$ s |
| **$10^6$** | $212$ | $0.0025$ s | $0.0039$ s |
| **$10^7$** | $1,037$ | $0.0156$ s | $0.0174$ s |

### NVIDIA Tesla T4 (16GB VRAM, Turing `sm_75` - CUDA Benchmark Sweep)

Retained from the earlier T4 run — not re-measured here (no CUDA-capable
device on the Apple M1 host used above).

| Target $X$ | Exact $M(X)$ | Sieve Time (s) | H2D Copy (s) | GPU Kernel (s) | Total Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^8$** | $1,928$ | $0.0016$ | $0.0002$ | **$0.00607$** | $0.1778$ |
| **$10^9$** | $-222$ | $0.0081$ | $0.0004$ | **$0.00538$** | $0.0141$ |
| **$10^{10}$** | $-33,722$ | $0.0306$ | $0.0011$ | **$0.01121$** | $0.0435$ |
| **$10^{11}$** | $-87,856$ | $0.1409$ | $0.0046$ | **$0.02856$** | $0.1756$ |
| **$10^{12}$** | $62,366$ | $0.6833$ | $0.0215$ | **$0.11964$** | $0.8267$ |
| **$10^{13}$** | $599,582$ | $4.1615$ | $0.1126$ | **$0.28687$** | $4.5677$ |
| **$10^{14}$** | $-875,575$ | $8.9891$ | $0.2537$ | **$0.73207$** | $9.9804$ |
| **$10^{15}$** | $-3,216,373$ | $9.8777$ | $0.2634$ | **$4.08156$** | $14.2351$ |
| **$10^{16}$** | $-3,195,437$ | $9.2805$ | $0.2868$ | **$35.62498$** | $45.2675$ |
| **$10^{17}$** | $-21,830,254$ | $10.0107$ | $0.4033$ | **$362.09725$** | $373.3884$ |
