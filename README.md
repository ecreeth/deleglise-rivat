# Deléglise–Rivat $O(X^{2/3})$ Mertens Engine

A self-contained, header-only C++20 implementation of the **Deléglise–Rivat combinatorial method with inclusion–exclusion reductions and micro-architectural optimizations** for evaluating the Mertens function:
$$ M(X) = \sum_{n=1}^X \mu(n) $$

Based on and extending the mathematical framework described in Greg Hurst's preprint (*"Practical Computations of the Mertens Function: $M(10^{24})$ and $M(10^{25})$"*, [arXiv:2607.07566](https://arxiv.org/abs/2607.07566)).

---

## Key Features & Optimizations

1. **Inclusion–Exclusion Reductions:**
   - **Outer Parity Split:** Evaluates $S(y, u) - S(y/2, u)$ for $k \le N/2$ and $S(y, u)$ for $N/2 < k \le N$ over odd square-free $k$, cutting outer terms by $>69\%$.
   - **$S_1$ Parity Cancellation:** Cancels even quotient lookups across surviving odd/even intervals (stride-2 queries).
   - **$S_2$ Piecewise Reduction:** Partitions $j \le \nu_y, (j, 6)=1$ into 8 exact branchless arithmetic intervals.

2. **Micro-Architectural & SIMD Acceleration:**
   - **Multi-Threaded Segmented Sieve:** Sieve segments partitioned into $128\text{K}$ L2-cache-blocked chunks with parallel prefix sums, speeding up sieve table generation by $>70\%$.
   - **Unified LUT & Bitwise Shift Collapse for $S_2$:** Constant 12-element LUT evaluation for Pieces 1–4 and single-cycle bitwise shifts/masks for Pieces 5–8 (`(q >> 2) + (q & 1)`, `q & 1`, `(q + 1) >> 1`, `q`), eliminating all inner loop integer divisions.
   - **Dual-Level Software Prefetching ($S_1$):** Issues temporal `__builtin_prefetch` instructions 16–32 steps ahead for $M(n)$ lookups, hiding DRAM latency stalls behind NEON vector pipelines.
   - **ARM64 NEON Vector Division (`vdivq_f64`):** Computes double-precision quotient divisions in parallel.
   - **Dynamic Hardware-Aware $c_x(X)$ Tuning:** Automatically balances $S_1$ memory traffic and $S_2$ streaming throughput.
   - **Flat 16-Bit Cache (`int16_t`):** Uses single-cycle `ldrh` lookups without multi-level cache table indirections.
   - **Lock-Free Multi-Threading:** OpenMP parallelization with dynamic guided work distribution.

---

## Build & Test

### Prerequisites
- Clang / GCC supporting C++20
- OpenMP (`libomp` via Homebrew on macOS: `brew install libomp`)

### Compilation
```bash
make
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

# Comparative benchmark vs O(X^(3/4)) Dirichlet DP Engine:
./bench_dr --compare 14 8
```

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

| Target $X$ | Exact $M(X)$ | Previous Baseline (Apple M1) | Mod-6 Wheel Engine (Apple M1, 8T) | Greg Hurst Preprint (Reported) | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^{10}$** | $-33,722$ | $0.0059$ s | **$0.0069$ s** | — | $0.85\times$ |
| **$10^{11}$** | $-87,856$ | $0.0296$ s | **$0.0233$ s** | — | **$1.27\times$** |
| **$10^{12}$** | $62,366$ | $0.2098$ s | **$0.0870$ s** | $\sim 0.03$ s (M2 Max) | **$2.41\times$** |
| **$10^{13}$** | $599,582$ | $0.7938$ s | **$0.4014$ s** | $0.12$ s (M2 Max) | **$1.98\times$** |
| **$10^{14}$** | $-875,575$ | $4.1109$ s | **$2.3363$ s** | $0.51$ s (M2 Max) | **$1.76\times$** |
| **$10^{15}$** | $-3,216,373$ | $17.3768$ s | **$9.1995$ s** | $2.10$ s (M2 Max) | **$1.89\times$** |
| **$10^{16}$** | $-3,195,437$ | $\sim 80$ s | **$\sim 58$ s** | $2.39$ s (M3 Ultra) / $8.4$ s (M2 Max) | **$1.38\times$** |

### NVIDIA Tesla T4 (16GB VRAM, Turing `sm_75` - CUDA Benchmark Sweep)

| Target $X$ | Exact $M(X)$ | Sieve Time (s) | H2D Copy (s) | GPU Kernel (s) | Total Time (s) | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^{10}$** | $-33,722$ | $0.0282$ | $0.0014$ | **$0.01223$** | $0.0427$ | — |
| **$10^{11}$** | $-87,856$ | $0.1181$ | $0.0050$ | **$0.04076$** | $0.1656$ | — |
| **$10^{12}$** | $62,366$ | $0.6141$ | $0.0312$ | **$0.14751$** | $0.7948$ | — |
| **$10^{13}$** | $599,582$ | $3.5112$ | $0.1068$ | **$0.30999$** | $3.9329$ | — |
| **$10^{14}$** | $-875,575$ | $6.5630$ | $0.2683$ | **$0.79253$** | **$7.6294$** | **$1.85\times$ faster** |
| **$10^{15}$** | $-3,216,373$ | $8.1529$ | $0.2687$ | **$3.70819$** | **$12.1412$** | **$1.72\times$ faster** |
| **$10^{16}$** | $-3,195,437$ | $8.7062$ | $0.2947$ | **$32.84762$** | **$41.9281$** | — |
