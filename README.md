# Deléglise–Rivat $O(X^{2/3})$ Mertens Engine

A self-contained, header-only C++20 implementation of the **Deléglise–Rivat combinatorial method with inclusion–exclusion reductions and micro-architectural optimizations** for evaluating the Mertens function:
$$ M(X) = \sum_{n=1}^X \mu(n) $$

Based on and extending the mathematical framework described in Greg Hurst's preprint (*"Practical Computations of the Mertens Function: $M(10^{24})$ and $M(10^{25})$"*, [arXiv:2607.07566](https://arxiv.org/abs/2607.07566)).

---

## Key Features & Optimizations

1. **Inclusion–Exclusion Reductions:**
   - **Outer Parity Split & Mod-6 Wheel Reduction:** Evaluates $S(y, u) - S(y/2, u) - S(y/3, u) + S(y/6, u)$ across square-free $k$ coprime to 6, cutting outer terms by $>69\%$.
   - **$S_1$ Parity Cancellation:** Cancels even quotient lookups across surviving odd/even intervals (stride-2 queries).
   - **$S_2$ Piecewise Reduction:** Partitions $j \le A, (j, 6)=1$ into 8 exact branchless arithmetic intervals.

2. **Compressed Hierarchical Mertens Table (CHMT) Algorithm (Unified Engine for all $X > 10^5$):**
   - **Seamless Multi-Tier Scaling:** Dynamically sizes `u_dense = std::min(u_total, 600M)`. For $X \le 10^{14}$ ($u \le 600\text{M}$), it operates as a 100% dense table with 0 compressed chunks (zero overhead). For massive $X \ge 10^{15}$, it seamlessly engages Tier 2 (2-bit bit-plane chunks), scaling $u$ up to **8 Billion entries** in ~3.1 GB RAM.
   - **13.3× Outer Term Reduction:** Slashes the outer summation workload for $10^{16}$ from 16.7 million terms down to 1.25 million terms ($N = X / u$).
   - **Constant-Time Hardware Popcount Reconstruction:** Evaluates $M(q)$ in single-digit CPU clock cycles via 4 branchless 64-bit hardware popcount instructions (`__builtin_popcountll(nz & ~sg)` and `__builtin_popcountll(nz & sg)`), requiring zero memory decompression.
   - **Branchless 8-Element Bit-Plane Packing:** Exploits $\mu(8k+4)=\mu(8k+8)=0$ and $\mu(8k+2)=-\mu_{\text{odd}}[2k+1]$ to pack 8 numbers per iteration directly from odd sieve bytes, achieving a **13× speedup** in table compression.
   - **Compile-Time Reciprocal Prime Sieving:** Replaces runtime 8-cycle `UDIV` instructions for small primes ($p \in \{3, 5, \dots, 71\}$) with 1-cycle `UMULH` compile-time fixed-point reciprocals (`sieve_prime_const<P>`).
   - **Two-Tier $S_1$ Execution Engine:** Resolves initial large quotients via compressed popcount reconstruction and the remaining 99.9% of quotients ($q \le u_{\text{dense}}$) through an 8-way unrolled direct read from `M_dense`.

3. **Micro-Architectural & SIMD Acceleration:**
   - **Zero-Allocation Compact Sieve (`FastSieve`):** Generates $M(n)$ prefix sums directly from odd-sieved segments via sequential 64-bit quad writes (`int16_t` quads), capping the $\mu$ table allocation to $<40\text{ MB}$ ($A_{\max}$) instead of $u$ (600 MB) and saving $>1.2\text{ GB}$ of DRAM traffic.
   - **4-Way Pipelined NEON $S_2$ Runner:** Unrolled 4-way (8 coprime-to-6 values per iteration), maintaining $j$ strictly in SIMD registers with vector step increments (`vaddq_f64`), pipelining 4 `vdivq_f64` instructions, and using `vcvtq_s64_f64` hardware conversions.
   - **Pure SIMD $S_1$ and Even-$n$ Correction:** Eliminates division overhead from software prefetching and evaluates $S_1$ arithmetic with SIMD vector additions and hardware integer truncation.
   - **Unified Dynamic Parallel Dispatch:** Merges all 4 OpenMP ranges into a single unified term vector with dynamic chunk scheduling (`schedule(dynamic, 8)`), eliminating barrier syncs and fixing thread starvation on small $k$.
   - **Flat 16-Bit Cache (`int16_t`):** Uses single-cycle `ldrh` lookups without multi-level cache table indirections.
   - **Dynamic Hardware-Aware $c_x(X)$ Tuning:** Automatically balances $S_1$ memory traffic and $S_2$ streaming throughput ($c_x = 0.95$ for $10^{15..16}$, $0.95$ for $10^{11..14}$, $0.70$ for $X \le 10^{10}$).

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

| Target $X$ | Exact $M(X)$ | Baseline (Apple M1, 8T) | CHMT Engine (Apple M1, 8T) | Greg Hurst Preprint (Reported) | Total Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^8$** | $1,928$ | $0.0021$ s | **$0.0008$ s** | — | **$2.6\times$** |
| **$10^9$** | $-222$ | $0.0062$ s | **$0.0020$ s** | — | **$3.1\times$** |
| **$10^{10}$** | $-33,722$ | $0.0151$ s | **$0.0057$ s** | — | **$2.6\times$** |
| **$10^{11}$** | $-87,856$ | $0.0548$ s | **$0.0211$ s** | — | **$2.6\times$** |
| **$10^{12}$** | $62,366$ | $0.1030$ s | **$0.0690$ s** | $\sim 0.03$ s (M2 Max) | **$1.5\times$** |
| **$10^{13}$** | $599,582$ | $0.4500$ s | **$0.3102$ s** | $0.12$ s (M2 Max) | **$1.5\times$** |
| **$10^{14}$** | $-875,575$ | $2.3370$ s | **$1.2805$ s** | $0.51$ s (M2 Max) | **$1.8\times$** |
| **$10^{15}$** | $-3,216,373$ | $14.8000$ s | **$7.6001$ s** | $2.10$ s (M2 Max) | **$1.95\times$** |
| **$10^{16}$** | $-3,195,437$ | $69.6000$ s | **$35.5171$ s** | $2.39$ s (M3 Ultra) | **$1.96\times$** |

### NVIDIA Tesla T4 (16GB VRAM, Turing `sm_75` - CUDA Benchmark Sweep)

| Target $X$ | Exact $M(X)$ | Sieve Time (s) | H2D Copy (s) | GPU Kernel (s) | Total Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^8$** | $1,928$ | $0.0026$ | $0.0002$ | **$0.00642$** | $0.1936$ |
| **$10^9$** | $-222$ | $0.0074$ | $0.0004$ | **$0.00552$** | $0.0137$ |
| **$10^{10}$** | $-33,722$ | $0.0280$ | $0.0013$ | **$0.01063$** | $0.0408$ |
| **$10^{11}$** | $-87,856$ | $0.1345$ | $0.0051$ | **$0.02758$** | $0.1689$ |
| **$10^{12}$** | $62,366$ | $0.6266$ | $0.0257$ | **$0.09034$** | $0.7455$ |
| **$10^{13}$** | $599,582$ | $2.9205$ | $0.1011$ | **$0.29919$** | $3.3257$ |
| **$10^{14}$** | $-875,575$ | $9.1989$ | $0.2752$ | **$0.71253$** | $10.1923$ |
| **$10^{15}$** | $-3,216,373$ | $8.8133$ | $0.3067$ | **$3.68370$** | $12.8182$ |
| **$10^{16}$** | $-3,195,437$ | $8.5388$ | $0.3075$ | **$31.94584$** | $40.9253$ |
