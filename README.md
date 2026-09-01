# Deléglise–Rivat $O(X^{2/3})$ Mertens Engine

A self-contained, header-only C++20 implementation of the **Deléglise–Rivat combinatorial method with inclusion–exclusion reductions** for evaluating the Mertens function:
$$ M(X) = \sum_{n=1}^X \mu(n) $$

Based on the mathematical framework and optimizations described in Greg Hurst's preprint (*"Practical Computations of the Mertens Function: $M(10^{24})$ and $M(10^{25})$"*, [arXiv:2607.07566](https://arxiv.org/abs/2607.07566)).

---

## Features

1. **Inclusion–Exclusion Reductions:**
   - **Outer Parity Split:** Evaluates $S(y, u) - S(y/2, u)$ for $k \le N/2$ and $S(y, u)$ for $N/2 < k \le N$ over odd square-free $k$, cutting outer terms by $>69\%$.
   - **$S_1$ Parity Cancellation:** Cancels even quotient lookups across surviving odd/even intervals (stride-2 queries).
   - **$S_2$ Modulo-6 Piecewise Reduction:** Partitions $j \le \nu_y, (j, 6)=1$ into 8 exact branchless arithmetic intervals.
2. **SIMD & Micro-Architectural Acceleration:**
   - **ARM64 NEON Vector Division (`vdivq_f64`):** Computes 4 double-precision quotient divisions in parallel.
   - **Zero-Modulo Stream Loops:** Directly generates $(j, 6)=1$ via $j=6m+1, 6m+5$, eliminating scalar integer division latencies.
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
- **Branchless Piecewise $S_2$ Modulo-6:** Evaluates arithmetic ranges without branch divergence.
- **Warp-Level Parallel Reduction:** Uses `__shfl_down_sync` and shared memory tree reductions for high-occupancy 64-bit atomic accumulation.

### CUDA Prerequisites
- NVIDIA CUDA Toolkit ($\ge 11.0$, `nvcc`)
- GPU Compute Capability $\ge 7.0$ (Default `CUDA_ARCH=sm_75` for NVIDIA Tesla T4)

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
# Or:
./test_cuda

# Evaluate a single value on GPU with detailed timing breakdown:
./bench_cuda 1000000000000000

# Run CUDA benchmark sweep:
./bench_cuda --benchmark 16

# Compare CPU vs CUDA GPU performance:
./bench_cuda --compare 16
```

---

## Benchmark Results

### Apple M1 (8 Threads)

| Target $X$ | Exact $M(X)$ | Runtime | Speedup vs $O(X^{3/4})$ DP |
| :--- | :--- | :--- | :--- |
| $10^{10}$ | $-33,722$ | **$0.0059$ s** | $1.81\times$ |
| $10^{11}$ | $-87,856$ | **$0.0296$ s** | $1.35\times$ |
| $10^{12}$ | $62,366$ | **$0.2098$ s** | $0.91\times$ |
| $10^{13}$ | $599,582$ | **$0.7938$ s** | $1.20\times$ |
| $10^{14}$ | $-875,575$ | **$3.0009$ s** | $2.15\times$ |
| $10^{15}$ | $-3,216,373$ | **$12.14$ s** | $6.55\times$ |
| $10^{16}$ | $-3,195,437$ | **$\sim 80$ s** | $7.8\times$ |

### NVIDIA Tesla T4 (16GB VRAM, Turing `sm_75` - Warp-Collaborative Architecture)

| Target $X$ | Exact $M(X)$ | CPU Time (s) | GPU Kernel (s) | GPU Total Time (s) | Kernel Speedup vs CPU | VRAM Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^{10}$** | $-33,722$ | $0.0276$ | **$0.0092$** | $0.1433$ | **$3.01\times$** | $0.2$ GB |
| **$10^{11}$** | $-87,856$ | $0.1286$ | **$0.0287$** | $0.1038$ | **$4.47\times$** | $0.3$ GB |
| **$10^{12}$** | $62,366$ | $0.6124$ | **$0.0788$** | $0.4841$ | **$7.77\times$** | $0.5$ GB |
| **$10^{13}$** | $599,582$ | $3.1251$ | **$0.3634$** | $2.5529$ | **$8.60\times$** | $0.9$ GB |
| **$10^{14}$** | $-875,575$ | $14.0050$ | **$0.8763$** | $7.8991$ | **$15.98\times$** | $1.2$ GB |
| **$10^{15}$** | $-3,216,373$ | $93.5753$ | **$3.2188$** | $17.0726$ | **$29.07\times$** | $3.7$ GB |
| **$10^{16}$** | $-3,195,437$ | $\sim 580$ s | **$31.5958$** | $46.9633$ | **$\sim 18.3\times$** | $3.7$ GB |





