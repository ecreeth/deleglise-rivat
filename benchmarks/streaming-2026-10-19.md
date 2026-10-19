# Streaming-table CPU comparison — 2026-10-19

The revised CPU path improved every measured power from `10^6` through
`10^16` against commit `edeb00d`. This is the fastest configuration tested
in this tuning session, not a claim of the fastest Mertens implementation
on every machine. Gains vary with input size and memory behavior.

## Environment and method

- Intel Core i7-7700, 15.87 GiB RAM, Windows, four OpenMP workers.
- Native MSVC 2022 x64, `/O2 /std:c++20 /utf-8 /openmp /EHsc`.
- Baseline: `mertens_dr.hpp` at `edeb00d`, with MSVC aliases for its GCC
  prefetch/popcount intrinsics. Current: the streaming-table revision.
- Default tuning; no `MERTENS_*` environment overrides.
- Both versions were compiled into one comparison executable under separate
  namespaces. Each warmed up with `M(10^8)`. Measurement order alternated
  baseline/current and current/baseline to reduce order bias.
- For powers 6 through 15, each sample averaged respectively
  `64,16,128,64,16,4,1,1,1,1` calls. The table reports the median of seven
  samples through power 12, five at power 13, and three at powers 14 and 15.
  Every result was checked against the expected integer value.
- `10^16` used one separate process per version, baseline first, with the
  same warmup. Peak memory was sampled using Windows
  `GetProcessMemoryInfo`; committed memory includes process allocations.
  This large-input timing is a single observation, not a repeated median.
- Timed work includes table construction and summation. No competing build
  or benchmark jobs were run during the recorded comparison.

## Results

| X | `edeb00d` (s) | Current (s) | Time reduction |
|---|---:|---:|---:|
| 10^6 | 0.003661 | 0.000697 | 81.0% |
| 10^7 | 0.013829 | 0.005646 | 59.2% |
| 10^8 | 0.000652 | 0.000565 | 13.3% |
| 10^9 | 0.001664 | 0.001339 | 19.5% |
| 10^10 | 0.009740 | 0.007501 | 23.0% |
| 10^11 | 0.029642 | 0.022570 | 23.9% |
| 10^12 | 0.113996 | 0.089274 | 21.7% |
| 10^13 | 0.661647 | 0.489945 | 26.0% |
| 10^14 | 2.617211 | 2.000362 | 23.6% |
| 10^15 | 20.540045 | 14.297179 | 30.4% |
| 10^16 | 139.176680 | 102.109103 | 26.6% |

At `10^16`, both versions returned `-3195437`:

| Peak process memory | `edeb00d` | Current |
|---|---:|---:|
| Committed | 3.629 GiB | 2.688 GiB |
| Working set | 3.623 GiB | 2.678 GiB |

Peak committed memory decreased by about 26%. For context on timing noise,
the three `10^15` samples were `18.933,27.019,20.540` seconds for the baseline
and `13.565,14.818,14.297` seconds for current. Small sub-millisecond
differences should also be interpreted cautiously. Powers 1 through 5 retain
the existing small-input algorithm and were checked for correctness, not
included in the comparative timing sweep.

## What changed

The mathematical decomposition remains Deléglise–Rivat. The implementation
now streams full-number byte-log sieve segments directly into the dense
prefix and compressed table, avoiding a full temporary odd-Mobius array.
Scratch storage is 128 KiB per requested worker. Phase-correct stencils
pre-mark small prime weights and square factors, and SSE2 finalizes sixteen
Mobius bytes and packs sixteen bit-plane entries at once.

Each compressed 256-value chunk stores two 32-byte bit planes and four
32-bit checkpoints. A lookup reads one word from each plane and uses two
popcounts. The footprint is 0.3125 bytes per compressed entry, plus the
dense prefix, retained Mobius values, and small construction metadata.

On x86-64, automatic compression starts at `10^11`; smaller inputs retain
the dense strategy because compression did not improve them consistently.
The compressed cutoff defaults to eight billion, with the dense prefix
targeting the smaller of 128 million and one quarter of the cutoff, expanded
to cover the algorithm's endpoints. A separate cutoff experiment measured
106.84 seconds with eight billion versus 110.57 seconds with ten billion at
`10^16`, while eight billion used less memory. Those exploratory runs are
separate from the final comparison above.

Second-difference quotient predictors, reciprocal caches, and a 32-bit S2
variant were evaluated and rejected because they did not improve reliably
over the retained exact SSE2 quotient kernels. A constant or linearly growing
speedup is not expected across changes in algorithm and table footprint.

## Validation and reproduction

All four regression configurations passed: normal SIMD/OpenMP, scalar
quotients/OpenMP, scalar sieve/OpenMP, and scalar without OpenMP. Checks
compare dense, odd, and streaming sieves against an independent reference;
exercise powers of two, stencil periods, 64/256-value boundaries, block
edges, arbitrary dense cutoffs, reduced OpenMP teams, quotient boundaries,
and square roots near the signed 64-bit limit. The original 20 known-value
tests through `10^15` also passed. CUDA and non-x86 hardware were not tested
in this session.

Build and run current with:

```bat
cl /O2 /std:c++20 /utf-8 /openmp /EHsc bench_dr.cpp /Fe:bench_dr.exe
bench_dr.exe --benchmark 15 4 3
bench_dr.exe 10000000000000000 4
```

For a comparison on another machine, build `edeb00d` and current with the
same compiler and flags, warm both, and alternate repeated measurements.
The CLI sweep runs each input separately; its medians need not match the
batched comparison above.

`MERTENS_USE_CHMT=0` forces dense tables; `MERTENS_USE_CHMT=1` enables
compression from `10^11`, including on other architectures.
`MERTENS_CHMT_U_CAP` accepts 100 million through ten billion, subject to
covering the algorithm's endpoints. Explicit `MERTENS_FX` or
`MERTENS_U_CAP` preserves dense tuning unless CHMT is explicitly enabled.
Compiler definitions `MERTENS_USE_SSE2_DIV=0` and
`MERTENS_USE_SIMD_SIEVE=0` select the scalar implementations independently.
