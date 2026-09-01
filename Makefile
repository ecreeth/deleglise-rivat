CXX := clang++
CXXFLAGS := -O3 -std=c++20 -Wall -Wextra -march=native -funroll-loops
OMP_FLAGS := -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp

TARGETS := bench_dr test_dr
CUDA_TARGETS := bench_cuda test_cuda

# CUDA Configuration (Default: sm_75 for NVIDIA Tesla T4)
NVCC := nvcc
CUDA_ARCH ?= sm_75
NVCCFLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) -lineinfo --extended-lambda --expt-relaxed-constexpr -Xcompiler "-O3 -Wall -fopenmp"

all: $(TARGETS)

bench_dr: bench_dr.cpp mertens_dr.hpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) bench_dr.cpp -o bench_dr

test_dr: test_dr.cpp mertens_dr.hpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) test_dr.cpp -o test_dr

cuda: $(CUDA_TARGETS)

bench_cuda: bench_cuda.cu mertens_cuda.cuh mertens_dr.hpp
	$(NVCC) $(NVCCFLAGS) bench_cuda.cu -o bench_cuda

test_cuda: test_cuda.cu mertens_cuda.cuh mertens_dr.hpp
	$(NVCC) $(NVCCFLAGS) test_cuda.cu -o test_cuda

test: test_dr
	./test_dr

test-cuda: test_cuda
	./test_cuda

bench: bench_dr
	./bench_dr --benchmark 15

clean:
	rm -f $(TARGETS) $(CUDA_TARGETS) *.o *.dSYM

.PHONY: all cuda test test-cuda bench clean
