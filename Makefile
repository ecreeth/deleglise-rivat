CXX := clang++
CXXFLAGS := -O3 -std=c++20 -Wall -Wextra -march=native -funroll-loops
OMP_FLAGS := -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp

TARGETS := bench_dr test_dr

all: $(TARGETS)

bench_dr: bench_dr.cpp mertens_dr.hpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) bench_dr.cpp -o bench_dr

test_dr: test_dr.cpp mertens_dr.hpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) test_dr.cpp -o test_dr

test: test_dr
	./test_dr

bench: bench_dr
	./bench_dr --benchmark 15

clean:
	rm -f $(TARGETS) *.o *.dSYM

.PHONY: all test bench clean
