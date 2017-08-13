CXX=nvcc
CXXFLAGS= -std=c++11 -c
EXECFOLDER=../execs
all: benchmark pricing

benchmark: benchmark.o lattice_binomial.o cuda_binomial.o
	$(CXX) benchmark.o lattice_binomial.o cuda_binomial.o -o ${EXECFOLDER}/benchmark

pricing: pricing.o lattice_binomial.o cuda_binomial.o
	$(CXX) pricing.o lattice_binomial.o cuda_binomial.o -o ${EXECFOLDER}/pricing

benchmark.o: benchmark.cc
	$(CXX) $(CXXFLAGS) benchmark.cc

pricing.o: pricing.cc
	$(CXX) $(CXXFLAGS) pricing.cc

lattice_binomial.o: serial_binomial/lattice_binomial.cc
	$(CXX) $(CXXFLAGS) serial_binomial/lattice_binomial.cc

cuda_binomial.o: cuda_binomial/cuda_binomial.cu
	$(CXX) $(CXXFLAGS) cuda_binomial/cuda_binomial.cu

.PHONY: clean
clean:
	rm *.o *.d ${EXECFOLDER}/pricing ${EXECFOLDER}/benchmark
