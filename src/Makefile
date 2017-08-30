CXX=nvcc
CXXFLAGS= -arch=sm_60 -std=c++11 
EXECFOLDER=../execs
all: benchmark pricing

benchmark: benchmark.o lattice_binomial.o cuda_binomial.o serial_montecarlo.o
	$(CXX) $(CXXFLAGS) benchmark.o lattice_binomial.o cuda_binomial.o serial_montecarlo.o -o ${EXECFOLDER}/benchmark

pricing: pricing.o lattice_binomial.o cuda_binomial.o serial_montecarlo.o cuda_montecarlo.o
	$(CXX) $(CXXFLAGS) pricing.o lattice_binomial.o cuda_binomial.o serial_montecarlo.o cuda_montecarlo.o -o ${EXECFOLDER}/pricing

benchmark.o: benchmark.cc
	$(CXX) $(CXXFLAGS) -c benchmark.cc

pricing.o: pricing.cc
	$(CXX) $(CXXFLAGS) -c pricing.cc

lattice_binomial.o: serial_binomial/lattice_binomial.cc
	$(CXX) $(CXXFLAGS) -c serial_binomial/lattice_binomial.cc

cuda_binomial.o: cuda_binomial/cuda_binomial.cu
	$(CXX) $(CXXFLAGS) -c cuda_binomial/cuda_binomial.cu

serial_montecarlo.o: serial_montecarlo/serial_montecarlo.h
	$(CXX) $(CXXFLAGS) -c serial_montecarlo/serial_montecarlo.cc

cuda_montecarlo.o: cuda_montecarlo/cuda_montecarlo.cu
	$(CXX) $(CXXFLAGS) -c cuda_montecarlo/cuda_montecarlo.cu

.PHONY: clean
clean:
	rm *.o *.d ${EXECFOLDER}/pricing ${EXECFOLDER}/benchmark
