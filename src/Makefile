CXX=nvcc
CXXFLAGS= -std=c++11 -c
EXECFOLDER=../execs
all: pricing

pricing: main.o lattice_binomial.o cuda_binomial.o
	$(CXX) main.o lattice_binomial.o cuda_binomial.o -o ${EXECFOLDER}/pricing

main.o: main.cc
	$(CXX) $(CXXFLAGS) main.cc

lattice_binomial.o: serial_binomial/lattice_binomial.cc
	$(CXX) $(CXXFLAGS) serial_binomial/lattice_binomial.cc

cuda_binomial.o: cuda_binomial/cuda_binomial.cu
	$(CXX) $(CXXFLAGS) cuda_binomial/cuda_binomial.cu

.PHONY: clean
clean:
	rm *.o *.d ${EXECFOLDER}/pricing
