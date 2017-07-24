CXX=g++
CXXFLAGS= -std=c++14 -Wall -MMD -Wextra -c
EXECFOLDER=../execs
all: pricing

pricing: main.o lattice_binomial.o
	$(CXX) main.o lattice_binomial.o -o ${EXECFOLDER}/pricing

main.o: main.cc
	$(CXX) $(CXXFLAGS) main.cc

lattice_binomial.o: serial_binomial/lattice_binomial.cc
	$(CXX) $(CXXFLAGS) serial_binomial/lattice_binomial.cc

.PHONY: clean
clean:
	rm *.o *.d
