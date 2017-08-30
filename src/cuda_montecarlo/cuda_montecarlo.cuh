#ifndef __CUDA_MONTECARLO__
#define __CUDA_MONTECARLO__
#include <curand.h>
#include <curand_kernel.h>
#include "../AbstractValuation.h"
#include "../option.h"

// this contains serial implmentation of lattice model
class MonteCarloCuda : public AbstractValuation {
private:
  int timeSteps;
  int numOfPaths;
  curandState* devStates;
public:
  MonteCarloCuda(int timeSteps, int numOfPaths);
  ~MonteCarloCuda();

  double calcPrice(Option & opt);

};

#endif