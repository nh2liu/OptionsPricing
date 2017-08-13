#ifndef __CUDA_BINOMIAL__
#define __CUDA_BINOMIAL__

#include "../AbstractValuation.h"
#include "../option.h"

// this contains serial implmentation of lattice model
class LatticeBinomialCuda : public AbstractValuation {
private:
  int timeSteps;
public:
  LatticeBinomialCuda(int timeSteps);
  ~LatticeBinomialCuda();

  double calcPrice(Option & opt);

};

#endif