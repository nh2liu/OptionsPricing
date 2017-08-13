#include <algorithm>
#include "cuda_binomial.cuh"
#include "../option_enum.h"
#include <iostream>
using namespace std;

__global__
void calcEuropeanOption(int timeSteps,
                        double startPrice,
                        double strikePrice,
                        double riskFree,
                        double delta,
                        double u,
                        double p_u,
                        double * cache,
                        int callPutModifier) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > timeSteps) return;
  int colDim = timeSteps + 1;

  cache[timeSteps * colDim + i] = max(callPutModifier * (startPrice * pow(u, 2 * i - timeSteps) - strikePrice), 0.0);
  timeSteps--;

  while (timeSteps >= i) {
    cache[timeSteps * colDim + i] = (p_u * cache[(timeSteps + 1) * colDim + i + 1] +
                                    (1 - p_u) * cache[(timeSteps + 1) * colDim + i ]) * exp(-riskFree * delta);
    timeSteps--;
    __syncthreads();
  }
}

__global__
void calcAmericanOption(int timeSteps,
                        double startPrice,
                        double strikePrice,
                        double riskFree,
                        double delta,
                        double u,
                        double p_u,
                        double * cache,
                        int callPutModifier) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > timeSteps) return;
  int colDim = timeSteps + 1;

  cache[timeSteps * colDim + i] = max(callPutModifier * (startPrice * pow(u, 2 * i - timeSteps) - strikePrice), 0.0);
  timeSteps--;

  while (timeSteps >= i) {
    cache[timeSteps * colDim + i] = max((p_u * cache[(timeSteps + 1) * colDim + i + 1] +
                                        (1 - p_u) * cache[(timeSteps + 1) * colDim + i ]) * exp(-riskFree * delta),
                                        callPutModifier * (startPrice * pow(u, 2 * i - timeSteps) - strikePrice));
    timeSteps--;
    __syncthreads();
  }
}

LatticeBinomialCuda::LatticeBinomialCuda(int timeSteps) : AbstractValuation(), timeSteps{timeSteps} {}

LatticeBinomialCuda::~LatticeBinomialCuda() {}

double LatticeBinomialCuda::calcPrice(Option & opt) {
  double delta = opt.timeToExpiry / timeSteps;
  double u = exp(opt.vol * sqrt(delta));
  double p_u = (exp(opt.riskFree * delta) - 1/u) / (u - 1/u);
  int callPutModifier = opt.otype == OptionType::Call ? 1 : -1;

  int N = timeSteps + 1;
  double * d_cache;
  cudaMalloc(&d_cache, N * N * sizeof(double));

  if (opt.ostyle == OptionStyle::European) {
    calcEuropeanOption<<<(timeSteps + 255)/256, 256>>>(timeSteps, opt.startPrice, opt.strikePrice,
                                                       opt.riskFree, delta, u, p_u, d_cache, callPutModifier);
  } else if (opt.ostyle == OptionStyle::American) {
    calcAmericanOption<<<(timeSteps + 255)/256, 256>>>(timeSteps, opt.startPrice, opt.strikePrice,
                                                       opt.riskFree, delta, u, p_u, d_cache, callPutModifier);    
  }

  double finalPrice;
  cudaMemcpy(&finalPrice, d_cache,  sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_cache);
  
  return finalPrice;
}