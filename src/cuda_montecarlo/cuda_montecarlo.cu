#include <algorithm>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_montecarlo.cuh"
#include "../option_enum.h"
using namespace std;

// reduce sum version
__device__ void reduceSum(int idx, double * globalArr, double * finalSum) {
  __shared__ double sData[256];

  sData[threadIdx.x] = globalArr[idx];

  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset *=2) {
    int index = 2 * offset * threadIdx.x;
    if (index < blockDim.x) {
        sData[index] += sData[index + offset];
    }
    __syncthreads();
  }

  // if the thread idx is x, we add the local sum in shared memory
  // to the final sum
  if (threadIdx.x == 0) atomicAdd(finalSum, sData[0]);
}

__global__ void initRand(unsigned int seed, curandState * states) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              idx, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[idx]);
}

__global__ void pricePathAsian(double riskFree,
                               double startPrice,
                               double strikePrice,
                               double timeToExpiry,
                               double u,
                               double p_u,
                               int callPutModifier,
                               int timeSteps,
                               curandState* globalState,
                               double * pathPrices,
                               double * finalPrice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curandState localState = globalState[idx];

  double cumulativePrice = 0;
  double curPrice = startPrice;
  for (int j = 0; j < timeSteps; ++j) {
    curPrice = curand_uniform(&localState) < p_u ? curPrice * u : curPrice / u;
    cumulativePrice += curPrice;
  }
  pathPrices[idx] = max(callPutModifier * (cumulativePrice / timeSteps - strikePrice), 0.0) * exp(-riskFree * timeToExpiry);
  __syncthreads();
  reduceSum(idx, pathPrices, finalPrice);
}

__global__ void pricePathEuropean(double riskFree,
                                  double startPrice,
                                  double strikePrice,
                                  double timeToExpiry,
                                  double u,
                                  double p_u,
                                  int callPutModifier,
                                  int timeSteps,
                                  curandState* globalState,
                                  double * pathPrices,
                                  double * finalPrice) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curandState localState = globalState[idx];

  double curPrice = startPrice;
  for (int j = 0; j < timeSteps; ++j) {
    curPrice = curand_uniform(&localState) < p_u ? curPrice * u : curPrice / u;
  }
  pathPrices[idx] = max(callPutModifier * (curPrice - strikePrice), 0.0) * exp(-riskFree * timeToExpiry);
  __syncthreads();
  reduceSum(idx, pathPrices, finalPrice);
}

MonteCarloCuda::MonteCarloCuda(int timeSteps, int numOfPaths) : AbstractValuation(),
timeSteps{timeSteps}, numOfPaths{numOfPaths} {
  cudaMalloc(&devStates, numOfPaths * sizeof(curandState));
  initRand<<<(numOfPaths + 255)/256, 256>>>(1, devStates);
}

MonteCarloCuda::~MonteCarloCuda() {}

double MonteCarloCuda::calcPrice(Option & opt) {
    double delta = opt.timeToExpiry / timeSteps;
    double u = exp(opt.vol * sqrt(delta));
    double p_u = (exp(opt.riskFree * delta) - 1/u) / (u - 1/u);

    int callPutModifier = opt.otype == OptionType::Call ? 1 : -1;

    double finalPrice;
    double * d_pathPrices;
    double * d_finalPrice;
    cudaMalloc(&d_pathPrices, sizeof(double) * numOfPaths);
    cudaMalloc(&d_finalPrice, sizeof(double));
    
    
      // determining which type to call
    if (opt.ostyle == OptionStyle::European) {
      pricePathEuropean<<<(numOfPaths + 255)/256, 256>>>
                        (opt.riskFree,
                        opt.startPrice,
                        opt.strikePrice,
                        opt.timeToExpiry,
                        u,
                        p_u,
                        timeSteps,
                        callPutModifier,
                        devStates,
                        d_pathPrices,
                        d_finalPrice);

      cudaMemcpy(d_finalPrice, &finalPrice, sizeof(double), cudaMemcpyDeviceToHost);
    } else if (opt.ostyle == OptionStyle::Asian) {
      pricePathAsian<<<(numOfPaths + 255)/256, 256>>>
                        (opt.riskFree,
                        opt.startPrice,
                        opt.strikePrice,
                        opt.timeToExpiry,
                        u,
                        p_u,
                        timeSteps,
                        callPutModifier,
                        devStates,
                        d_pathPrices,
                        d_finalPrice);
      cudaMemcpy(d_finalPrice, &finalPrice, sizeof(double), cudaMemcpyDeviceToHost);
      finalPrice /= numOfPaths;
    }
    return finalPrice;
}