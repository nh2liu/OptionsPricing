#include <algorithm>
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
                        double * cache) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > timeSteps) return;
  int colDim = timeSteps + 1;

  cache[timeSteps * colDim + i] = max(startPrice * pow(u, 2 * i - timeSteps) - strikePrice, 0.0);
  timeSteps--;

  while (timeSteps >= i) {
    cache[timeSteps * colDim + i] = (p_u * cache[(timeSteps + 1) * colDim + i + 1] +
                                    (1 - p_u) * cache[(timeSteps + 1) * colDim + i ]) * exp(-riskFree * delta);
    timeSteps--;
    __syncthreads();
  }
}

int main() {
  double startPrice = 100;
  double strikePrice = 100;
  double timeToExpiry = 1.5;
  double vol = 0.12;
  double riskFree = 0.005;
  int timeSteps = 100;

  double delta = timeToExpiry / timeSteps;
  double u = exp(vol * sqrt(delta));
  double p_u = (exp(riskFree * delta) - 1/u) / (u - 1/u);

  int N = timeSteps + 1;
  double * cache = new double[N * N];
  double * d_cache;

  cudaMalloc(&d_cache, N * N * sizeof(double));

  calcEuropeanOption<<<(timeSteps + 255)/256, 256>>>(timeSteps, startPrice,strikePrice,
                                                     riskFree, delta, u, p_u, d_cache);

  double * finalPrice;
  cudaMemcpy(finalPrice, d_cache,  sizeof(double), cudaMemcpyDeviceToHost);

  cout << "Price: " << *finalPrice << endl;
  cudaFree(d_cache);
}
