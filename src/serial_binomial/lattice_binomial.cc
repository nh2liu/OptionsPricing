#include <algorithm>
#include "lattice_binomial.h"
using namespace std;

void LatticeBinomialSerial::calcAmericanPrice(double curPrice,
                                              int timeSteps,
                                              int level,
                                              int multiplier) {
 if (timeSteps == 0) {
   cache[timeSteps][level] = std::max(multiplier * (curPrice - strikePrice), 0.0);
 } else {
   int newTimeSteps = timeSteps - 1;
   int u_level = level + 1;

   if (cache[newTimeSteps][u_level] == -1) {
     calcAmericanPrice(curPrice * u, newTimeSteps, u_level, multiplier);
   }
   if (cache[newTimeSteps][level] == -1) {
     calcAmericanPrice(curPrice / u, newTimeSteps, level, multiplier);
   }
   double v_u = cache[newTimeSteps][u_level];
   double v_d = cache[newTimeSteps][level];

   cache[timeSteps][level] = std::max((p_u * v_u + (1 - p_u) * v_d) *
                                  exp(-riskFree * delta),
                                  multiplier * (curPrice - strikePrice));
 }
}

void LatticeBinomialSerial::calcEuropeanPrice(double curPrice,
                                              int timeSteps,
                                              int level,
                                              int multiplier) {
  if (timeSteps == 0) {
    cache[timeSteps][level] = max(multiplier * (curPrice - strikePrice), 0.0);
  } else {
    int newTimeSteps = timeSteps - 1;
    int u_level = level + 1;

    if (cache[newTimeSteps][u_level] == -1) {
      calcEuropeanPrice(curPrice * u, newTimeSteps, u_level, multiplier);
    }
    if (cache[newTimeSteps][level] == -1) {
      calcEuropeanPrice(curPrice / u, newTimeSteps, level, multiplier);
    }
    double v_u = cache[newTimeSteps][u_level];
    double v_d = cache[newTimeSteps][level];

    cache[timeSteps][level] = (p_u * v_u + (1 - p_u) * v_d) *
                               exp(-riskFree * delta);
  }
}

LatticeBinomialSerial::LatticeBinomialSerial(double startPrice,
                                             double strikePrice,
                                             double timeToExpiry,
                                             double vol,
                                             double riskFree):
startPrice{startPrice}, strikePrice{strikePrice}, timeToExpiry{timeToExpiry},
vol{vol}, riskFree{riskFree}, cache{nullptr} {}

double LatticeBinomialSerial::calcPrice(OptionStyle ostyle,
                                        OptionType otype,
                                        int timeSteps) {
    // establishing cache for option prices
  int cache_dim = timeSteps + 1;
  cache = new double * [cache_dim];
  for (int i = 0; i < cache_dim; ++i) {
    cache[i] = new double[cache_dim];
    for (int j = 0; j < cache_dim; ++j) {
      cache[i][j] = -1;
    }
  }

  // setting up option variables.
  delta = timeToExpiry / timeSteps;
  u = exp(vol * sqrt(delta));
  p_u = (exp(riskFree * delta) - 1/u) / (u - 1/u);

  int multiplier = otype == OptionType::Call ? 1 : -1;
  // determining which type to call
  if (ostyle == OptionStyle::European) {
    calcEuropeanPrice(startPrice, timeSteps, 0, multiplier);
  } else if (ostyle == OptionStyle::American) {
    calcAmericanPrice(startPrice, timeSteps, 0, multiplier);
  }

  double p = cache[timeSteps][0];

  // resetting cache
  for (int i = 0; i < cache_dim; ++i) delete[] cache[i];
  delete[] cache;

  return p;
};
