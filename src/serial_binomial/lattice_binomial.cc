#include <algorithm>
#include "lattice_binomial.h"
#include "../option_enum.h"
using namespace std;

LatticeBinomialSerial::LatticeBinomialSerial(int timeSteps) : AbstractValuation(), timeSteps{timeSteps} {
  // establishing cache for option prices
  int cache_dim = timeSteps + 1;

  cache = new double * [cache_dim];
  for (int i = 0; i < cache_dim; ++i) {
    cache[i] = new double[cache_dim];
    for (int j = 0; j < cache_dim; ++j) {
      cache[i][j] = -1;
    }
  }
}

LatticeBinomialSerial::~LatticeBinomialSerial() {
  // resetting cache
  int cache_dim = timeSteps + 1;
  for (int i = 0; i < cache_dim; ++i) delete[] cache[i];
  delete[] cache;
}

void LatticeBinomialSerial::calcAmericanPrice(double riskFree,
                                              double strikePrice,
                                              double priceAtCurStep,
                                              double delta,
                                              double u,
                                              double p_u,
                                              int currentStep,
                                              int level,
                                              int callPutModifier) {
 if (currentStep == 0) {
   cache[currentStep][level] = max(callPutModifier * (priceAtCurStep - strikePrice), 0.0);
 } else {
   int newStep = currentStep - 1;
   int u_level = level + 1;

   if (cache[newStep][u_level] == -1) {
     calcAmericanPrice(riskFree, strikePrice, priceAtCurStep * u, delta, u, p_u, newStep, u_level, callPutModifier);
   }
   if (cache[newStep][level] == -1) {
     calcAmericanPrice(riskFree, strikePrice, priceAtCurStep / u, delta, u, p_u, newStep, level, callPutModifier);
   }
   double v_u = cache[newStep][u_level];
   double v_d = cache[newStep][level];

   cache[currentStep][level] = max((p_u * v_u + (1 - p_u) * v_d) *
                                  exp(-riskFree * delta),
                                  callPutModifier * (priceAtCurStep - strikePrice));
 }
}

void LatticeBinomialSerial::calcEuropeanPrice(double riskFree,
                                              double strikePrice,
                                              double priceAtCurStep,
                                              double delta,
                                              double u,
                                              double p_u,
                                              int currentStep,
                                              int level,
                                              int callPutModifier) {
 if (currentStep == 0) {
   cache[currentStep][level] = max(callPutModifier * (priceAtCurStep - strikePrice), 0.0);
 } else {
   int newStep = currentStep - 1;
   int u_level = level + 1;

   if (cache[newStep][u_level] == -1) {
      calcEuropeanPrice(riskFree, strikePrice, priceAtCurStep * u, delta, u, p_u, newStep, u_level, callPutModifier);
   }
   if (cache[newStep][level] == -1) {
      calcEuropeanPrice(riskFree, strikePrice, priceAtCurStep / u, delta, u, p_u, newStep, level, callPutModifier);
   }
   double v_u = cache[newStep][u_level];
   double v_d = cache[newStep][level];

   cache[currentStep][level] = (p_u * v_u + (1 - p_u) * v_d) *
                                  exp(-riskFree * delta);
 }
}



double LatticeBinomialSerial::calcPrice(Option & opt) {
  // setting up option variables.
  double delta = opt.timeToExpiry / timeSteps;
  double u = exp(opt.vol * sqrt(delta));
  double p_u = (exp(opt.riskFree * delta) - 1/u) / (u - 1/u);

  int callPutModifier = opt.otype == OptionType::Call ? 1 : -1;

  // determining which type to call
  if (opt.ostyle == OptionStyle::European) {
    calcEuropeanPrice(opt.riskFree,
                      opt.strikePrice,
                      opt.startPrice,
                      delta,
                      u,
                      p_u,
                      timeSteps,
                      0,
                      callPutModifier);
  } else if (opt.ostyle == OptionStyle::American) {
    calcAmericanPrice(opt.riskFree,
                      opt.strikePrice,
                      opt.startPrice,
                      delta,
                      u,
                      p_u,
                      timeSteps,
                      0,
                      callPutModifier);
  }

  double p = cache[timeSteps][0];
  
  int cache_dim = timeSteps + 1;
  for (int i = 0; i < cache_dim; ++i) {
    for (int j = 0; j < cache_dim; ++j) {
      cache[i][j] = -1;
    }
  }
  return p;
};
