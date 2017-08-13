#ifndef __LATTICE_BINOMIAL_H__
#define __LATTICE_BINOMIAL_H__

#include "../AbstractValuation.h"
#include "../option.h"

// this contains serial implmentation of lattice model
class LatticeBinomialSerial : public AbstractValuation {
private:
  double ** cache;
  int timeSteps;

  void calcEuropeanPrice(double riskFree,
                         double strikePrice,
                         double priceAtCurStep,
                         double delta,
                         double u,
                         double p_u,
                         int currentStep,
                         int level,
                         int callPutModifier);

  void calcAmericanPrice(double riskFree,
                         double strikePrice,
                         double priceAtCurStep,
                         double delta,
                         double u,
                         double p_u,
                         int currentStep,
                         int level,
                         int callPutModifier);
public:
  LatticeBinomialSerial(int timeSteps);
  ~LatticeBinomialSerial();

  double calcPrice(Option & opt);

};

#endif
