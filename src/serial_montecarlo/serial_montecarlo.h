#ifndef __SERIAL_MONTECARLO_H__
#define __SERIAL_MONTECARLO_H__

#include "../AbstractValuation.h"
#include "../option.h"

// this contains serial implmentation of lattice model
class MonteCarloSerial : public AbstractValuation {
private:
  int timeSteps;
  int numOfPaths;

  double calcAsianPrice(double riskFree,
                 double startPrice,
                 double strikePrice,
                 double timeToExpiry,
                 double u,
                 double p_u,
                 int callPutModifier);
  double calcEuropeanPrice(double riskFree,
                    double startPrice,
                    double strikePrice,
                    double timeToExpiry,
                    double u,
                    double p_u,
                    int callPutModifier);
public:
  MonteCarloSerial(int timeSteps, int numOfPaths = 10000);
  ~MonteCarloSerial();

  double calcPrice(Option & opt);

};

#endif
