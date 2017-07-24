#ifndef LATTICE_BINOMIAL_H
#define LATTICE_BINOMIAL_H

#include "../option_enum.h"


// this contains serial implmentation of lattice model
class LatticeBinomialSerial {
private:
  double startPrice;
  double strikePrice;
  double timeToExpiry;
  double vol;
  double riskFree;
  double delta;
  double u;
  double p_u;

  double ** cache;

  void calcAmericanPrice(double curPrice,
                         int timeSteps,
                         int level,
                         int multiplier);

 void calcEuropeanPrice(double curPrice,
                        int timeSteps,
                        int level,
                        int multiplier);

public:
  LatticeBinomialSerial(double startPrice,
                     double strikePrice,
                     double timeToExpiry,
                     double vol,
                     double riskFree);

  double calcPrice(OptionStyle ostyle,
                   OptionType otype,
                   int time_steps = 100);
};

#endif
