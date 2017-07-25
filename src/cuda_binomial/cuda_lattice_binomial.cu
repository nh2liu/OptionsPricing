#include <iostream>

class LatticeBinomialCuda {
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
  LatticeBinomialCuda(double startPrice,
                     double strikePrice,
                     double timeToExpiry,
                     double vol,
                     double riskFree);

  double calcPrice(OptionStyle ostyle,
                   OptionType otype,
                   int time_steps = 100);
};

__global__
void calcEuropeanOption() {

}
