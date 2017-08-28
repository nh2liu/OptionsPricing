#include <algorithm>
#include <random>
#include "serial_montecarlo.h"
#include "../option_enum.h"
using namespace std;

double MonteCarloSerial::calcAsianPrice(double riskFree,
                                       double startPrice,
                                       double strikePrice,
                                       double timeToExpiry,
                                       double u,
                                       double p_u,
                                       int callPutModifier) {
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0,1.0);

    double cumulativeSum = 0;

    for (int i = 0; i < numOfPaths; ++i) {
        double cumulativePrice = 0;
        double curPrice = startPrice;
        for (int j = 0; j < timeSteps; ++j) {
            curPrice = distribution(generator) < p_u ? curPrice * u : curPrice / u;
            cumulativePrice += curPrice;
        }
        cumulativeSum += max(callPutModifier * (cumulativePrice / timeSteps - strikePrice), 0.0) * exp(-riskFree * timeToExpiry);
    }
    return cumulativeSum / numOfPaths;
 }

double MonteCarloSerial::calcEuropeanPrice(double riskFree,
                                          double startPrice,
                                          double strikePrice,
                                          double timeToExpiry,
                                          double u,
                                          double p_u,
                                          int callPutModifier) {
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0,1.0);

    double cumulativeSum = 0;

    for (int i = 0; i < numOfPaths; ++i) {
        double curPrice = startPrice;
        for (int j = 0; j < timeSteps; ++j) {
            curPrice = distribution(generator) < p_u ? curPrice * u : curPrice / u;
        }
        cumulativeSum += max(callPutModifier * (curPrice - strikePrice), 0.0) * exp(-riskFree * timeToExpiry);
    }
    return cumulativeSum / numOfPaths;
 }

MonteCarloSerial::MonteCarloSerial(int timeSteps, int numOfPaths) : AbstractValuation(),
timeSteps{timeSteps}, numOfPaths{numOfPaths} {}

MonteCarloSerial::~MonteCarloSerial() {}

double MonteCarloSerial::calcPrice(Option & opt) {
    double delta = opt.timeToExpiry / timeSteps;
    double u = exp(opt.vol * sqrt(delta));
    double p_u = (exp(opt.riskFree * delta) - 1/u) / (u - 1/u);

    int callPutModifier = opt.otype == OptionType::Call ? 1 : -1;

    double price;
      // determining which type to call
    if (opt.ostyle == OptionStyle::European) {
        price = calcEuropeanPrice(opt.riskFree,
                                  opt.startPrice,
                                  opt.strikePrice,
                                  opt.timeToExpiry,
                                  u,
                                  p_u,
                                  callPutModifier);
    } else if (opt.ostyle == OptionStyle::Asian) {
        price = calcAsianPrice(opt.riskFree,
                               opt.startPrice,
                               opt.strikePrice,
                               opt.timeToExpiry,
                               u,
                               p_u,
                               callPutModifier);
    }
    return price;
}