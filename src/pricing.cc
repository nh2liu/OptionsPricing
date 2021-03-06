#include <iostream>
#include "serial_binomial/lattice_binomial.h"
#include "cuda_binomial/cuda_binomial.cuh"
#include "serial_montecarlo/serial_montecarlo.h"
#include "cuda_montecarlo/cuda_montecarlo.cuh"
#include "option_enum.h"
#include "option.h"

using namespace std;

// module to price single options using
// all the valuation methods
// note that this is not hard tested and  montecarlo will produce out-of-bounds for 
// american style options

int main() {
  double price;
  double strike;
  double tte;
  double vol;
  double r;
  cout << "Enter: Current Price, Strike, Time To Expiry, Vol, Risk-Free Rate" << endl;
  cin >> price >> strike >> tte >> vol >> r;

  cout << "Enter: a or e." << endl;
  char ostyle_str;
  cin >> ostyle_str;
  OptionStyle ostyle;
  if (ostyle_str == 'e') {
    cout << "European option selected." << endl;
    ostyle = OptionStyle :: European;
  } else if (ostyle_str == 'a') {
    cout << "American option selected." << endl;
    ostyle = OptionStyle :: American;
  } else if (ostyle_str == 's') {
    cout << "Asian option selected." << endl;
    ostyle = OptionStyle :: Asian;
  }

  cout << "Enter: c or p." << endl;
  char otype_str;
  cin >> otype_str;
  OptionType otype;
  if (otype_str == 'c') {
    cout << "Call option selected." << endl;
    otype = OptionType::Call;
  } else {
    cout << "Put option selected." << endl;
    otype = OptionType :: Put;
  }

  Option opt(ostyle, otype, price, strike, tte, vol, r);

  cout << "Timesteps: " << endl;
  int t;
  cin >> t;
  cout << "t: " << t << endl;
  LatticeBinomialSerial bValSerial(t);
  cout << bValSerial.calcPrice(opt) << endl;

  LatticeBinomialCuda bValCuda(t);
  cout << bValCuda.calcPrice(opt) << endl;

  MonteCarloSerial bValMonteCarloSerial(t, 10000);
  cout << bValMonteCarloSerial.calcPrice(opt) << endl;

  MonteCarloCuda bValMonteCarloCuda(t, 10000);
  cout << bValMonteCarloCuda.calcPrice(opt) << endl;
}
