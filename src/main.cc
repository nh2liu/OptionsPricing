#include <iostream>
#include "serial_binomial/lattice_binomial.h"
#include "option_enum.h"
#include "option.h"

using namespace std;

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
  } else {
    cout << "American option selected." << endl;
    ostyle = OptionStyle :: American;
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
  while (cin >> t) {
    cout << "t: " << t << endl;
    LatticeBinomialSerial bVal(t);
    cout << bVal.calcPrice(opt) << endl;
  }
}
