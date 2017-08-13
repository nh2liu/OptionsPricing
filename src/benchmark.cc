#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include "serial_binomial/lattice_binomial.h"
#include "cuda_binomial/cuda_binomial.cuh"
#include "AbstractValuation.h"
#include "option_enum.h"
#include "option.h"


using namespace std;

int main() {
  int t;
  cout << "Number of timesteps: " << endl;
  ifstream ifs{"timesteps.txt"};
  ifs >> t;
  cout << t << " timesteps confirmed." << endl;
  double price;
  double strike;
  double tte;
  double vol;
  double r;
  char ostyle_str;
  char otype_str;
  OptionStyle ostyle;
  OptionType otype;

  vector <Option> optionBasket;

  cout << "Loading Basket." << endl;
  while (cin >> price >> strike >> tte >> vol >> r >> ostyle_str >> otype_str) {   
    
    if (ostyle_str == 'e') {
      ostyle = OptionStyle :: European;
    } else {
      ostyle = OptionStyle :: American;
    }

    
    if (otype_str == 'c') {
      otype = OptionType :: Call;
    } else {
      otype = OptionType :: Put;
    }

    optionBasket.push_back(Option(ostyle, otype, price, strike, tte, vol, r));
  }
  cout << optionBasket.size() << " options will be evaluated." << endl;
  
  vector<AbstractValuation *> oVals;

  oVals.push_back(new LatticeBinomialSerial(t));
  oVals.push_back(new LatticeBinomialCuda(t));

  for (AbstractValuation * oVal : oVals) {
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    for (Option & opt : optionBasket) {
      oVal->calcPrice(opt);
    }

    chrono::steady_clock::time_point end= chrono::steady_clock::now();
    cout << "t=" << chrono::duration_cast<chrono::microseconds>(end - begin).count() <<endl;
  }

  for (AbstractValuation * AV : oVals) {
    delete AV;
  }

  
}
