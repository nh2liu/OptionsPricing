#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include "serial_binomial/lattice_binomial.h"
#include "cuda_binomial/cuda_binomial.cuh"
#include "AbstractValuation.h"
#include "option_enum.h"
#include "option.h"

/* benchmark is the testing suite for options
*/

using namespace std;

// exports benchmark results in a file
void exportResults(string fileName,
                   vector<double * > & modelPrices,
                   vector <Option> & optionBasket,
                   vector<AbstractValuation*> & oVals) {
  ofstream ofs{fileName};

  int sizeOfBasket = optionBasket.size();
  ofs << "OptionStyle,OptionType,CurrentPrice,Strike,TimeToExpiry,Vol,RiskFreeRate,";

  for (AbstractValuation * oVal : oVals) {
    ofs << oVal->getName() << ",";
  }
  ofs << endl;

  // printing out options 1 by 1
  for (int i = 0; i < sizeOfBasket; ++i) {
    Option & option = optionBasket[i];

    if (option.ostyle == OptionStyle::American) {
      ofs << "American";
    } else if (option.ostyle == OptionStyle::European) {
      ofs << "European";
    } else if (option.ostyle == OptionStyle::Asian) {
      ofs << "Asian";
    }
    ofs << ",";

    if (option.otype == OptionType::Call) {
      ofs << "Call";
    } else if (option.otype == OptionType::Put) {
      ofs << "Put";
    }
    ofs << ",";

    ofs << option.startPrice << "," << option.strikePrice << "," << option.timeToExpiry << "," << option.vol << "," << option.riskFree << ",";

    for (double * prices : modelPrices) {
      ofs << prices[i] << ",";
    }

    ofs << endl;
  }
  ofs.close();
}


int main() {
  

  cout << "This is the testing suite for options" << endl;
  cout << "Please indicate which files you want to benchmark." << endl;
  cout << "[1] LatticeSerial\n[2] LatticeCuda\n[3] All benchmarks" << endl;
  
  int c;
  cin >> c;

  int t;
  cout << "Number of timesteps: " << endl;
  cin >> t;
  cout << t << " timesteps confirmed." << endl;

  // pushing in models
  vector<AbstractValuation *> oVals;

  if (c == 1 || c == 3) {
   oVals.push_back(new LatticeBinomialSerial(t));
  }
  if (c == 2 || c == 3) {
    oVals.push_back(new LatticeBinomialCuda(t));
  }


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

  cout << "Please indicate which porfolio to test." << endl;
  string portfolioName;
  cin >> portfolioName;
  ifstream ifs{portfolioName};

  while (!ifs.good()) {
    cout << "Invalid file. Please re-enter porfolio name." << endl;
    cin >> portfolioName;
    ifs.open((portfolioName));
  }

  cout << "Loading Basket." << endl;
  while (ifs >> price >> strike >> tte >> vol >> r >> ostyle_str >> otype_str) {   
    
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

  ifs.close();
  int sizeOfBasket = optionBasket.size();
  cout << sizeOfBasket << " options will be evaluated." << endl;
  
  vector<double * > modelPrices;

  for (AbstractValuation * oVal : oVals) {
    double * prices = new double[sizeOfBasket];
    modelPrices.push_back(prices);

    cout << "Evaluating " << oVal->getName() << "." << endl;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    for (int i = 0; i < sizeOfBasket; ++i) {
      prices[i] = oVal->calcPrice(optionBasket[i]);
    }

    chrono::steady_clock::time_point end= chrono::steady_clock::now();
    cout << "Took " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << " microseconds to evaluate portfolio." <<endl;
  }

  cout.precision(3);

  exportResults(portfolioName + ".csv", modelPrices, optionBasket, oVals);

  for (double * prices : modelPrices) {
    delete prices;
  }

  for (AbstractValuation * AV : oVals) {
    delete AV;
  }
}
