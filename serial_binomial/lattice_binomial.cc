#include <iostream>
#include <math.h>
using namespace std;

enum class OptionType {American, European};

class lattice_model_serial {
private:
  double start_price;
  double strike_price;
  double time_to_expiry;
  double vol;
  double risk_free;
  double delta;
  double u;
  double p_u;

  double ** cache;
  void calc_european_price(double cur_price,
                           int time_steps,
                           int level) {
    if (time_steps == 0) {
      cache[time_steps][level] = max(cur_price - strike_price, 0.0);
    } else {
      int new_timesteps = time_steps - 1;
      int u_level = level + 1;

      if (cache[new_timesteps][u_level] == -1) {
        calc_european_price(cur_price * u, new_timesteps, u_level);
      }
      if (cache[new_timesteps][level] == -1) {
        calc_european_price(cur_price / u, new_timesteps, level);
      }
      double v_u = cache[new_timesteps][u_level];
      double v_d = cache[new_timesteps][level];

      cache[time_steps][level] = (p_u * v_u + (1 - p_u) * v_d) *
                                 exp(-risk_free * delta);
    }
  }
public:
  lattice_model_serial(double start_price,
                       double strike_price,
                       double time_to_expiry,
                       double vol,
                       double risk_free)
                       : start_price{start_price}, strike_price{strike_price},
                         time_to_expiry{time_to_expiry}, vol{vol},
                         risk_free{risk_free} {}

  double price(int time_steps = 100) {
    int cache_dim = time_steps + 1;
    cache = new double * [cache_dim];
    for (int i = 0; i < cache_dim; ++i) {
      cache[i] = new double[cache_dim];
      for (int j = 0; j < cache_dim; ++j) {
        cache[i][j] = -1;
      }
    }

    delta = time_to_expiry / time_steps;
    u = exp(vol * sqrt(delta));
    p_u = (exp(risk_free * delta) - 1/u) / (u - 1/u);
    
    calc_european_price(start_price, time_steps, 0);

    double p = cache[time_steps][0];

    for (int i = 0; i < cache_dim; ++i) delete[] cache[i];
    delete[] cache;

    return p;
  }
};

int main() {
  double price;
  double strike;
  double tte;
  double vol;
  double r;
  cout << "Enter: Current Price, Strike, Time To Expiry, Vol, Risk-Free Rate" << endl;
  cin >> price >> strike >> tte >> vol >> r;

  lattice_model_serial test(price, strike, tte, vol, r);
  while (true) {
    cout << "Enter timesteps: ";
    int t;
    cin >> t;
    cout << test.price(t) << endl;
  }
}
