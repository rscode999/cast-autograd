#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/loss_calculator.hpp"
#include "include/network.hpp"

int main() {
  using namespace std;
  using namespace xt;
  using namespace cast;

  Network net = Network();

  //(2,4) -> (4,1)
  net.add_operator(make_shared<Linear1d>(2, 4));
  net.add_operator(make_shared<Linear1d>(4, 1));

  shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
  net.set_loss_calculator(loss_calc);
  net.set_optimizer(make_shared<SGD>(0.02, 0.9));
  
  net.enable();


  vector<xarray<double>> inputs = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
  };

  vector<xarray<double>> expected_outputs = {
    {0},
    {1},
    {1},
    {0}
  };


  for(int32_t e = 0; e < 1000; e++) {

    for(int32_t i = 0; i < inputs.size(); i++) {
      xarray<double> predicted = net.forward(inputs[i]);
      net.backward(predicted, expected_outputs[i]);
      net.optimize();
    }

    
  }
}

