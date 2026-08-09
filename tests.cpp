#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/loss_calculator.hpp"
#include "include/network.hpp"

void test_xor() {
  // using namespace std;
  // using namespace xt;
  // using namespace cast;

  // Network net = Network();

  // //(2,4) -> (4,1)
  // net.add_operator(make_shared<Linear1d>(2, 4));
  // net.add_operator(make_shared<Sigmoid>());
  // net.add_operator(make_shared<Linear1d>(4, 1));

  // shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
  // net.set_loss_calculator(loss_calc);
  // net.set_optimizer(make_shared<SGD>(0.02, 0.9));
  
  // net.enable();


  // vector<Tensor> inputs = {
  //   Tensor({0, 0}),
  //   Tensor({0, 1}),
  //   Tensor({1, 0}),
  //   Tensor({1, 1})
  // };

  // vector<Tensor> expected_outputs = {
  //   Tensor(xarray<double>{0}),
  //   Tensor(xarray<double>{1}),
  //   Tensor(xarray<double>{1}),
  //   Tensor(xarray<double>{0})
  // };


  // for(int32_t e = 0; e < 1000; e++) {

  //   double total_loss = 0;

  //   for(int32_t i = 0; i < inputs.size(); i++) {
  //     Tensor predicted = net.forward(inputs[i]);
  //     total_loss += loss_calc->compute(predicted, expected_outputs[i]);
  //     net.backward(predicted, expected_outputs[i]);
  //     net.optimize();
  //   }

  //   if(e % 100 == 0) {
  //     cout << "Total loss: " << total_loss << endl;
  //   }
  // }

  // for(int32_t i = 0; i < inputs.size(); i++) {
  //   cout << "Predictions for " << inputs[i] << ": " << net.forward(inputs[i]) << endl;
  // }
}



int main() {
    test_xor();
}