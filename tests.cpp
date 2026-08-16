#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/loss_calculator.hpp"
#include "include/network.hpp"

using namespace std;
using namespace xt;
using namespace cast;



void test_xor() {
  Network net = Network();

  //(2,4) -> (4,1)
  net.add_operator(make_shared<Linear1d>(2, 4));
  net.add_operator(make_shared<Sigmoid>());
  net.add_operator(make_shared<Linear1d>(4, 1));

  shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
  net.set_loss_calculator(loss_calc);
  net.set_optimizer(make_shared<SGD>(0.02, 0.9));
  
  net.enable();


  vector<xarray<double>> inputs = {
      xarray<double>{0, 0},
      xarray<double>{0, 1},
      xarray<double>{1, 0},
      xarray<double>{1, 1}
  };

  vector<xarray<double>> expected_outputs = {
      xarray<double>{0},
      xarray<double>{1},
      xarray<double>{1},
      xarray<double>{0}
  };

  
  for(int e = 0; e < 1000; e++) {

      double loss = 0;
      for(int i = 0; i < (int)inputs.size(); i++) {
          xt::xarray<double> prediction = net.forward(inputs[i]);
          loss += loss_calc->compute(prediction, expected_outputs[i]);
          net.backward(prediction, expected_outputs[i]);
          net.optimize();
      }

      if(e % 100 == 0) {
          std::cout << "Loss: " << loss << std::endl;
      }
  }

  for(int i = 0; i < (int)inputs.size(); i++) {
      xt::xarray<double> prediction = net.forward(inputs[i]);
      cout << "Prediction for " << inputs[i] << ": " << prediction << endl;
  }
}



void test_branches() {
    /*
    architecture:
    l1 > sigmoid > branch 0  > l1 (3) > branch (5)     > sigmoid (6)
                          1  > l1 (4)  > l1 (8)      2 > sigmoid (7)
    */

    Network net;

    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    net.add_operator(make_shared<Linear1d>(2, 3));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Branch>(2));
    net.add_operator(make_shared<Linear1d>(3, 4));

    net.add_operator(make_shared<Linear1d>(3, 4), 1);
    net.add_operator(make_shared<Branch>(2));
    net.add_operator(make_shared<Sigmoid>(), 0);
    net.add_operator(make_shared<Sigmoid>(), 2);
    
    net.add_operator(make_shared<Linear1d>(3, 4), 1);

    net.enable();
}



void test_combiners_simple() {
    /*
    architecture:
    l1 > sigmoid > branch 0  > l1 (3) > combiner 0,1 (5)
                          1  > l1 (4) ^
    */

    Network net;

    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    net.add_operator(make_shared<Linear1d>(2, 3));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Branch>(2));
    net.add_operator(make_shared<Linear1d>(3, 4));

    net.add_operator(make_shared<Linear1d>(3, 4), 1);
    net.add_operator(shared_ptr<Combiner>(new Combiner{1}));

    net.enable();
}




void test_combiners_compound() {
    /*
    architecture:
    l1 > branch (1)  0 > sigmoid (2) >  combiner 0,1 (6)
                     1 > l1 (3)      > combiner 1,2  (5)
                     2 > sigmoid (4) ^
    */

    Network net;

    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    net.add_operator(make_shared<Linear1d>(2, 3));

    net.add_operator(make_shared<Branch>(3));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(2, 3), 1);
    net.add_operator(make_shared<Sigmoid>(), 2);

    net.add_operator(shared_ptr<Combiner>(new Combiner{2}), 1);
    net.add_operator(shared_ptr<Combiner>(new Combiner{1}));

    net.enable();
}



void test_combiners_other_branches() {
        /*
    architecture:
    l1 > branch (1)  0 > sigmoid (2) \
                     1 > l1 (3)      > combiner 1 (5) > combiner 2 (7)
                     2 > sigmoid (4) > l1 (6)         ^
    */

    Network net;

    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    net.add_operator(make_shared<Linear1d>(2, 3));

    net.add_operator(make_shared<Branch>(3));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(2, 3), 1);
    net.add_operator(make_shared<Sigmoid>(), 2);

    net.add_operator(shared_ptr<Combiner>(new Combiner{0}), 1);
    net.add_operator(make_shared<Linear1d>(2, 3), 2);
    net.add_operator(shared_ptr<Combiner>(new Combiner{2}), 1);

    net.enable();
}



void test_branch_forward() {
    Network net;

    /*
    l1 2-3 > branch (1)  0 > sigmoid (2) > combiner (5)
                         1 > sigmoid (3) ^
                         2 > sigmoid (4) ^
    */
    
    net.set_optimizer(make_shared<SGD>(0.9, 0.02));
    net.set_loss_calculator(make_shared<MeanSquaredError>());

    net.add_operator(make_shared<Linear1d>(2, 3));

    net.add_operator(make_shared<Branch>(3));
    net.add_operator(make_shared<Sigmoid>(), 0);
    net.add_operator(make_shared<Sigmoid>(), 1);
    net.add_operator(make_shared<Sigmoid>(), 2);
    net.add_operator(make_shared<Combiner>(initializer_list<int32_t>{1, 2}));

    net.enable();

    xt::xarray<double> out = net.forward({1, 2});
    cout << "OUTPUT: " << out << endl;
}


int main() {
    test_xor();
}