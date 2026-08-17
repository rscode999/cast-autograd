#include <initializer_list>
#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/network.hpp"

using namespace std;
using namespace xt;
using namespace cast;


int main() {
    Network net;
    std::shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    /*
    This is a very complicated XOR classifier.
    l1 2-4 > branch (1)  0 > l1 4-1                                                 > combiner (9)  
                         1 > branch (3)  1 > l1 4-1 (4) > sigmoid (6) > combiner (8) ^    
                                         2 > l1 4-1 (5) > sigmoid (7) ^
    */

    net.add_operator(make_shared<Linear1d>(2, 4));

    net.add_operator(make_shared<Splitter>(2));
    net.add_operator(make_shared<Linear1d>(4, 1));
    net.add_operator(make_shared<Splitter>(2), 1);
    net.add_operator(make_shared<Linear1d>(4, 1), 1);
    
    net.add_operator(make_shared<Linear1d>(4, 1), 2);
    net.add_operator(make_shared<Sigmoid>(), 1);
    net.add_operator(make_shared<Sigmoid>(), 2);
    net.add_combiner(std::shared_ptr<Combiner>(new Combiner{2}), 1);

    net.add_combiner(std::shared_ptr<Combiner>(new Combiner{1}), 0);
  
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

    
    for(int e = 0; e < 100; e++) {

        double loss = 0;
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            loss += loss_calc->compute(prediction, expected_outputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }

        if(e % 50 == 0) {
            std::cout << "Loss: " << loss << std::endl;
        }
    }

    for(int i = 0; i < (int)inputs.size(); i++) {
        xt::xarray<double> prediction = net.forward(inputs[i]);
        cout << "Prediction for " << inputs[i] << ": " << prediction << endl;
    }
}