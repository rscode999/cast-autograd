#include <cstdint>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <xtensor/io/xio.hpp>

#include "include/cast.hpp"
#include "include/loss_calculator.hpp"

using namespace std;
using namespace xt;
using namespace cast;



void train_no_batch() {
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

    vector<xarray<double>> expected_outputs;
    expected_outputs.push_back(xt::xarray<double>({1}, 0.0));
    expected_outputs.push_back(xt::xarray<double>({1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1}, 0.0));


    //Train
    for(int e = 0; e < 1000; e++) {
        double loss = 0;
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            loss += loss_calc->compute(prediction, expected_outputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }

        if(e % 100 == 0) {
            cout << "loss: " << loss << endl;
        }
    }
}


void train_batch_1() {
    Network net = Network();

    //(2,4) -> (4,1)
    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));
    
    net.enable();

    net.set_batch_size(1);

    vector<xarray<double>> inputs = {
        xarray<double>{{0, 0}},
        xarray<double>{{0, 1}},
        xarray<double>{{1, 0}},
        xarray<double>{{1, 1}}
    };

    vector<xarray<double>> expected_outputs;
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 0.0));
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 1.0));
    expected_outputs.push_back(xt::xarray<double>({1, 1}, 0.0));


    //Train
    for(int e = 0; e < 1000; e++) {
        double loss = 0;
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            loss += loss_calc->compute(prediction, expected_outputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }

        if(e % 100 == 0) {
            cout << "loss: " << loss << endl;
        }
    }
}



void train_batch_2() {
    Network net = Network();

    //(2,4) -> (4,1)
    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));
    
    net.enable();

    net.set_batch_size(2);

    vector<xarray<double>> inputs = {
        xarray<double>{{0, 0}, {0,1}},
        xarray<double>{{1, 0}, {1, 1}},
    };

    vector<xarray<double>> expected_outputs;
    xarray<double> first = {0.0, 1.0};
    expected_outputs.push_back(first.reshape({2, 1}));
    xarray<double> second = {1.0, 0.0};
    expected_outputs.push_back(second.reshape({2, 1}));

    //Train
    for(int e = 0; e < 1000; e++) {
        double loss = 0;
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            loss += loss_calc->compute(prediction, expected_outputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }

        if(e % 100 == 0) {
            cout << "loss: " << loss << endl;
        }
    }
}


int main() {
    // train_no_batch();
    train_batch_2();
}