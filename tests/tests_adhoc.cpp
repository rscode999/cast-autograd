#include "../include/cast.hpp"

#include <xtensor/io/xio.hpp>

#include <memory>


using namespace std;
using namespace xt;
using namespace cast;



int main() {
    Network net;

    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_splitter(2);
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 4));
    net.add_operator(make_shared<Linear1d>(4, 4), 1);
    net.add_combiner({1});

    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.05, 0.9));

    net.enable();

    xt::xarray<double> in = {1, 2};

    xt::xarray<double> out = net.forward(in);
    cout << "FORWARD RESULT: " << out << endl;
    net.backward(out, xt::xarray<double> {0,0,0,0});
}