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


void speed_test() {
    Network* net = new Network();
    net->add_operator(make_shared<Linear1d>(2, 4));
    net->add_operator(make_shared<Linear1d>(4, 8));
    net->add_operator(make_shared<Linear1d>(8, 1));
    net->set_loss_calculator(make_shared<MeanSquaredError>());

    std::vector<int32_t> order = net->compile();
    for(int32_t c : order) {
        cout << c << endl;
    }
    cout << endl;
    delete net;

    net = new Network();
    net->add_operator(make_shared<Linear1d>(2, 4));
    net->add_splitter(3);
    net->add_operator(make_shared<Linear1d>(4, 4), 0);
    net->add_operator(make_shared<Linear1d>(4, 4), 1);
    net->add_operator(make_shared<Linear1d>(4, 4), 2);
    net->add_combiner({0, 1}, 2);
    net->set_loss_calculator(make_shared<MeanSquaredError>());

    order = net->compile();
    for(int32_t c : order) {
        cout << c << endl;
    }
    cout << endl;
    delete net;



    net = new Network();
    net->add_operator(make_shared<Linear1d>(2, 4));
    net->set_loss_calculator(make_shared<MeanSquaredError>());

    order = net->compile();
    for(int32_t c : order) {
        cout << c << endl;
    }
    cout << endl;
    delete net;


    net = new Network();
    net->set_loss_calculator(make_shared<MeanSquaredError>());
    order = net->compile();
    for(int32_t c : order) {
        cout << c << endl;
    }
    cout << endl;
    delete net;
}


int main() {
    speed_test();
}