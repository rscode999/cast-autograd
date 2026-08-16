#include <initializer_list>
#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/network.hpp"

using namespace std;
using namespace xt;
using namespace cast;


int main() {
    Network net;
    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    /*
    l1 > branch (1)  0 > sigmoid (2)                                  >  combiner (7)
                     1 > branch (3)   1 > sigmoid (4)  \\             
                                      2 > sigmoid(5)  > combiner (6)  ^
    */

    net.add_operator(make_shared<Linear1d>(2, 3));

    net.add_operator(make_shared<Branch>(2));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Branch>(2), 1);
    net.add_operator(make_shared<Sigmoid>(), 1);
    
    net.add_operator(make_shared<Sigmoid>(), 2);
    net.add_combiner(make_shared<Combiner>(initializer_list<int32_t>{1}), 2);
    net.add_combiner(make_shared<Combiner>(initializer_list<int32_t>{2}), 0);

    net.enable();

    // xt::xarray<double> out = net.forward({1, 2});
    // cout << "RESULT: " << out << endl;
}