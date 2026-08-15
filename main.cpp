#include <initializer_list>
#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/network.hpp"

using namespace std;
using namespace xt;
using namespace cast;


int main() {
    shared_ptr<Combiner> c = make_shared<Combiner>(initializer_list<int32_t>{0});

    std::vector<std::vector<xt::xarray<double>>> in;
    for (int i = 0; i < 3; i++) {
        in.push_back(std::vector<xt::xarray<double>>());


        in[i].push_back({1, 2, 3});
        in[i].push_back({1, 2});
    }

    std::vector<std::vector<xt::xarray<double>>> out = c->compute(in);

    for(xt::xarray<double> out_elem : out[0]) {
        cout << out_elem << endl;
    }
}