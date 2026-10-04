#include "../include/cast.hpp"

#include <xtensor/io/xio.hpp>

#include <memory>


using namespace std;
using namespace xt;
using namespace cast;



int main() {
    xt::xarray<double> logits = {
    {1.0, 2.0, 3.0, 4.0}
    };

    xt::xarray<double> target = {
        {0.0, 0.0, 0.0, 1.0}
    };

    Softmax s;
    xt::xarray<double> out = s.forward(logits);
    std::cout << out << endl;

    CrossEntropy ce;

    std::cout << ce.compute(out, target) << std::endl;
    std::cout << ce.compute_gradient(out, target) << std::endl;

    auto ce_grad = ce.compute_gradient(out, target);
    auto logits_grad = s.backward(ce_grad);

    std::cout << "CE grad:\n" << ce_grad << "\n";
    std::cout << "logits grad:\n" << logits_grad << "\n";
}