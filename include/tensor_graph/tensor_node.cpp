#include "tensor_node.hpp"


namespace cast {




xt::xarray<double> TensorNode::data() const noexcept {
    return data_;
}


xt::xarray<double> TensorNode::gradients() const noexcept {
    return gradients_;
}




}