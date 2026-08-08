#include "tensor.hpp"


namespace cast {




const xt::xarray<double>& Tensor::data() const {
    return data_;
}

xt::xarray<double>& Tensor::data() {
    return data_;
}



}