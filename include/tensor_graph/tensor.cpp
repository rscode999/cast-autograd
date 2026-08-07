#include "tensor.hpp"


namespace cast {




xt::xarray<double> Tensor::data() const {
    return data_;
}


void Tensor::set_data(xt::xarray<double> new_data) {
    data_ = new_data;
}


}