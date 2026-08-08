#include "tensor.hpp"


namespace cast {




xt::xarray<double> Tensor::data_deepcopy() const {
    return data_;
}

xt::xarray<double>& Tensor::data() {
    return data_;
}



void Tensor::set_data(xt::xarray<double> new_data) {
    data_ = new_data;
}


}