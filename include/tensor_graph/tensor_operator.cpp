#include "tensor_operator.hpp"

#include <memory>
#include <iostream>

namespace cast {





int32_t TensorOperator::n_inputs() const {
    return n_inputs_;
}


int32_t TensorOperator::n_outputs() const {
    return n_outputs_;
}


std::string TensorOperator::name() const {
    return "TensorOperator";
}




}