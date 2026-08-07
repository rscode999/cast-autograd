#include "tensor_operator.hpp"

#include "tensor.hpp"

#include <memory>
#include <iostream>

namespace cast {





int32_t TensorOperator::n_inputs() const {
    return n_inputs_;
}


int32_t TensorOperator::n_outputs() const {
    return n_outputs_;
}



std::vector<std::shared_ptr<Tensor>> TensorOperator::compute_and_link(std::vector<std::shared_ptr<Tensor>> inputs) {
    predecessors_.clear();

    std::vector<xt::xarray<double>> input_tensors = {};

    for(std::shared_ptr<Tensor> node_ptr : inputs) {
        //Register the input node as a predecessor
        predecessors_.push_back(node_ptr);

        //Collect the input
        input_tensors.push_back(node_ptr->data());
    }
    
    //Carry out the operation
    std::vector<xt::xarray<double>> outputs = compute(input_tensors);


    std::vector<std::shared_ptr<Tensor>> output_node_ptrs = {};
    for(xt::xarray<double> output_tensor : outputs) {

        //Wrap operation's output in tensor node
        Tensor output_node = Tensor(output_tensor);

        //Wrap operation output tensor node in a std::shared_ptr
        std::shared_ptr<Tensor> output_node_ptr = std::make_shared<Tensor>(std::move(output_node));

        //Register this operation object as the new output's precedessor
        output_node_ptr->prev_operator_ = shared_from_this();

        output_node_ptrs.push_back(output_node_ptr);
    }

    // std::cout << "this node's preds" << std::endl;
    // for(std::shared_ptr<Tensor> n_ptr : predecessors_) {
    //     std::cout << n_ptr->data() << std::endl;
    // }
    // std::cout << "output tensor node" << std::endl;
    // for(std::shared_ptr<Tensor> n_ptr : output_node_ptrs) {
    //     std::cout << n_ptr->prev_operator_->name() << std::endl;
    // }

    return output_node_ptrs;
}



std::string TensorOperator::name() const {
    return "TensorOperator";
}




}