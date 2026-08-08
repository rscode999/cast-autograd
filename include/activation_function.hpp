#ifndef CAST_ACTIVATION_FUNCTION_
#define CAST_ACTIVATION_FUNCTION_

#include "tensor_graph/tensor_graph.hpp"


namespace cast {




/**
* Computes an element-wise function and its derivative across tensors
*/
class ActivationFunction : public TensorOperator {
public:

    /**
    * Creates a new activation function
    */
    ActivationFunction() {
    }

    /**
    * @return the string "activation_function"
    */
    virtual std::string name() const override {
        return "activation_function";
    }
};



/**
* Sigmoid activation function
*/
class Sigmoid : public ActivationFunction {
private:
    /**
    * Outputs from the last Sigmoid computation.
    *
    * Makes calculation of the backwards pass easier.
    */
    std::vector<Tensor> prev_outputs_;

public:


    /**
    * @return the string "sigmoid"
    */
    std::string name() const override {
        return "sigmoid";
    }

    /**
    * Computes the Sigmoid activation function on each parameter in `inputs`
    */
    std::vector<Tensor> compute(std::vector<Tensor> inputs) override {

        std::vector<Tensor> output = {};
        for(Tensor params : inputs) {
            output.push_back( Tensor(1 / (1 + exp(-params.data()))) );
        }

        prev_outputs_ = output;
        return output;
    }

    /**
    * Computes the gradient of Sigmoid for each parameter of `upstream_gradients`
    */
    std::vector<Tensor> compute_backwards_pass(std::vector<Tensor> upstream_gradients) override {
        std::vector<Tensor> output;
        output.reserve(upstream_gradients.size());

        // sigmoid(x) * (1.0 - sigmoid(x));
        for(int32_t i = 0; i < (int32_t)upstream_gradients.size(); i++) {
            output.push_back(Tensor(upstream_gradients[i].data() * prev_outputs_[i].data() * (1 - prev_outputs_[i].data())));
        }
        return output;
    }
};




}
#endif 
