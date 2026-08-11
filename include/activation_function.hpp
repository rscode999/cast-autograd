#ifndef CAST_ACTIVATION_FUNCTION_
#define CAST_ACTIVATION_FUNCTION_

#include "tensor_operator.hpp"


namespace cast {




/**
* Computes an element-wise function and its derivative across tensors.
*
* Given a std::vector of parameters, each of type xt::xarray<double>, the function is computed
* for each element of each parameter.
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
    std::vector<xt::xarray<double>> prev_outputs_;

public:


    /**
    * @return the string "sigmoid"
    */
    std::string name() const override {
        return "sigmoid";
    }

    /**
    * Returns the Sigmoid activation function applied to each parameter in `inputs`
    * @param inputs list of values to compute
    * @return sigmoid(x) for each element of `inputs`
    */
    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> inputs) override {

        std::vector<xt::xarray<double>> output = {};
        for(xt::xarray<double> params : inputs) {
            output.push_back(1 / (1 + exp(-params)) );
        }

        prev_outputs_ = output;
        return output;
    }

    /**
    * Returns the derivative of Sigmoid applied to each parameter of `upstream_gradients`.
    * @param upstream_gradients list of values to compute
    * @return d(Sigmoid(x))/dx for each element x of `inputs`
    */
    std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> upstream_gradients) override {
        std::vector<xt::xarray<double>> output;
        output.reserve(upstream_gradients.size());

        // sigmoid(x) * (1.0 - sigmoid(x));
        for(int32_t i = 0; i < (int32_t)upstream_gradients.size(); i++) {
            output.push_back(upstream_gradients[i] * prev_outputs_[i] * (1 - prev_outputs_[i]));
        }
        return output;
    }
};




}
#endif 
