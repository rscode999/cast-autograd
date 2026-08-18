#ifndef CAST_ACTIVATION_FUNCTION_
#define CAST_ACTIVATION_FUNCTION_

#include "cast_exceptions.hpp"
#include "network_component.hpp"
#include "operator.hpp"


namespace cast {




/**
* Computes an element-wise function and its derivative across tensors.
*
* Given a std::vector of parameters, each of type xt::xarray<double>, the function is computed
* for each element of each parameter.
*/
class ActivationFunction : public Operator {
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
    * @return deep pointer copy of this Sigmoid object
    */
    std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const override {
        return std::make_shared<Sigmoid>(*this);
    }


    /**
    * @return the string "sigmoid"
    */
    std::string name() const override {
        return "sigmoid";
    }


    
    /**
    * Returns the Sigmoid activation function applied to each parameter in `inputs`
    * @param inputs list of values to compute. Non-empty
    * @return sigmoid(x) for each element of `inputs`
    */
    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> inputs) override {
        str_assert(inputs.size() > 0, "Input vector must be non-empty");

        std::vector<xt::xarray<double>> output = {};
        for(xt::xarray<double> params : inputs) {
            output.push_back(1 / (1 + exp(-params)) );
        }

        prev_outputs_ = output;
        return output;
    }


    
    /**
    * Returns the derivative of Sigmoid applied to each parameter of `upstream_gradients`.
    * YOU MUST HAVE PREVIOUSLY USED THIS OBJECT'S `compute` METHOD TO GET A RESULT.
    * @param upstream_gradients list of values to compute. Non-empty
    * @return d(Sigmoid(x))/dx for each element x of `inputs`
    */
    std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients in Sigmoid backwards pass must be non-empty");
        str_assert(prev_outputs_.size() == upstream_gradients.size(), "The forward-pass Sigmoid function must have been previously computed on an input of the same length as `upstream_gradients`");

        std::vector<xt::xarray<double>> output;
        output.reserve(upstream_gradients.size());

        // sigmoid(x) * (1.0 - sigmoid(x));
        for(int32_t i = 0; i < (int32_t)upstream_gradients.size(); i++) {
            str_assert(prev_outputs_[i].shape() == upstream_gradients[i].shape(), "Upstream gradient element " + std::to_string(i) + " shape does not match the previous input's shape");
            output.push_back(upstream_gradients[i] * prev_outputs_[i] * (1 - prev_outputs_[i]));
        }

        //Clear the previous outputs
        prev_outputs_.clear();
        
        return output;
    }
};




}
#endif 
