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
    * Creates a new Sigmoid activation function
    */
    Sigmoid() {
    }

    /**
    * @return deep pointer copy of this Sigmoid object
    */
    std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const override {
        return std::make_shared<Sigmoid>(*this);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * @return the string "sigmoid"
    */
    std::string to_string() const override {
        return "sigmoid";
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
    * Returns the Sigmoid activation function applied to each parameter in `inputs`.
    *
    * sigmoid(x) = 1 / (1 + exp(-x)) for a scalar value x.
    * @param inputs list of values to compute. Non-empty
    * @return sigmoid(x) for each element of `inputs`
    */
    std::vector<xt::xarray<double>> forward(std::vector<xt::xarray<double>> inputs) override {
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
    * @return d(Sigmoid(x))/dx for each element x of `upstream_gradients`
    */
    std::vector<xt::xarray<double>> backward(std::vector<xt::xarray<double>> upstream_gradients) override {
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




/**
* Rectified Linear Unit (ReLU) function
*/
class ReLU : public ActivationFunction {
public:
    /**
    * Creates a new ReLU activation function
    */
    ReLU() {
    }

    /**
    * @return deep pointer copy of this Relu object
    */
    std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const override {
        return std::make_shared<ReLU>(*this);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * @return the string "relu"
    */
    std::string to_string() const override {
        return "relu";
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * Returns the ReLU activation function applied to each parameter in `inputs`
    *
    * ReLU(x) = max(0, x) for a scalar x
    * @param inputs list of values to compute. Non-empty
    * @return ReLU(x) for each element of `inputs`
    */
    std::vector<xt::xarray<double>> forward(std::vector<xt::xarray<double>> inputs) override {
        str_assert(inputs.size() > 0, "Input vector must be non-empty");

        std::vector<xt::xarray<double>> output = {};
        for(xt::xarray<double> params : inputs) {
            output.push_back(xt::maximum(params, 0.0));
        }

        return output;
    }


    /**
    * Returns the derivative of ReLU applied to each parameter of `upstream_gradients`.
    *
    * d(ReLU(x))/dx = 0 if x is negative, otherwise 1
    * @param upstream_gradients list of values to compute. Non-empty
    * @return d(ReLU(x))/dx for each element x of `upstream_gradients`
    */
    std::vector<xt::xarray<double>> backward(std::vector<xt::xarray<double>> upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients in ReLU backwards pass must be non-empty");

        std::vector<xt::xarray<double>> output;
        output.reserve(upstream_gradients.size());

        for(xt::xarray<double> grad : upstream_gradients) {
            output.push_back(xt::where(grad >= 0.0, 1.0, 0.0));
        }
        
        return output;
    }
};



/**
* Computes a probability distribution derived from its input
*/
class Softmax : public ActivationFunction {
private:
    /**
    * Outputs from the last time that this object computed values
    */
    std::vector<xt::xarray<double>> prev_outputs_;

    /**
    * Dictates differences in outputs- higher values cause less distinct outputs
    */
    double temp_coeff_;

public:

    /**
    * Creates a new Softmax object with temperature coefficient `temperature_coefficient`.
    * @param temperature_coefficient amplification or attenuation of differences between outputs. Positive.
    */
    Softmax(double temperature_coefficient = 1) : temp_coeff_(temperature_coefficient) {
        str_assert(temperature_coefficient > 0, "Temperature coefficient must be positive- got " + std::to_string(temperature_coefficient));
    }


    /**
    * @return deep pointer copy of this Softmax object
    */
    std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const override {
        return std::make_shared<Softmax>(*this);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////

    /**
    * @return the object's temperature coefficient
    */
    double temperature_coefficient() const {
        return temp_coeff_;
    }


    /**
    * @return the string "softmax"
    */
    std::string to_string() const override {
        return "softmax";
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////

    /**
    * Sets the Softmax's temperature coefficient to `new_temp_coeff`.
    * @param new_temp_coeff temperature coefficient to set. Positive.
    */
    void set_temperature_coefficient(double new_temp_coeff) {
        str_assert(new_temp_coeff > 0, "New temperature coefficient must be positive- got " + std::to_string(new_temp_coeff));
        temp_coeff_ = new_temp_coeff;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * Returns the Softmax function applied to each element in `inputs`. Each element has the Softmax function applied to it.
    * @param inputs list of values to compute. Non-empty
    * @return Softmax(x) for each element of `inputs`
    */
    std::vector<xt::xarray<double>> forward(std::vector<xt::xarray<double>> inputs) override {
        str_assert(inputs.size() > 0, "Inputs cannot be empty");

        prev_outputs_.clear();
        prev_outputs_.reserve(inputs.size());
        std::vector<xt::xarray<double>> outputs;
        outputs.reserve(inputs.size());

        for (const auto& x : inputs) {
            str_assert(inputs.size() > 0, "Each input must be non-empty");

            // Scale inputs by the temperature coefficient
            auto scaled_inputs = x / temp_coeff_;
            
            // Numerically stable softmax by subtracting the maximum value
            auto max_val = xt::amax(scaled_inputs);
            auto exp_inputs = xt::exp(scaled_inputs - max_val);
            auto softmax_out = exp_inputs / xt::sum(exp_inputs);
            
            outputs.push_back(softmax_out);
            prev_outputs_.push_back(softmax_out); // Cache for backward pass
        }
        return outputs;
    }


    /**
    * Returns the derivative of Softmax applied to each parameter of `upstream_gradients`.
    * @param upstream_gradients list of values to compute. Non-empty
    * @return d(Softmax(x))/dx for each element x of `upstream_gradients`
    */
    std::vector<xt::xarray<double>> backward(std::vector<xt::xarray<double>> upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients cannot be empty");

        std::vector<xt::xarray<double>> input_gradients;
        input_gradients.reserve(upstream_gradients.size());

        for (size_t i = 0; i < upstream_gradients.size(); ++i) {
            str_assert(upstream_gradients[i].size() > 0, "Each upstream gradient must be non-empty");

            const auto& g = upstream_gradients[i];
            const auto& y = prev_outputs_[i];

            // Softmax derivative with temperature scaling:
            // dL/dx = (1 / T) * y * (g - sum(g * y))
            auto sum_g_y = xt::sum(g * y);
            auto grad = (y * (g - sum_g_y)) / temp_coeff_;
            
            input_gradients.push_back(grad);
        }
        return input_gradients;
    }
};




}
#endif 
