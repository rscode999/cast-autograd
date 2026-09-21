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
    xt::xarray<double> prev_outputs_;

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
    * Returns the Sigmoid activation function applied to each parameter in `input`.
    *
    * sigmoid(x) = 1 / (1 + exp(-x)) for a scalar value x.
    * @param input list of values to compute. Non-empty
    * @return sigmoid(x) for each element of `inputs`
    */
    xt::xarray<double> forward(xt::xarray<double> input) override {
        str_assert(input.size() > 0, "Input vector must be non-empty");

        xt::xarray<double> output = (1 / (1 + exp(-input)) );

        prev_outputs_ = output;
        return output;
    }


    
    /**
    * Returns the derivative of Sigmoid applied to each parameter of `upstream_gradients`.
    * YOU MUST HAVE PREVIOUSLY USED THIS OBJECT'S `compute` METHOD TO GET A RESULT.
    * @param upstream_gradients list of values to compute. Non-empty
    * @return d(Sigmoid(x))/dx for each element x of `upstream_gradients`
    */
    xt::xarray<double> backward(xt::xarray<double> upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients in Sigmoid backwards pass must be non-empty");
        str_assert(prev_outputs_.shape() == upstream_gradients.shape(), "The forward-pass Sigmoid function must have been previously computed on an input of the same length as `upstream_gradients`");

        xt::xarray<double> output;

        // sigmoid(x) * (1.0 - sigmoid(x));
        output = (upstream_gradients * prev_outputs_ * (1 - prev_outputs_));

        //Clear the previous outputs
        prev_outputs_ = xt::xarray<double>{};
        
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
    * Returns the ReLU activation function applied to each parameter in `input`
    *
    * ReLU(x) = max(0, x) for a scalar x
    * @param input list of values to compute. Non-empty
    * @return ReLU(x) for each element of `input`
    */
    xt::xarray<double> forward(xt::xarray<double> input) override {
        str_assert(input.size() > 0, "Input vector must be non-empty");
        return xt::maximum(input, 0.0);
    }


    /**
    * Returns the derivative of ReLU applied to each parameter of `upstream_gradients`.
    *
    * d(ReLU(x))/dx = 0 if x is negative, otherwise 1
    * @param upstream_gradients list of values to compute. Non-empty
    * @return d(ReLU(x))/dx for each element x of `upstream_gradients`
    */
    xt::xarray<double> backward(xt::xarray<double> upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients in ReLU backwards pass must be non-empty");

        xt::xarray<double> output = xt::where(upstream_gradients >= 0.0, 1.0, 0.0);
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
    xt::xarray<double> prev_outputs_;

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
    * Returns the Softmax function applied to each element in `input`. Each element has the Softmax function applied to it.
    * Axis 0 of `input` divides batches, where each index of axis 0 is a single vector.
    * @param input list of values to compute. Non-empty, and with more than 1 axis
    * @return Softmax(x) for each element of `input`
    */
    xt::xarray<double> forward(xt::xarray<double> input) override {
        str_assert(input.size() > 0, "Input cannot be empty");
        str_assert(input.shape().size() > 1, "Input must have multiple axes (axis 0 is for batch size only)");

        // Store the last inputs of the calculation
        prev_outputs_ = input;

        // Apply the temperature coefficient scaling
        xt::xarray<double> scaled_input = input / temp_coeff_;

        // Subtract the maximum along axis 1 (features) for numerical stability, 
        // keeping dimensions intact for proper broadcasting across the batch axis (axis 0).
        auto max_vals = xt::amax(scaled_input, {1}, xt::keep_dims);
        auto shifted = scaled_input - max_vals;

        // Compute the exponential of the shifted values
        auto exp_vals = xt::exp(shifted);

        // Compute the sum of exponentials along axis 1, keeping dimensions
        auto sum_exp = xt::sum(exp_vals, {1}, xt::keep_dims);

        // Divide exponentiated values by the sum to get the softmax probabilities
        return exp_vals / sum_exp;
    }


    /**
    * Returns the derivative of Softmax applied to each parameter of `upstream_gradients`.
    * Axis 0 of `input` divides batches, where each index of axis 0 is a single vector.
    * @param upstream_gradients list of values to compute. Non-empty, with more than 1 axis
    * @return d(Softmax(x))/dx for each element x of `upstream_gradients`
    */
    xt::xarray<double> backward(xt::xarray<double> upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients cannot be empty");
        str_assert(upstream_gradients.shape().size() > 1, "Input must have multiple axes (axis 0 is for batch size only)");

        // Recompute the forward softmax output (S) using prev_outputs_ and temp_coeff_
        auto scaled = prev_outputs_ / temp_coeff_;
        auto max_vals = xt::amax(scaled, {1}, xt::keep_dims);
        auto exp_vals = xt::exp(scaled - max_vals);
        auto S = exp_vals / xt::sum(exp_vals, {1}, xt::keep_dims);

        // Compute the dot product term: sum(upstream_gradients * S) along axis 1
        auto sum_grad_s = xt::sum(upstream_gradients * S, {1}, xt::keep_dims);

        // Apply the softmax Jacobian-vector product formula scaled by the temperature:
        // dz = S * (upstream_gradients - sum_grad_s) / temp_coeff_
        return S * (upstream_gradients - sum_grad_s) / temp_coeff_;
    }
};




}
#endif 
