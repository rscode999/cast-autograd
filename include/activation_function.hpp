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
    xt::xarray<double> forward(const xt::xarray<double>& input) override {
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
    xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients) override {
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
    xt::xarray<double> forward(const xt::xarray<double>& input) override {
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
    xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients in ReLU backwards pass must be non-empty");

        xt::xarray<double> output = xt::where(upstream_gradients >= 0.0, 1.0, 0.0);
        return output;
    }
};



/**
* Computes a probability distribution from its input. *For 1d input vectors only.*
*
* Has an adjustable temperature coefficient.
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
    * @param input list of values to compute. Non-empty, and with exactly 2 axes
    * @return Softmax(x) for each element of `input`
    */
    xt::xarray<double> forward(const xt::xarray<double>& input) override {
        str_assert(input.size() > 0, "Input cannot be empty");
        str_assert(input.dimension() == 2, "Softmax must have exactly 2 axes");

        prev_outputs_ = input;

        // Temperature scaling
        xt::xarray<double> scaled_input = xt::eval(input / temp_coeff_);

        // Maximum for each sample: [batch, 1]
        xt::xarray<double> max_vals =
            xt::eval(xt::amax(scaled_input, {1}, xt::keep_dims));

        // [batch, features] - [batch, 1]
        xt::xarray<double> shifted =
            xt::eval(scaled_input - max_vals);

        // [batch, features]
        xt::xarray<double> exp_vals =
            xt::eval(xt::exp(shifted));

        // Sum for each sample: [batch, 1]
        xt::xarray<double> sum_exp =
            xt::eval(xt::sum(exp_vals, {1}, xt::keep_dims));

        // [batch, features]
        xt::xarray<double> output =
            xt::eval(exp_vals / sum_exp);

        return output;
    }



    /**
    * Returns the derivative of Softmax applied to each parameter of `upstream_gradients`.
    * Axis 0 of `input` divides batches, where each index of axis 0 is a single vector.
    * @param upstream_gradients list of values to compute. Non-empty, with exactly 2 axes
    * @return d(Softmax(x))/dx for each element x of `upstream_gradients`
    */
    xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients) override {
        str_assert(upstream_gradients.size() > 0, "Upstream gradients cannot be empty");
        str_assert(upstream_gradients.dimension() == 2,"Upstream gradients must have exactly 2 axes");
        str_assert(prev_outputs_.dimension() == 2, "Previous outputs must have exactly 2 axes");
        str_assert(prev_outputs_.shape() == upstream_gradients.shape(), "Upstream gradients must have the same shape as the input");

        xt::xarray<double> scaled = xt::eval(prev_outputs_ / temp_coeff_);

        xt::xarray<double> max_vals = xt::eval(xt::amax(scaled, {1}, xt::keep_dims));

        xt::xarray<double> shifted = xt::eval(scaled - max_vals);
        xt::xarray<double> exp_vals = xt::eval(xt::exp(shifted));

        xt::xarray<double> sum_exp = xt::eval(xt::sum(exp_vals, {1}, xt::keep_dims));

        xt::xarray<double> S = xt::eval(exp_vals / sum_exp);

        xt::xarray<double> weighted_grad = xt::eval(upstream_gradients * S);
        xt::xarray<double> sum_grad_s = xt::eval(xt::sum(weighted_grad, {1}, xt::keep_dims));
        xt::xarray<double> result = xt::eval(S * (upstream_gradients - sum_grad_s) / temp_coeff_);

        return result;
    }

};




}
#endif 
