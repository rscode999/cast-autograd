#ifndef CAST_ACTIVATION_FUNCTION_
#define CAST_ACTIVATION_FUNCTION_

#include <xtensor/core/xnoalias.hpp>

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

        xt::xarray<double> output = input;

        output *= -1.0;
        xt::noalias(output) = xt::exp(output);
        output += 1.0;
        xt::noalias(output) = 1.0 / output;

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

        // sigmoid(x) * (1.0 - sigmoid(x));
        return (upstream_gradients * prev_outputs_ * (1 - prev_outputs_));
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

        return xt::where(upstream_gradients >= 0.0, 1.0, 0.0);
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
        str_assert(temp_coeff_ > 0.0, "Temperature must be positive");

        const std::size_t batch = input.shape()[0];
        const std::size_t features = input.shape()[1];

        xt::xarray<double> output = xt::xarray<double>::from_shape(input.shape());

        const double* x = input.data();
        double* p = output.data();
        const double inv_temp = 1.0 / temp_coeff_;

        for (std::size_t b = 0; b < batch; ++b) {
            const std::size_t offset = b * features;

            // Find maximum logit for numerical stability.
            double max_val = x[offset] * inv_temp;

            for (std::size_t j = 1; j < features; ++j) {
                const double value = x[offset + j] * inv_temp;
                if (value > max_val) {
                    max_val = value;
                }
            }

            // Exponentiate and sum.
            double sum_exp = 0.0;
            for (std::size_t j = 0; j < features; ++j) {
                const double value = std::exp(x[offset + j] * inv_temp - max_val);

                p[offset + j] = value;
                sum_exp += value;
            }

            // Normalize.
            const double inv_sum = 1.0 / sum_exp;

            for (std::size_t j = 0; j < features; ++j) {
                p[offset + j] *= inv_sum;
            }
        }

        // Save probabilities for backward propagation.
        prev_outputs_ = output;

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

        const std::size_t batch = upstream_gradients.shape()[0];
        const std::size_t features = upstream_gradients.shape()[1];

        xt::xarray<double> result =
            xt::xarray<double>::from_shape(upstream_gradients.shape());

        const double* p = prev_outputs_.data();
        const double* g = upstream_gradients.data();
        double* dx = result.data();

        const double inv_temp = 1.0 / temp_coeff_;

        for (std::size_t b = 0; b < batch; ++b) {
            const std::size_t offset = b * features;

            // sum_j(g_j * p_j)
            double weighted_sum = 0.0;

            for (std::size_t j = 0; j < features; ++j) {
                weighted_sum +=
                    g[offset + j] * p[offset + j];
            }

            // dL/dx_i = p_i * (g_i - weighted_sum) / T
            for (std::size_t j = 0; j < features; ++j) {
                dx[offset + j] =
                    p[offset + j]
                    * (g[offset + j] - weighted_sum)
                    * inv_temp;
            }
        }

        // Clear saved probabilities only after their final use.
        prev_outputs_ = xt::xarray<double>{};

        return result;
    }

};




}
#endif 
