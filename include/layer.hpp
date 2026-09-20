#ifndef CAST_LAYER_
#define CAST_LAYER_

#include "operator.hpp"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <source_location>
#include <string>

namespace cast {




/*
NOTICE TO IMPLEMENTERS:
The Layer object's parameters and gradients can be changed by outside users.
The forward or backwards pass must check the dimensions of the params and gradients before calculation.
*/

/**
* Layer in a network. Contains parameters and gradients for each parameter.
*
* Note that layers can accept multiple inputs and give multiple outputs.
*/
class Layer : public Operator {
protected:

    /**
     * Parameters of the layer. Each index has a specialized role (i.e. weight matrix, bias vector)
     */
    std::vector<xt::xarray<double>> parameters_;

    /**
     * Gradients of each tensor in the `parameters_` list.
     * Index `i` corresponds to the gradients of `parameters_[i]`.
     */
    std::vector<xt::xarray<double>> gradients_;

    /**
     * Tensors before this operation was applied
     */
    xt::xarray<double> prev_inputs_;

public:

    /**
     * @return gradients of the weights, biases, etc. of this layer
     */
    std::vector<xt::xarray<double>>& gradients() {
        return gradients_;
    }


    /**
     * @return parameters (weights, biases, ...) of this layer
     */
    std::vector<xt::xarray<double>>& parameters() {
        return parameters_;
    }
};





/**
 * Performs a fully-connected dense linear operation on a single 1d vector. Produces a single 1d vector.
 */
class Linear1d : public Layer {
private:
    
    /**
     * Required size of 1d input vectors
     */
    int32_t input_vector_dimension_;

    /**
     * Size of 1d vectors coming from the linear forward operation
     */
    int32_t output_vector_dimension_;

    /**
    * Asserts that the weights and biases of this layer are in the proper shape and dimension.
    * 
    * The layer must have 2 parameters.
    * Weights (index 0) must be a 2d matrix of shape (`output_vector_dimension_`, `input_vector_dimension_`).
    * Biases (index 1) must be a 1d vector of shape `output_vector_dimension_`.
    *
    * Does nothing if `NDEBUG` is defined.
    * @param loc location where this assertion was made
    */
    void assert_parameter_list_preconditions_(std::source_location loc = std::source_location::current()) {
        #ifndef NDEBUG

        str_assert(parameters_.size() == 2, "Linear1d layer must have two parameters", loc);

        xt::svector<std::size_t> weight_shape = parameters_[Weights].shape();
        str_assert(weight_shape.size() == 2, "Weights matrix must be 2d; weight's current rank is " + std::to_string(weight_shape.size()), loc);
        
        str_assert((int32_t)weight_shape[0] == output_vector_dimension_
                && (int32_t)weight_shape[1] == input_vector_dimension_, 
                "Linear1d weight matrix (index " + std::to_string(Weights) + " in parameters) must have shape (" + std::to_string(output_vector_dimension_) + ", " + std::to_string(input_vector_dimension_) + "); " +
                "instead got shape (" + std::to_string(weight_shape[0]) + ", " + std::to_string(weight_shape[1]) + ")",
                
                loc);

        xt::svector<std::size_t> bias_shape = parameters_[Biases].shape();
        str_assert(bias_shape.size() == 1, "Biases must be a 1d vector; bias vector currently has rank " + std::to_string(bias_shape.size()), loc);
        str_assert(bias_shape[0] == output_vector_dimension_, "Bias vector must have " + std::to_string(output_vector_dimension_) + " elements; currently has " + std::to_string(bias_shape[0]), loc);

        #endif
    }

public:
    /**
     * Names of indices: Weights=0, Biases=1
     */
    enum ParameterIndices {
        /**
         * Index 0 of parameters and gradients = 2d weight matrix
         */
        Weights = 0,

        /**
         * Index 1 of parameters and gradients = 1d bias vector
         */
        Biases = 1
    };


    /**
     * Creates a 1d linear layer with `input_dimension` inputs and `output_dimension` outputs.
     *
     * Weights and biases are randomly initialized, using a normal distribution with mean 0 and standard deviation 1.
     * Gradients are initialized to zeros.
     * 
     * @param input_dimension required size of input vectors. Positive
     * @param output_dimension size of output vectors. Positive
     */
    Linear1d(int32_t input_dimension, int32_t output_dimension) : input_vector_dimension_(input_dimension), output_vector_dimension_(output_dimension) {
        str_assert(input_dimension > 0, "Input dimension (" + std::to_string(input_dimension) + ") must be positive");
        str_assert(output_dimension > 0, "Output dimension (" + std::to_string(output_dimension) + ") must be positive");
        
        // PARAMETERS (weights, biases)
        parameters_.emplace_back(xt::random::randn<double>({output_dimension, input_dimension}, 0, 1));
        parameters_.emplace_back(xt::random::randn<double>({output_dimension}, 0, 1));

        // GRADIENTS
        gradients_.emplace_back(xt::zeros<double>({output_dimension, input_dimension}));
        gradients_.emplace_back(xt::zeros<double>({output_dimension}));
    }


    /**
    * @return deep pointer copy of this layer object
    */
    std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const override {
        return std::make_shared<Linear1d>(*this);
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //GETTERS

    /**
    * @return the string "linear1d ({input dimension}, {output dimension})"
    */
    std::string to_string() const override {
        return "linear1d (" + std::to_string(input_vector_dimension_) + ", " + std::to_string(output_vector_dimension_) + ")";
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //METHODS
    
    /**
     * Returns the result of the linear forward pass on `input`.
     *
     * If `input`'s single value is not a vector of this layer's input dimension, throws `cast::shape_error`.
     * @param input list containing the layer input. Has multiple axes
     * @return forward pass result
     */
    xt::xarray<double> forward(xt::xarray<double> input) override {
        str_assert(input.shape().size() > 1, "Input must have multiple axes (axis 0 is for batches)");
        assert_parameter_list_preconditions_();

        prev_inputs_ = input;
        return xt::linalg::dot(input, xt::transpose(parameters_[Weights])) + parameters_[Biases];
    }



    /**
     * Returns the gradients with respect to this layer and `upstream_gradients`, updating this layer's gradients.
     * @param upstream_gradients gradients from this layer's successor
     * @return dY/dL, where Y is the overall derivative and L is this layer's data, contained in index 0 of the output
     */
    xt::xarray<double> backward(xt::xarray<double> upstream_gradients) override {
        // std::cout << upstream_gradients << std::endl;
        str_assert(upstream_gradients.shape().size() > 1, "Input must have multiple axes (axis 0 is for batches)");
        assert_parameter_list_preconditions_();

        auto d_input = xt::linalg::dot(upstream_gradients, parameters_[Weights]); 
        gradients_[Weights] = xt::linalg::dot(xt::transpose(upstream_gradients), prev_inputs_); 
        gradients_[Biases] = xt::sum(upstream_gradients, {0}); 

        return d_input;
    }

};




}
#endif 