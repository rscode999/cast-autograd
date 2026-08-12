#ifndef CAST_LAYER_
#define CAST_LAYER_

#include "cast_exceptions.hpp"
#include "tensor_operator.hpp"

#include <source_location>
#include <string>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <cassert>

namespace cast {




/**
* Layer in a network. Contains parameters and gradients for each parameter.
*
* Note that layers can accept multiple inputs and give multiple outputs.
*/
class Layer : public TensorOperator {
protected:

    /**
     * Rank of the inputs taken by this layer. 1=vector, 2=matrix, 3=3d tensor, etc.
     */
    int32_t input_tensor_rank_;

    /**
     * Rank of the outputs taken by this layer. 1=vector, 2=matrix, 3=3d tensor, etc.
     */
    int32_t output_tensor_rank_;


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
    std::vector<xt::xarray<double>> prev_inputs_;


public:

    virtual std::string name() const override {
        return "layer";
    }

    /**
     * @return parameters (weights, biases, ...) of this layer, as a std::vector of Tensors
     */
    std::vector<xt::xarray<double>>& parameters() {
        return parameters_;
    }

    /**
     * @return gradients of the weights, biases, etc. of this layer, as a std::vector of Tensors
     */
    std::vector<xt::xarray<double>>& gradients() {
        return gradients_;
    }
};




/**
 * Performs a fully-connected dense linear operation on a single 1d vector. Produces a single 1d vector.
 *
 * Index 0 of its parameters is a 2d weight matrix. Index 1 is a 1d bias vector.
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
    void assert_parameter_shape(std::source_location loc = std::source_location::current()) {
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
     * Weights and biases are randomly initialized, using a normal distribution with mean 0 and std. dev. 1.
     * Gradients are initialized to zeros.
     * 
     * @param input_dimension required size of input vectors. Precondition: Positive
     * @param output_dimension size of output vectors. Precondition: Positive
     */
    Linear1d(int32_t input_dimension, int32_t output_dimension) : input_vector_dimension_(input_dimension), output_vector_dimension_(output_dimension) {
        str_assert(input_dimension > 0, "Input dimension (" + std::to_string(input_dimension) + ") must be positive");
        str_assert(output_dimension > 0, "Output dimension (" + std::to_string(output_dimension) + ") must be positive");
        
        // Exclusively uses inputs and outputs of rank 1 (vectors).
        input_tensor_rank_ = 1;
        output_tensor_rank_ = 1;
        
        // PARAMETERS (weights, biases) - wrapped in Tensor
        parameters_.emplace_back(xt::random::randn<double>({output_dimension, input_dimension}, 0, 1));
        parameters_.emplace_back(xt::random::randn<double>({output_dimension}, 0, 1));

        // GRADIENTS - wrapped in Tensor
        gradients_.emplace_back(xt::zeros<double>({output_dimension, input_dimension}));
        gradients_.emplace_back(xt::zeros<double>({output_dimension}));

        // PREV INPUT - wrapped in Tensor
        prev_inputs_.emplace_back(xt::zeros<double>({input_dimension}));
    }



    /**
    * @return the string "linear1d"
    */
    std::string name() const override {
        return "linear1d";
    }


    
    /**
     * Returns the result of the linear forward pass on `input`.
     *
     * To properly compute, the weights and biases of this layer must be compatible with the preset input and output dimensions of this layer.
     * @param input list containing the layer input. Has exactly 1 element
     * @return forward pass result
     */
    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> input) override {
        str_assert(input.size() == 1, "Linear1d forward pass computation takes 1 input; received " + std::to_string(input.size()) + " inputs");
        str_assert(input[0].dimension() == 1, "Linear1d layers require vector (rank 1) inputs; input is of rank " + std::to_string(input[0].dimension()));
        str_assert(input[0].shape()[0] == input_vector_dimension_, "This layer requires vectors of length " + std::to_string(input_vector_dimension_) + "; received length " + std::to_string(input[0].shape()[0]));
        assert_parameter_shape();

        prev_inputs_[0] = input[0];

        xt::xarray<double> output_tensor = xt::linalg::dot(parameters_[Weights], input[0]) + parameters_[Biases];
        return {output_tensor};
    }



    /**
     * Returns the gradients with respect to this layer and `upstream_gradients`, updating this layer's gradients.
     * @param upstream_gradients gradients from this layer's successor. Precondition: contains a single 1d vector
     * @return dY/dL, where Y is the overall derivative and L is this layer's data, contained in index 0 of the output
     */
    std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> upstream_gradients) override {
        str_assert(upstream_gradients.size() == 1, "Linear1d backwards operation must have one input; got " + std::to_string(upstream_gradients.size()));
        str_assert(upstream_gradients[0].shape().size() == 1, "Linear1d backwards requires a vector (rank 1)");
        str_assert(upstream_gradients[0].shape()[0] == output_vector_dimension_, "Linear1d backwards requires a vector of size " + std::to_string(output_vector_dimension_) + "; got size " + std::to_string(upstream_gradients[0].shape()[0]));
        assert_parameter_shape();

        xt::xarray<double> d_output = upstream_gradients[0];

        // dW incremented by: d_output * transpose of prev. input
        gradients_[Weights] += xt::view(d_output, xt::all(), xt::newaxis()) * xt::view(prev_inputs_[0], xt::newaxis(), xt::all());

        // dB incremented by d_output
        gradients_[Biases] += d_output;

        // d_Input = transpose of weights * d_output, to next layer
        xt::xarray<double> d_input = xt::linalg::dot(xt::transpose(parameters_[Weights]), d_output);

        // std::cout << "gradients " << d_input << std::endl;

        // Return the gradient vector for the previous layer wrapped in a Tensor
        return {d_input};
    }
};




}
#endif 