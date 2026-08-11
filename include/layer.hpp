#ifndef CAST_LAYER_
#define CAST_LAYER_

#include "tensor_operator.hpp"

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
    Linear1d(int32_t input_dimension, int32_t output_dimension) 
    : 
    input_vector_dimension_(input_dimension), output_vector_dimension_(output_dimension) {
        assert(input_dimension > 0);
        assert(output_dimension > 0);
        
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
     * Returns the result of the linear forward pass on `input`
     * @param input list containing the single layer input. Must have exactly 1 element
     * @return forward pass result
     */
    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> input) override {
        assert(input.size() == 1 && "Linear1d forward pass computation has one input");

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
        assert(upstream_gradients.size() == 1 && "Linear1d backwards operation must have one input");

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