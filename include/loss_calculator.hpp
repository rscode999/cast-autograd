#ifndef CAST_LOSS_CALCULATOR_
#define CAST_LOSS_CALCULATOR_

#include "tensor_graph/tensor_graph.hpp"

#include <cassert>
#include <string>
#include <xtensor/containers/xarray.hpp>

namespace cast {




/**
 * Computes loss, the error between the expected and predicted network outputs
 */
class LossCalculator {
public:

    /**
     * Creates a new LossCalculator
     */
    LossCalculator() = default;


    /**
     * @return the calculator's identifying string. Defaults to "loss_calculator" if not overridden by an implementing class.
     */
    virtual std::string name() {
        return "loss_calculator";
    }

    /**
     * Returns the loss between `predicted` and `expected`, as computed by this calculator.
     * @param predicted network's predictions for a given input
     * @param expected what the network should have predicted for the input
     * @return loss of `predicted` and `expected`
     */
    virtual double compute(Tensor predicted, Tensor expected) const = 0;

    /**
     * Returns the tensor-valued gradient of the loss, between `predicted` and `expected`, as computed by this calculator.
     * @param predicted network's predictions for a given input
     * @param expected what the network should have predicted for the input
     * @return gradient of the loss between `predicted` and `expected` wrapped in a Tensor
     */
    virtual Tensor compute_gradient(Tensor predicted, Tensor expected) const = 0;

};



/**
 * Calculates Mean Squared Error loss.
 *
 * For each element in the output, MSE subtracts corresponding elements of the predicted and actual loss,
 * then squares the difference. The loss is the sum of the squared differences, divided by the number of 
 * elements in the input, divided by 2.
 */
class MeanSquaredError : public LossCalculator {
public:

    /**
     * @return the string "mean_squared_error"
     */
    std::string name() override {
        return "mean_squared_error";
    }

    /**
     * Returns the computed Mean Squared Error loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Precondition: Non-empty
     * @param expected what the model should have predicted for a given input. Precondition: Has the same number of elements as `predicted`
     * @return MSE loss between `predicted` and `expected`.
     */
    double compute(Tensor predicted, Tensor expected) const override {
        xt::xarray<double> pred_data = predicted.data();
        xt::xarray<double> exp_data = expected.data();

        assert(pred_data.size() > 0 && "Predicted input must be non-empty");
        assert(pred_data.size() == exp_data.size() && "Number of elements in predicted and expected must match");

        double sum_sq = xt::sum(xt::square(pred_data - exp_data))();
        return sum_sq / (2.0 * static_cast<double>(pred_data.size()));
    }

    /**
     * Returns the gradient of MSE loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Precondition: Non-empty
     * @param expected what the model should have predicted for a given input. Precondition: Has the same number of elements as `predicted`
     * @return gradient of MSE loss between `predicted` and `expected` wrapped in a Tensor
     */
    Tensor compute_gradient(Tensor predicted, Tensor expected) const override {
        xt::xarray<double> predicted_data = predicted.data();
        xt::xarray<double> expected_data = expected.data();

        assert(predicted_data.size() > 0 && "Predicted input must be non-empty");
        assert(predicted_data.size() == expected_data.size() && "Number of elements in predicted and expected must match");
        
        xt::xarray<double> grad_data = (predicted_data - expected_data) / predicted_data.size();
        return Tensor(grad_data);
    }
};




}
#endif 