#ifndef CAST_LOSS_CALCULATOR_
#define CAST_LOSS_CALCULATOR_

#include "cast_exceptions.hpp"
#include "network_component.hpp"

#include <string>
#include <xtensor/containers/xarray.hpp>

namespace cast {




/**
 * Computes loss, the error between the expected and predicted network outputs
 */
class LossCalculator {
protected:
    /**
    * Asserts that `predicted` and `expected` are non-empty and of the same shape.
    *
    * Does nothing if `NDEBUG` is defined.
    * @param predicted predictions for a given input
    * @param expected expected predictions for the same input
    */
    void assert_nonempty_same_shape_(xt::xarray<double> predicted, xt::xarray<double> expected) const {
        #ifndef NDEBUG

        str_assert(predicted.size() > 0, "Predicted value must be non-empty");

        xt::svector<std::size_t> predicted_shape = predicted.shape();
        xt::svector<std::size_t> expected_shape = expected.shape();
        str_assert(expected_shape.size() == predicted_shape.size(), "Expected value (" + std::to_string(expected_shape.size()) + ") must have the same rank as predicted (" + std::to_string(predicted_shape.size()) + ")");
        
        for(int i = 0; i < predicted_shape.size(); i++) {
            str_assert(predicted_shape[i] == expected_shape[i], "Predicted shape and expected shape mismatch on axis " + std::to_string(i));
        }

        #endif
    }


public:

    /**
    * @return deep pointer copy of this loss calculator. The deep copy cannot be used to modify the original.
    */
    virtual std::shared_ptr<LossCalculator> shared_ptr_deep_copy() const = 0;

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
    virtual double compute(xt::xarray<double> predicted, xt::xarray<double> expected) const = 0;

    /**
     * Returns the tensor-valued gradient of the loss, between `predicted` and `expected`, as computed by this calculator.
     * @param predicted network's predictions for a given input
     * @param expected what the network should have predicted for the input
     * @return gradient of the loss between `predicted` and `expected` wrapped in a Tensor
     */
    virtual xt::xarray<double> compute_gradient(xt::xarray<double> predicted, xt::xarray<double> expected) const = 0;

};



/**
 * Calculates Mean Squared Error (MSE) loss.
 *
 * For each element in the output, MSE subtracts corresponding elements of the predicted and actual loss,
 * then squares the difference. The loss is the sum of the squared differences, divided by the number of 
 * elements in the input, divided by 2.
 */
class MeanSquaredError : public LossCalculator {
public:

    /**
    * Creates a new MSE loss calculator
    */
    MeanSquaredError() = default;


    /**
    * @return deep pointer copy of this Sigmoid object
    */
    std::shared_ptr<LossCalculator> shared_ptr_deep_copy() const override {
        return std::make_shared<MeanSquaredError>(*this);
    }



    /**
     * @return the string "mean_squared_error"
     */
    std::string name() override {
        return "mean_squared_error";
    }

    /**
     * Returns the computed Mean Squared Error loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Non-empty
     * @param expected what the model should have predicted for a given input. Has the same shape as `predicted`
     * @return MSE loss between `predicted` and `expected`
     */
    double compute(xt::xarray<double> predicted, xt::xarray<double> expected) const override {
        assert_nonempty_same_shape_(predicted, expected);

        double sum_sq = xt::sum(xt::square(predicted - expected))();
        return sum_sq / (2.0 * static_cast<double>(predicted.size()));
    }


    
    /**
     * Returns the gradient of MSE loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Non-empty
     * @param expected what the model should have predicted for a given input. Has the same number of elements as `predicted`
     * @return gradient of MSE loss between `predicted` and `expected`
     */
    xt::xarray<double> compute_gradient(xt::xarray<double> predicted, xt::xarray<double> expected) const override {
        assert_nonempty_same_shape_(predicted, expected);

        xt::xarray<double> grad_data = (predicted - expected) / predicted.size();
        return grad_data;
    }
};




}
#endif 