#ifndef CAST_LOSS_CALCULATOR_
#define CAST_LOSS_CALCULATOR_

#include "cast_exceptions.hpp"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>

#include <string>


namespace cast {




/**
 * Computes loss, the error between the expected and predicted network outputs
 *
 * Has a reconfigurable batch size. Batch size default is 0, meaning that the network does not use batches.
 */
class LossCalculator {
protected:
    /**
    * Number of examples per batch. 0 if the calculator accepts single inputs.
    */
    int32_t batch_size_ = 0;

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
    * @return the calculator's batch size. Equals 0 if the calculator does not use batches.
    */
    int32_t batch_size() const {
        return batch_size_;
    }

    /**
     * @return the calculator's identifying string. Defaults to "loss_calculator" if not overridden by an implementing class.
     */
    virtual std::string to_string() const {
        return "loss_calculator";
    }

    /**
    * Sets the calculator's batch size to `new_batch_size`.
    * A batch size of 0 means that the calculator does not use batches.
    * @param new_batch_size batch size to set. Non-negative.
    */
    void set_batch_size(int32_t new_batch_size) {
        str_assert(new_batch_size >= 0, "New batch size must be non-negative- got " + std::to_string(new_batch_size));
        batch_size_ = new_batch_size;
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

    /**
    * Exports `calc` to the output stream `output_stream`, returning `output_stream` with `calc`'s information inside.
    * @param output_stream stream to put the loss calculator into
    * @param calc LossCalculator object to export
    * @return `output_stream` with `calc` inserted
    */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const LossCalculator& calc);
};

template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const LossCalculator& calc) {
    std::string calc_str = calc.to_string();

    output_stream << std::basic_string<CharT>(calc_str.begin(), calc_str.end());
    return output_stream;
}




/**
 * Calculates Mean Squared Error (MSE) loss.
 *
 * For each element in the output, MSE subtracts corresponding elements of the predicted and expected values,
 * then squares the difference. The loss is the sum of the squared differences, divided by the number of 
 * elements in the predicted value, divided by 2.
 */
class MeanSquaredError : public LossCalculator {
public:

    /**
    * Creates a new MSE loss calculator
    */
    MeanSquaredError() = default;


    /**
    * @return deep pointer copy of this loss calculator object
    */
    std::shared_ptr<LossCalculator> shared_ptr_deep_copy() const override {
        return std::make_shared<MeanSquaredError>(*this);
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the string "mean_squared_error"
     */
    std::string to_string() const override {
        return "mean_squared_error";
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Returns the computed Mean Squared Error loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Non-empty
     * @param expected what the model should have predicted for a given input. Has the same shape as `predicted`
     * @return MSE loss between `predicted` and `expected`
     */
    double compute(xt::xarray<double> predicted, xt::xarray<double> expected) const override {
        assert_nonempty_same_shape_(predicted, expected);

        double sum_sq = xt::sum(xt::square(predicted - expected))();
        double single_loss = sum_sq / (2.0 * static_cast<double>(predicted.size()));

        // Return total MSE loss across the batch if batch_size_ is nonzero
        if (batch_size_ != 0) {
            return single_loss * static_cast<double>(batch_size_);
        }

        // Otherwise, return the single loss value
        return single_loss;
    }


    
    /**
     * Returns the gradient of MSE loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Non-empty
     * @param expected what the model should have predicted for a given input. Has the same number of elements as `predicted`
     * @return gradient of MSE loss between `predicted` and `expected`
     */
    xt::xarray<double> compute_gradient(xt::xarray<double> predicted, xt::xarray<double> expected) const override {
        assert_nonempty_same_shape_(predicted, expected);

        xt::xarray<double> single_grad = (predicted - expected) / static_cast<double>(predicted.size());

        // Return total MSE gradient across the batch if batch_size_ is nonzero
        if (batch_size_ != 0) {
            return single_grad * static_cast<double>(batch_size_);
        }

        // Otherwise, return the single gradient value
        return single_grad;
    }
};



/**
* Calculates cross-entropy loss
*/
class CrossEntropy : public LossCalculator {
public:
    /**
    * Small constant added to prevent log(0)
    */
    static constexpr double epsilon = 1e-15;

    /**
    * Creates a new cross-entropy loss calculator
    */
    CrossEntropy() = default;


    /**
    * @return deep pointer copy of this loss calculator object
    */
    std::shared_ptr<LossCalculator> shared_ptr_deep_copy() const override {
        return std::make_shared<CrossEntropy>(*this);
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the string "cross_entropy"
     */
    std::string to_string() const override {
        return "cross_entropy";
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Returns the computed cross-entropy loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Non-empty
     * @param expected what the model should have predicted for a given input. Has the same shape as `predicted`
     * @return cross entropy loss between `predicted` and `expected`
     */
    double compute(xt::xarray<double> predicted, xt::xarray<double> expected) const override {
        assert_nonempty_same_shape_(predicted, expected);

        auto clipped_pred = xt::clip(predicted, epsilon, 1.0 - epsilon);
        // double batch_size = static_cast<double>(predicted.shape(0));
    
        // Mean categorical/binary cross-entropy loss over the batch
        return -xt::sum(expected * xt::log(clipped_pred))();
    }


    
    /**
     * Returns the gradient of cross-entropy loss between `predicted` and `expected`.
     * @param predicted model's predictions for a given input. Non-empty
     * @param expected what the model should have predicted for a given input. Has the same number of elements as `predicted`
     * @return gradient of cross entropy loss between `predicted` and `expected`
     */
    xt::xarray<double> compute_gradient(xt::xarray<double> predicted, xt::xarray<double> expected) const override {
        assert_nonempty_same_shape_(predicted, expected);

        auto clipped_pred = xt::clip(predicted, epsilon, 1.0 - epsilon);
        // double batch_size = static_cast<double>(predicted.shape(0));
        
        // Derivative of -expected * log(predicted) divided by batch size
        return (-expected / clipped_pred);
    }
};




}
#endif 