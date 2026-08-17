#ifndef CAST_OPTIMIZER_
#define CAST_OPTIMIZER_


#include "layer.hpp"

#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xio.hpp>

#include <iostream>
#include <memory>
#include <vector>


namespace cast {




/**
* Updates weights of a network's layers
*/
class Optimizer {
public:

    /**
    * Loads the optimizer with all information needed for training.
    * @param operators network components to optimize
    */
    virtual void initialize(std::vector<std::shared_ptr<NetworkComponent>>& operators) = 0;

    /**
    * Updates the parameters of each Layer object in `operators` using each layer's stored gradients.
    * Non-Layers are unchanged.
    *
    * Mutates `operators`.
    *
    * @param operators network operators to update
    * @param zero_grad whether to set each operator's gradients to 0, after computing the optimization pass
    */
    virtual void step(bool zero_grad) = 0;
};



/**
* Stochastic Gradient Descent optimizer
*/
class SGD : public Optimizer {
private:

    /**
    * Dictates speed of convergence
    */
    double learning_rate_;

    /**
    * Dictates momentum
    */
    double momentum_coefficient_;

    /**
    * Operators that this optimizer improves
    */
    std::vector<std::shared_ptr<NetworkComponent>> operators_;

    /**
    * Velocities for each parameter, for each operator.
    * Index `i` corresponds to the velocities for operator `i`.
    *
    * The velocity list for each non-Layer is empty.
    */
    std::vector<std::vector<xt::xarray<double>>> velocities_;

public:
    /**
    * Creates a new SGD optimizer with initial learning rate `initial_lr`
    * @param initial_lr initial learning rate to use. Positive
    * @param initial_momentum_coeff initial momentum coefficient to use. Non-negative
    */
    SGD(double initial_lr, double initial_momentum_coeff) : learning_rate_(initial_lr), momentum_coefficient_(initial_momentum_coeff) {
        str_assert(initial_lr > 0, "Initial learning rate (" + std::to_string(initial_lr) + ") must be positive");
        str_assert(initial_momentum_coeff >= 0, "Initial momentum coefficient (" + std::to_string(initial_momentum_coeff) + ") must be non-negative");
    }



    /**
    * Loads the SGD optimizer with all information needed for training, taken from `operators`.
    * @param operators operators to optimize. Non-empty, and no element can be `nullptr`
    */
    void initialize(std::vector<std::shared_ptr<NetworkComponent>>& operators) override {
        str_assert(operators.size() > 0, "Operator list must be non-empty");

        velocities_.clear(); // Clear out any old state if re-initializing

        operators_ = operators;

        for (const std::shared_ptr<NetworkComponent>& op : operators) {
            std::vector<xt::xarray<double>> layer_vels;

            str_assert(op != nullptr, "All operators in initialization cannot be nullptr");

            // Check if this operator is a subclass of Layer
            std::shared_ptr<Layer> layer = std::dynamic_pointer_cast<Layer>(op);
            
            if (layer != nullptr) {
                // It is a layer: initialize velocities for its parameters
                for (const xt::xarray<double>& param : layer->parameters()) {
                    layer_vels.push_back(xt::zeros_like(param));
                }
            }

            velocities_.push_back(layer_vels);
        }
    }


    /**
    * Updates `operators` using SGD. Any non-layer (i.e. operators that are not subclasses of `Layer`) are ignored.
    *
    * This method must be called after using `initialize`. The operators cannot have been modified since calling `initialize`.
    * @param zero_grad whether to set each operator's gradients to 0, after computing the optimization pass
    */
    void step(bool zero_grad = true) override {
        str_assert(velocities_.size() > 0, "The optimizer must have been initialized prior to calling this method");

        for (int32_t l = 0; l < (int32_t)operators_.size(); l++) {
            // Check if this operator is a subclass of Layer. If not, skip it
            auto current_layer = std::dynamic_pointer_cast<Layer>(operators_[l]);
            if(current_layer == nullptr) {
                continue;
            }

            std::vector<xt::xarray<double>>& params = current_layer->parameters();
            std::vector<xt::xarray<double>>& grads = current_layer->gradients();
            std::vector<xt::xarray<double>>& vels = velocities_[l];

            // Do SGD update on each of the layer's parameters
            for(int32_t i = 0; i < current_layer->parameters().size(); i++) {

                //v = momentum * v + learning_rate * gradient
                vels[i] = momentum_coefficient_ * vels[i] + learning_rate_ * grads[i];

                //param = param - v
                params[i] -= vels[i];
                
                //set gradients to 0
                if(zero_grad) {
                    current_layer->gradients() [i] = xt::zeros_like(current_layer->gradients() [i]);
                }
            }
        }
    }
};




}
#endif