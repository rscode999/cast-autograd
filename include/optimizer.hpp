#ifndef CAST_OPTIMIZER_
#define CAST_OPTIMIZER_

#include "tensor_graph/tensor_graph.hpp"
#include "xtensor/generators/xbuilder.hpp"

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

    virtual void initialize(std::vector<std::shared_ptr<TensorOperator>> operators) = 0;

    /**
    * Updates the parameters of each Layer object in `operators` using each layer's stored gradients.
    * Non-Layers are unchanged.
    *
    * Mutates `operators`.
    *
    * @param operators network operators to update
    */
    virtual void step(std::vector<std::shared_ptr<TensorOperator>> operators, bool zero_grad) = 0;
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

    double momentum_coefficient_;

    std::vector<std::vector<xt::xarray<double>>> velocities_;

public:
    /**
    * Creates a new SGD optimizer with initial learning rate `initial_lr`
    * @param initial_lr initial learning rate to use. Precondition: Positive
    * @param initial_momentum_coeff initial momentum coefficient to use. Precondition: Non-negative
    */
    SGD(double initial_lr, double initial_momentum_coeff) : learning_rate_(initial_lr), momentum_coefficient_(initial_momentum_coeff) {
    }

    /**
    * Loads the SGD optimizer with all information needed for training 
    */
    void initialize(std::vector<std::shared_ptr<TensorOperator>> operators) override {
        velocities_.clear(); // Clear out any old state if re-initializing

        for (const std::shared_ptr<TensorOperator>& op : operators) {
            std::vector<xt::xarray<double>> layer_vels;

            // Check if this operator is a subclass of Layer
            auto layer = std::dynamic_pointer_cast<Layer>(op);
            
            if (layer != nullptr) {
                // It is a layer: initialize velocities for its parameters
                for (const auto& param : layer->parameters()) {
                    layer_vels.push_back(xt::zeros_like(param));
                }
            }

            velocities_.push_back(layer_vels);
        }
    }


    /**
    * Updates `operators` using SGD.
    *
    * Mutates `operators`.
    * @param operators layers to update
    */
    void step(std::vector<std::shared_ptr<TensorOperator>> operators, bool zero_grad = true) override {

        for (int32_t l = 0; l < (int32_t)operators.size(); l++) {
            // Check if this operator is a subclass of Layer. If not, skip it
            auto current_layer = std::dynamic_pointer_cast<Layer>(operators[l]);
            if(current_layer == nullptr) {
                continue;
            }

            auto& params = current_layer->parameters();
            auto& grads = current_layer->gradients();
            auto& vels = velocities_[l];

            // Do SGD update on each of the layer's parameters
            for(int32_t i = 0; i < current_layer->parameters().size(); i++) {
                //v = momentum * v + learning_rate * gradient
                vels[i] = momentum_coefficient_ * vels[i] + learning_rate_ * grads[i];

                //param = param - v
                params[i] -= vels[i];

                std::cout << "params updated to " << params[i] << std::endl;
                
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