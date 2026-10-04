#ifndef CAST_OPTIMIZER_
#define CAST_OPTIMIZER_


#include "layer.hpp"

#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xio.hpp>

#include <iostream>
#include <initializer_list>
#include <memory>
#include <vector>


namespace cast {




/**
* Updates weights of a network's layers
*/
class Optimizer {
protected:
    /**
    * Network components that this optimizer improves
    */
    std::vector<std::shared_ptr<NetworkComponent>> components_;

    /**
    * Stores hyperparameters of the optimizer
    */
    std::vector<double> hyperparams_;


public:

    /**
    * @return deep copy of the optimizer's pointer. The new pointer cannot be used to modify the original.
    */
    virtual std::shared_ptr<Optimizer> shared_ptr_deep_copy() const = 0;

    //////////////////////////////////////////////////////////////////////////

    /**
    * @return optimizer's hyperparameters. Meaning of each index depends on the optimizer implementation.
    */
    virtual std::vector<double> hyperparameters() const {
        return hyperparams_;
    }

    /**
    * @return string representation of the optimizer object and its hyperparameters
    */
    virtual std::string to_string() const {
        return "optimizer";
    }

    //////////////////////////////////////////////////////////////////////////

    /**
    * Removes network components under optimization, as well as any data used for training.
    */
    virtual void clear_training_state() = 0;

    /**
    * Sets the hyperparameters to `new_hyperparams`.
    * @param new_hyperparams hyperparameters to set. Length and preconditions for each hyperparameter depend on the optimizer subclass
    */
    virtual void set_hyperparameters(std::initializer_list<double> new_hyperparams) = 0;

    //////////////////////////////////////////////////////////////////////////

    /**
    * Loads the optimizer with all information needed for training.
    * @param operators network components to optimize
    */
    virtual void initialize(std::vector<std::shared_ptr<NetworkComponent>>& operators) = 0;

    /**
    * Updates the parameters of each Layer object in the optimizer's stored components using each layer's stored gradients.
    * Non-Layers are unchanged.
    * @param zero_grad whether to set each operator's gradients to 0, after computing the optimization pass
    */
    virtual void step(bool zero_grad) = 0;

    //////////////////////////////////////////////////////////////////////////

    /**
    * Exports `optimizer` to the output stream `output_stream`, returning `output_stream` with `optimizer`'s information inside.
    * @param output_stream stream to put the optimizer into
    * @param optimizer Optimizer object to export
    * @return `output_stream` with `optimizer` inserted
    */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Optimizer& optimizer);
};

template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Optimizer& optimizer) {
    std::string optimizer_str = optimizer.to_string();
    output_stream << std::basic_string<CharT>(optimizer_str.begin(), optimizer_str.end());
    return output_stream;
}




/**
* Stochastic Gradient Descent optimizer with momentum
*/
class SGD : public Optimizer {
private:

    /**
    * Velocities for each parameter, for each component.
    * Index `i` corresponds to the velocities for component `i`.
    *
    * The velocities for each non-Layer is the empty list, {}.
    */
    std::vector<std::vector<xt::xarray<double>>> velocities_;

public:
    /**
    * SGD hyperparameter indices
    */
    enum HyperparamIndices {
        /**
        * Speed of convergence
        */
        LearningRate = 0,

        /**
        * Momentum coefficient
        */
        MomentumCoefficient = 1
    };


    /**
    * Creates a new SGD optimizer with initial learning rate `initial_lr`
    * @param initial_lr initial learning rate to use. Positive
    * @param initial_momentum_coeff initial momentum coefficient to use. Non-negative
    */
    SGD(double initial_lr, double initial_momentum_coeff) {
        str_assert(initial_lr > 0, "Initial learning rate (" + std::to_string(initial_lr) + ") must be positive");
        str_assert(initial_momentum_coeff >= 0, "Initial momentum coefficient (" + std::to_string(initial_momentum_coeff) + ") must be non-negative");

        hyperparams_.push_back(initial_lr);
        hyperparams_.push_back(initial_momentum_coeff);
    }


    /**
    * @return deep pointer copy of this SGD object
    */
    std::shared_ptr<Optimizer> shared_ptr_deep_copy() const override {
        return std::make_shared<SGD>(*this);
    }


    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //GETTERS
    
    /**
    * @return learning rate used by this optimizer
    */
    double learning_rate() const {
        return hyperparams_[LearningRate];
    }


    /**
    * @return momentum coefficient used by this optimizer
    */
    double momentum_coefficient() const {
        return hyperparams_[MomentumCoefficient];
    }


    /**
    * @return the string "sgd (learning rate {learning rate}, momentum coefficient {momentum coefficient})"
    */
    std::string to_string() const override {
        return "sgd (learning rate " + std::to_string(hyperparams_[LearningRate]) + ", momentum coefficient " + std::to_string(hyperparams_[MomentumCoefficient]) + ")";
    }

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //SETTERS

    /**
    * Removes the components of this optimizer and layer velocities used in training.
    */
    void clear_training_state() override {
        components_.clear();
        velocities_.clear();
    }


    /**
    * Sets this SGD optimizer's momentum coefficient to `new_momentum_coeff`.
    * @param new_momentum_coeff momentum coefficient to set. Non-negative
    */
    void set_momentum_coefficient(double new_momentum_coeff) {
        str_assert(new_momentum_coeff >= 0, "New momentum coeff must be positive- received " + std::to_string(new_momentum_coeff));
        hyperparams_[MomentumCoefficient] = new_momentum_coeff;
    }


    /**
    * Sets this SGD optimizer's learning rate to `new_learning_rate`.
    * @param new_learning_rate learning rate to set. Positive.
    */
    void set_learning_rate(double new_learning_rate) {
        str_assert(new_learning_rate > 0, "New learning rate must be positive- received " + std::to_string(new_learning_rate));
        hyperparams_[LearningRate] = new_learning_rate;
    }


    /**
    * Sets the optimizer's learning rate to `new_hyperparams[0]`, and the momentum coefficient to `new_hyperparams[1]`.
    * @param new_hyperparams new hyperparameters. Of length 2. Index 0 is positive, index 1 is non-negative
    */
    void set_hyperparameters(std::initializer_list<double> new_hyperparams) override {
        std::vector<double> new_hyperparams_vec = new_hyperparams;
        str_assert(new_hyperparams_vec.size() == 2, "New hyperparameter list must be of length 2");
        str_assert(new_hyperparams_vec[LearningRate] > 0, "Learning rate must be positive; received " + std::to_string(new_hyperparams_vec[LearningRate]));
        str_assert(new_hyperparams_vec[MomentumCoefficient] >= 0, "Momentum coefficient must be non-negative; received " + std::to_string(new_hyperparams_vec[MomentumCoefficient]));

        hyperparams_ = new_hyperparams;
    }


    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //METHODS

    
    /**
    * Loads the SGD optimizer with layer velocities taken from `components`.
    * @param components network components to optimize. Non-empty, and no element can be `nullptr`
    */
    void initialize(std::vector<std::shared_ptr<NetworkComponent>>& components) override {
        str_assert(components.size() > 0, "Operator list must be non-empty");

        velocities_.clear(); // Clear out any old state if re-initializing

        components_ = components;

        for (const std::shared_ptr<NetworkComponent>& op : components) {
            std::vector<xt::xarray<double>> layer_vels;

            str_assert(op != nullptr, "All operators in initialization cannot be nullptr");

            // Check if this operator is a subclass of Layer
            std::shared_ptr<Layer> layer = std::dynamic_pointer_cast<Layer>(op);
            
            // It is a layer: initialize velocities for its parameters
            if (layer != nullptr) {
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
    * This method must be called after the optimizer has been initialized.
    * @param zero_grad whether to set each operator's gradients to 0, after computing the optimization pass
    */
    void step(bool zero_grad = true) override {
        str_assert(velocities_.size() > 0, "The optimizer must have been initialized prior to calling this method");

        for (int32_t l = 0; l < (int32_t)components_.size(); l++) {
            // Check if this operator is a subclass of Layer. If not, skip it
            auto current_layer = std::dynamic_pointer_cast<Layer>(components_[l]);
            if(current_layer == nullptr) {
                continue;
            }

            std::vector<xt::xarray<double>>& params = current_layer->parameters();
            std::vector<xt::xarray<double>>& grads = current_layer->gradients();
            std::vector<xt::xarray<double>>& vels = velocities_[l];

            // Do SGD update on each of the layer's parameters
            for(int32_t i = 0; i < current_layer->parameters().size(); i++) {

                //v = momentum * v + learning_rate * gradient
                vels[i] = hyperparams_[MomentumCoefficient] * vels[i] + hyperparams_[LearningRate] * grads[i];

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