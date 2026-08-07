#ifndef CAST_NETWORK_
#define CAST_NETWORK_


#include "cast_exceptions.hpp"
#include "activation_function.hpp"
#include "loss_calculator.hpp"
#include "optimizer.hpp"
#include "tensor_graph/tensor_graph.hpp"

#include <iostream>
#include <memory>
#include <xtensor/containers/xarray.hpp>



namespace cast {




/**
 * Network that allows users to arrange layers and operators in any order
 */
class CustomNetwork {
public:
    /**
     * Creates an empty custom network object
     */
    CustomNetwork() {
    }


    /**
     * Returns the result of the network's forward pass.
     * 
     * This method is user-defined.
     */
    virtual xt::xarray<double> forward(xt::xarray<double> input) = 0;
};



/**
 * (Sequentially defined) network
 */
class Network : public CustomNetwork {
public: //normally private
    /**
     * Operators that the network uses, in order
     */
    std::vector<std::shared_ptr<TensorOperator>> operators_;

    /**
     * Holds the tensor first given to this network
     */
    std::shared_ptr<Tensor> initial_input_;

    /**
    * Holds the most recent output to this network
    */
    std::shared_ptr<Tensor> output_;

 
    /**
     * Loss metric used by this network
     */
    std::shared_ptr<LossCalculator> loss_calc_;

    /**
    * Optimizer used by this network
    */
    std::shared_ptr<Optimizer> optimizer_;

    /**
    * True if the network is ready for training and evaluation
    */
    bool enabled_;

public:
    /**
     * Creates a new network
     */
    Network() : enabled_(false) {       
    };

    /**
     * Registers `op` as the next operator to execute in the network
     * @param op new operator to add
     */
    void add_operator(std::shared_ptr<TensorOperator> op) {
        operators_.push_back(op);
    }

    /**
     * Sets this network's loss calculator to `calc`.
     * @param calc new loss calculator to use
     */
    void set_loss_calculator(std::shared_ptr<LossCalculator> calc) {
        //Reset the loss calculator if it exists
        if(loss_calc_) {
            loss_calc_.reset();
        }

        //Create deep pointer of the new calculator (OK for now- LossCalculators have no internal state)
        loss_calc_ = calc;
    }

    /**
     * Sets this network's optimizer to `optim`.
     * @param optim new optimizer to use
     */
    void set_optimizer(std::shared_ptr<Optimizer> optim) {
        //Reset optimizer if it exists
        if(optimizer_) {
            optimizer_.reset();
        }

        optimizer_ = optim;
    }


    /**
    * Checks if the network has the necessary components to run. 
    * If not, throws `invalid_config`. If so, allows training and optimization.
    */
    void enable() {
        if(!loss_calc_) {
            throw invalid_config("Network needs a defined loss calculator");
        }
        if(!optimizer_) {
            throw invalid_config("Network needs a defined optimizer");
        }

        optimizer_->initialize(operators_);

        enabled_ = true;
    }


    /**
     * Returns the result of the network's forward pass on `input`
     * 
     * @param input tensor to compute forward pass on
     * @return result of forward pass
     */
    xt::xarray<double> forward(xt::xarray<double> input) override {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to training");
        }

        //Anchor the forward pass tensors by saving the input
        initial_input_ = std::make_shared<Tensor>(input);

        //Cycle the input through the operators
        std::vector<std::shared_ptr<Tensor>> current_node = {initial_input_};
        for(const std::shared_ptr<TensorOperator>& op : operators_) {
            //Each operator creates and saves a shared pointer to its intermediate output
            current_node = op->compute_and_link(current_node);
        }

        output_ = current_node[0]; //save output (only the first one)
        return current_node[0]->data();
    }


    /**
     * Computes the backward pass, initially using `predicted` and `expected`.
     * @param predicted network's prediction for a given input
     * @param expected what the network should have predicted for the input
     */
    void backward(xt::xarray<double> predicted, xt::xarray<double> expected) {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to computing backwards pass");
        }
        
        //Compute loss gradient and set it as the initial gradient
        xt::xarray<double> loss_gradient = loss_calc_->compute_gradient(predicted, expected);
        std::vector<xt::xarray<double>> current_gradients = {loss_gradient};

        //Check the output
        if(!output_) {
            throw invalid_config("No previous output found- must call forward pass prior to using backwards pass");
        }

        //Initialize the current output
        std::shared_ptr<Tensor> current_result = output_;
        

        if(!current_result) {
            throw invalid_config("The network's forward method must be called prior to computing a backwards pass");
        }
        // std::cout << *current_result << std::endl;

        //Cycle through each node and operator, backwards
        while(current_result != nullptr && current_result->prev_operator_ != nullptr) {
            // std::cout << *current_result << std::endl;
            std::shared_ptr<TensorOperator> current_operator = current_result->prev_operator_; 

            // std::cout << current_operator->name() << std::endl;
            
            //Pass loss/gradients backward through operator (the operators are what contain the weights)
            current_gradients = current_operator->compute_backwards_pass(current_gradients);

            //Move one operator backward
            if (!current_operator->predecessors_.empty()) {
                current_result = current_operator->predecessors_[0];
            } 
            else {
                break;
            }
        }

    }


    /**
    * Runs an optimization pass on the network's layers, using the stored optimizer
    */
    void optimize() {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to optimizing"); 
        }

        optimizer_->step(operators_, true);
        output_.reset();
    }

};




}
#endif