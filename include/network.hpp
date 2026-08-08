#ifndef CAST_NETWORK_
#define CAST_NETWORK_


#include "cast_exceptions.hpp"
#include "activation_function.hpp"
#include "loss_calculator.hpp"
#include "optimizer.hpp"
#include "tensor_graph/tensor.hpp"
#include "tensor_graph/tensor_graph.hpp"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <xtensor/containers/xarray.hpp>



namespace cast {




/**
 * Network that allows users to arrange layers and operators in any order
 */
class CustomNetwork {
private:
    /**
     * Holds the tensor first given to this network
     */
    std::shared_ptr<Tensor> initial_input_;

    /**
    * Holds the most recent output to this network
    */
    std::shared_ptr<Tensor> output_;

    std::shared_ptr<LossCalculator> loss_calc_;


    void topological_sort(std::shared_ptr<Tensor> node, 
                          std::unordered_set<TensorOperator*>& visited, 
                          std::vector<std::shared_ptr<TensorOperator>>& sorted_ops) {
        if (!node || !node->prev_operator_) {
            return; // Reached input leaf node
        }

        std::shared_ptr<TensorOperator> op = node->prev_operator_;
        if (visited.find(op.get()) != visited.end()) {
            return; // Already visited this operator
        }

        visited.insert(op.get());

        // Visit all predecessors (inputs to this operator) first
        for (const std::shared_ptr<Tensor>& pred : op->predecessors_) {
            topological_sort(pred, visited, sorted_ops);
        }

        sorted_ops.push_back(op);
    }

protected:
    /**
     * Returns the result of the network's forward pass.
     * 
     * This method is user-defined.
     */
    virtual Tensor forward(Tensor input) = 0;

public:
    /**
     * Creates an empty custom network object
     */
    CustomNetwork() {
    }


    /**
     * Returns the result of the network's forward pass, saving the initial and final inputs
     */
    virtual Tensor forward_and_link(Tensor input) {
        initial_input_ = std::make_shared<Tensor>(input);

        Tensor forward_result = forward(input);
        output_ = std::make_shared<Tensor>(forward_result);

        return forward_result;
    }


    void backward(Tensor predicted, Tensor expected) {
        // 1. Initialize loss gradient (e.g., MSE: predicted - expected, or similar)
        Tensor loss_grad = loss_calc_->compute_gradient(predicted, expected);

        // 2. Topological sort the tensor nodes
        std::unordered_set<TensorOperator*> visited;
        std::vector<std::shared_ptr<TensorOperator>> sorted_ops;
        topological_sort(output_, visited, sorted_ops);

        // 3. Map to accumulate incoming gradients for each tensor
        //    Using a pointer or a unique ID of the tensor as the key
        std::unordered_map<Tensor*, xt::xarray<double>> grad_map;

        // Seed the output/loss tensor with the initial gradient
        // (Assuming output_ is the final tensor of the network)
        grad_map[output_.get()] = loss_grad.data();

        // 4. Pass the gradients backward
        for (int32_t i = (int32_t)sorted_ops.size() - 1; i >= 0; i--) {
            auto& op = sorted_ops[i];

            // Collect upstream gradients for all outputs of this operator
            std::vector<Tensor> upstream_gradients;
            

            for (const std::shared_ptr<Tensor>& t_ptr : op->successors_) {
                Tensor* t_raw = t_ptr.get();
                auto it = grad_map.find(t_raw);
                if (it != grad_map.end()) {
                    upstream_gradients.push_back(Tensor(it->second));
                } 
                else {
                    upstream_gradients.push_back(Tensor(xt::zeros_like(t_ptr->data())));
                }
            }

            // Compute the backward pass for this operator
            std::vector<Tensor> input_gradients = op->compute_backwards_pass(upstream_gradients);

            // Route the resulting input gradients to the operator's predecessor tensors
            std::vector<Tensor> inputs;
            for(const std::shared_ptr<Tensor>& t_ptr : op->predecessors_) {
                inputs.push_back(*t_ptr);
            }

            assert(input_gradients.size() == inputs.size() && "Must provide one gradient per input");

            for (size_t j = 0; j < inputs.size(); ++j) {
                Tensor* pred_tensor = &inputs[j];
                
                // Accumulate gradients if the predecessor tensor already has gradients 
                // (handles branching / shared weights / multiple consumers)
                if (grad_map.find(pred_tensor) != grad_map.end()) {
                    grad_map[pred_tensor] += input_gradients[j].data();
                }
                else {
                    grad_map[pred_tensor] = input_gradients[j].data();
                }
            }
        }
    }
};



/**
 * (Sequentially defined) network
 */
class Network {
private:
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
     * @return result of forward pass wrapped in a Tensor
     */
    Tensor forward(Tensor input) {
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
        return *current_node[0];
    }


    /**
     * Computes the backward pass, initially using `predicted` and `expected`.
     * @param predicted network's prediction for a given input
     * @param expected what the network should have predicted for the input
     */
    void backward(Tensor predicted, Tensor expected) {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to computing backwards pass");
        }
        
        //Compute loss gradient and set it as the initial gradient
        Tensor loss_gradient = loss_calc_->compute_gradient(predicted, expected);
        std::vector<Tensor> current_gradients = {loss_gradient};

        //Check the output
        if(!output_) {
            throw invalid_config("No previous output found- must call forward pass prior to using backwards pass");
        }

        //Initialize the current output
        std::shared_ptr<Tensor> current_result = output_;
        

        if(!current_result) {
            throw invalid_config("The network's forward method must be called prior to computing a backwards pass");
        }

        //Cycle through each node and operator, backwards
        while(current_result != nullptr && current_result->prev_operator_ != nullptr) {
            std::shared_ptr<TensorOperator> current_operator = current_result->prev_operator_; 
            
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