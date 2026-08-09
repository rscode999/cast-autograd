#ifndef CAST_NETWORK_
#define CAST_NETWORK_


#include "cast_exceptions.hpp"
#include "activation_function.hpp"
#include "loss_calculator.hpp"
#include "optimizer.hpp"
#include "tensor_graph/tensor_graph.hpp"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <xtensor/containers/xarray.hpp>



namespace cast {



/**
* Index that signals that there are no more operators in a branch, and that the branch is waiting for other branches to finish executing
*/
const int32_t BRANCH_END = -1000;


/**
 * (Sequentially defined) network
 */
class Network {
protected:
    /**
     * Operators that the network uses, in order
     */
    std::vector<std::shared_ptr<TensorOperator>> operators_;

    /**
    * Indices in `operators_` that are leaf nodes, i.e. have no successors.
    *
    * Leaf nodes are the only nodes that can be added to.
    * The length of this vector is the current number of branches.
    */
    std::vector<int32_t> leaf_node_indices_;

    /**
     * Holds the tensor first given to this network
     */
    xt::xarray<double> initial_input_;

    /**
     * Holds the most recent output to this network
     */
    xt::xarray<double> output_;

 
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
    void add_operator(std::shared_ptr<TensorOperator> op, int32_t branch_index = 0) {
        operators_.push_back(op);

        //First operator loaded: Add the current node as an output
        if(leaf_node_indices_.size() == 0) {
            leaf_node_indices_.push_back(0);
        }
        //Handle branch index out of range
        else if(branch_index < 0 || branch_index >= leaf_node_indices_.size()) {
            throw std::logic_error("Branch index out of range");
        }
        //Otherwise: The currently added leaf node index is incremented
        else {
            //Load the recently added operator's predecessor index
            op->predecessors_.clear();
            op->predecessors_.push_back(leaf_node_indices_[branch_index]);

            //Register recently added node as the current branch leaf node's successor
            operators_[leaf_node_indices_[branch_index]]->successors_.push_back( (int32_t)operators_.size() - 1 );

            leaf_node_indices_[branch_index] = (int32_t)operators_.size() - 1;
        }
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

        // for(std::shared_ptr<TensorOperator> op : operators_) {
        //     std::cout << op->name() << " ";
        //     std::cout << "successors: ";
        //     for(int32_t successor_idx : op->successors_) {
        //         std::cout << successor_idx << ", ";
        //     }
        //     std::cout << std::endl;
        // }

        optimizer_->initialize(operators_);

        enabled_ = true;
    }


    /**
     * Returns the result of the network's forward pass on `input`
     * 
     * @param input tensor to compute forward pass on
     * @return result of forward pass
     */
    xt::xarray<double> forward(xt::xarray<double> input) {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to training");
        }

        //Cycle the input through the operators
        std::vector<std::vector<xt::xarray<double>>> current_branch_outputs = {{input}}; //Each index is the output(s) of a branch. Layers can have many outputs
        std::vector<int32_t> next_operator_indices = {0}; //Each index is the next operator to compute

        //PRECONDITION: There can be one, and exactly one, operator with no successors
        bool successors_remaining = true;
        while(successors_remaining) {

            std::cout << "Computing operator ";
            for(int32_t op_index : next_operator_indices) {
                std::cout << op_index << ", ";
            }
            std::cout << std::endl;
            
            //Compute one operation for each branch
            for(int32_t i = 0; i < (int32_t)next_operator_indices.size(); i++) {
                //Handle branch starts and ends
                if(false) { //If the current operator is a branch
                }
                //Handle single inputs and outputs
                else {
                    //Get result
                    std::cout << operators_[next_operator_indices[i]] -> name() << std::endl;
                    current_branch_outputs[i] = operators_[next_operator_indices[i]] -> compute(current_branch_outputs[i]);
                    
                    //Next operator is a leaf: There can be only one network output, so exit the entire operation
                    if(operators_[next_operator_indices[i]] -> successors_.size() == 0) {
                        successors_remaining = false;
                        break;
                    }

                    //Update next to check (Non-branches have exactly one successor, stored in index 0)
                    next_operator_indices[i] = operators_[next_operator_indices[i]] -> successors_[0];
                }

            }
        }


        return current_branch_outputs[0][0];
    }


    /**
     * Computes the backward pass, initially using `predicted` and `expected`.
     * @param predicted network's prediction for a given input
     * @param expected what the network should have predicted for the input
     */
    void backward(xt::xarray<double> predicted, xt::xarray<double> expected) {
        
    }


    /**
     * Runs an optimization pass on the network's layers, using the stored optimizer
     */
    void optimize() {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to optimizing"); 
        }

        optimizer_->step(operators_, true);
        output_ = xt::zeros_like(output_);
    }

};




}
#endif