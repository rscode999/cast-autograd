#ifndef CAST_NETWORK_
#define CAST_NETWORK_


#include "cast_exceptions.hpp"
#include "control_flow.hpp"
#include "activation_function.hpp"
#include "loss_calculator.hpp"
#include "optimizer.hpp"
#include "tensor_operator.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <xtensor/containers/xarray.hpp>




namespace cast {

/**
* Signals that a branch index has been merged with another branch
*/
const int32_t NETWORK_BRANCH_COMBINED = -1000;



/**
 * Neural network with trainable weights.
 * Operators (i.e. layers, activation functions) are added to the network individually.
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
    * Index of the last operator in the network.
    *
    * Determined after the `enable` check
    */
    int32_t output_index_;

 
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
    * @return number of branches currently in the network
    */
    int32_t active_branches() const {
        return (int32_t)leaf_node_indices_.size();
    }



    /**
    * @return 0-based indices of the ends of each branch
    */
    std::vector<int32_t> active_branch_indices() const {
        return leaf_node_indices_;
    }

    

    /**
     * Registers `op` as the next operator to execute in the network
     * @param op new operator to add. Non-null
     * @param branch_index 0-based index where the operator should be added
     * @throws `std::logic_error` if the branch index is out of range
     */
    void add_operator(std::shared_ptr<TensorOperator> op, int32_t branch_index = 0) {
        str_assert(op != nullptr, "New operator cannot be nullptr");

        operators_.push_back(op);

        //First operator loaded: Add the current node as an output
        if(leaf_node_indices_.size() == 0) {
            leaf_node_indices_.push_back(0);
        }
        //Handle branch index out of range
        else if(branch_index < 0 || branch_index >= leaf_node_indices_.size()) {
            throw std::logic_error("Branch index out of range (max index " + std::to_string(leaf_node_indices_.size() - 1) + ", received " + std::to_string(branch_index) + ")");
        }
        //Handle branch that has already been combined
        else if(leaf_node_indices_[branch_index] == NETWORK_BRANCH_COMBINED) {
            throw std::logic_error("Cannot add to branch " + std::to_string(branch_index) + ", which has been combined with another branch");
        }
        //Otherwise: The currently added leaf node index is incremented
        else {
            //Load the recently added operator's predecessor index
            op->predecessors_.clear();
            op->predecessors_.push_back(leaf_node_indices_[branch_index]);

            //Handle combiners (separate function only)
            if(std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(op)) {
                throw not_implemented("Use add_combiner instead");
            }
            //Handle branches
            else if(std::shared_ptr<Branch> branch = std::dynamic_pointer_cast<Branch>(op)) {
                //Add new possible branches, marking the branch's index as successors
                int32_t branch_add_index = (int32_t)operators_.size() - 1;
                for(int32_t i = 0; i < branch->branch_count() - 1; i++) {
                    leaf_node_indices_.push_back(branch_add_index);
                }
            }

            //Register recently added node as the current branch leaf node's successor
            operators_[leaf_node_indices_[branch_index]]->successors_.push_back( (int32_t)operators_.size() - 1 );
            leaf_node_indices_[branch_index] = (int32_t)operators_.size() - 1;
        }
    }



    void add_combiner(std::shared_ptr<Combiner> combiner, int32_t branch_index = 0) {
        operators_.push_back(combiner);

        //check if indices used are valid
        if(combiner->branch_indices().size() > (int32_t)leaf_node_indices_.size()) {
            throw std::logic_error("Number of branches in combiner must be less than the number of branches in the network"); //TODO: Replace with Combiner-specific exception
        }
        for (int i = 0; i < (int32_t)combiner->branch_indices().size(); i++)  {
            if(combiner->branch_indices() [i] < 0 || combiner->branch_indices() [i] >= (int32_t)leaf_node_indices_.size()) {
                throw std::logic_error("Branch index out of range"); //TODO: Replace with Combiner-specific exception
            }
            if(leaf_node_indices_[combiner->branch_indices() [i]] < 0) {
                throw std::logic_error("Branch " + std::to_string(combiner->branch_indices() [i]) + " has been merged, so it no longer exists");
            }
            if(combiner->branch_indices() [i] == branch_index) {
                throw std::logic_error("Cannot merge the branch that the combiner is added to"); //TODO: Replace with Combiner-specific exception
            }
        }

        
        int32_t combiner_add_index = (int32_t)operators_.size() - 1;
        for(int32_t i = 0; i < (int32_t)combiner->branch_indices().size(); i++) {
            //Set all branches in the combiner's list to be predecessors to the new combiner
            combiner -> predecessors_.push_back(leaf_node_indices_[ combiner->branch_indices()[i] ]);
            
            //Set the combiner to be successors to all branches in the combiner's list
            operators_[leaf_node_indices_[ combiner->branch_indices()[i] ]] -> successors_ .push_back(combiner_add_index);
        }
        //Do the same with the combiner's own branch leaves
        combiner -> predecessors_.push_back(leaf_node_indices_[ branch_index ]);
        operators_[leaf_node_indices_[ branch_index ]] -> successors_ .push_back(combiner_add_index);
        
        //"Remove" all combined branches (perhaps use a hash map in the future)
        for(int32_t l = 0; l < (int32_t)leaf_node_indices_.size(); l++) {
            for(int32_t c = 0; c < (int32_t)combiner->branch_indices().size(); c++) {
                //Leaf node index equals one that was combined: Take it out
                if(l == combiner->branch_indices() [c]) {
                    leaf_node_indices_[ combiner->branch_indices() [c] ] = NETWORK_BRANCH_COMBINED;
                }
            }
            
        }

        //Log the operator's branch
        combiner->branch_id_ = branch_index;
        leaf_node_indices_[branch_index] = (int32_t)operators_.size() - 1;
    }


    /**
     * Sets this network's loss calculator to `calc`.
     * @param calc new loss calculator to use. Non-null
     */
    void set_loss_calculator(std::shared_ptr<LossCalculator> calc) {
        str_assert(calc != nullptr, "New loss calculator must be non-null");

        //Reset the loss calculator if it exists
        if(loss_calc_) {
            loss_calc_.reset();
        }

        //Create deep pointer of the new calculator (OK for now- LossCalculators have no internal state)
        loss_calc_ = calc;
    }



    /**
     * Sets this network's optimizer to `optim`.
     * @param optim new optimizer to use. Non-null
     */
    void set_optimizer(std::shared_ptr<Optimizer> optim) {
        str_assert(optim != nullptr, "New optimizer must be non-null");

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

        //Check that the network has operators
        if((int32_t)leaf_node_indices_.size() < 1) {
            throw invalid_config("Network must have at least one operator");
        }

        //Check that the network's first element is not a branch
        if(std::dynamic_pointer_cast<Branch>(operators_[0]) != nullptr) {
            throw invalid_config("First operator in the network cannot be a branch");
        }

        

        // //Check for single output, while also logging the output's index in the operators list
        // int32_t output_count = 0;
        // for(int32_t i = 0; i < (int32_t)leaf_node_indices_.size(); i++) {
        //     if(leaf_node_indices_[i] != NETWORK_BRANCH_COMBINED) {
        //         output_index_ = leaf_node_indices_[i];
        //         output_count++;
        //     }
        // }
        // if(output_count != 1) {
        //     throw invalid_config("Network must have exactly one output");
        // }

        for(std::shared_ptr<TensorOperator> op : operators_) {
            std::cout << op->name() << " ";
            std::cout << "predecessors: ";
            for(int32_t successor_idx : op->predecessors_) {
                std::cout << successor_idx << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        for(std::shared_ptr<TensorOperator> op : operators_) {
            std::cout << op->name() << " ";
            std::cout << "successors: ";
            for(int32_t successor_idx : op->successors_) {
                std::cout << successor_idx << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
        std::cout << "LEAF NODE INDICES" << std::endl;
        for(int32_t l : leaf_node_indices_) {
            std::cout << l << ", ";
        }
        std::cout << std::endl;

        optimizer_->initialize(operators_);

        enabled_ = true;
    }



    /**
     * Returns the result of the network's forward pass on `input`.
     *
     * To use this method, the network must be enabled. 
     * @param input tensor to compute forward pass on
     * @return result of forward pass
     */
    xt::xarray<double> forward(xt::xarray<double> input) {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to training");
        }

        //Cycle the input through the operators
        std::vector<std::vector<xt::xarray<double>>> current_outputs = {{input}}; //Each index is the output(s) of a branch. Layers can have many outputs
        std::vector<int32_t> next_operator_indices = {0}; //Each index is the next operator to compute

        //PRECONDITION: There can be one, and exactly one, operator with no successors
        bool successors_remaining = true;
        while(successors_remaining) {

            //DEBUG
            std::cout << "Current Outputs" << "\n";
            for (int c = 0; c < current_outputs.size(); c++) {
                for (int o = 0; o < current_outputs[c].size(); o++) {
                    std::cout << current_outputs[c][o] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "Next Op Indices" << "\n";
            for(int32_t idx : next_operator_indices) {
                std::cout << idx << ", ";
            }
            std::cout << "\n" << std::endl;

            
            //Compute one operation for each currently calculating branch
            int32_t current_operator_amount = (int32_t)next_operator_indices.size();
            for(int32_t op = 0; op < current_operator_amount; op++) {
     
                //The branch is empty: Continue
                if(next_operator_indices[op] == NETWORK_BRANCH_COMBINED) {
                    continue;
                }


                //Handle branches
                if(std::shared_ptr<Branch> branch = std::dynamic_pointer_cast<Branch>(operators_[next_operator_indices[op]])) {

                    std::vector<std::vector<xt::xarray<double>>> branch_outputs = branch->compute({{current_outputs[branch->branch_id_]}}, true);
                    //Add all the new branch outputs
                    current_outputs[0] = branch_outputs[0];
                    for(int32_t b = 1; b < (int32_t)branch_outputs.size(); b++) {
                        current_outputs.push_back(branch_outputs[b]);
                    }

                    std::vector<int32_t> successor_indices = branch->successors();
                    //Update next to check
                    // for(int32_t b = 0; b < (int32_t)successor_indices.size(); b++) {
                    //     std::cout << successor_indices[b] << ", ";
                    // }
                    // std::cout << std::endl;
                    next_operator_indices[0] = successor_indices[0];
                    for(int32_t b = 1; b < (int32_t)successor_indices.size(); b++) {
                        next_operator_indices.push_back(successor_indices[b]);
                    }
                }
                //Handle combiners
                else if(std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(operators_[next_operator_indices[op]])) {
                    std::vector<xt::xarray<double>> combiner_output = combiner->compute(current_outputs[op]);

                    //Set the currently executing branch as waiting for the combiner to finish
                    next_operator_indices[op] = NETWORK_BRANCH_COMBINED;
                    current_outputs[op] = {};
                    

                    //The Combiner received all its inputs: The Combiner returns a non-empty vector
                    if(combiner_output.size() != 0) {
                        //Save combiner output
                        current_outputs[combiner->branch_id_] = combiner_output;

                        //Combiner has no successors: Terminate. The network can have exactly one output.
                        if(combiner->successors() .size() == 0) {
                            successors_remaining = false;
                            current_outputs[0] = combiner_output;
                            break;
                        }

                        //Assign successor to take the output
                        next_operator_indices[op] = combiner->successors() [0];
                    }
                }
                //Handle single layers
                else {
                    //Get result
                    current_outputs[op] = operators_[next_operator_indices[op]] -> compute(current_outputs[op]); //note: this is a single layer that may have multiple outputs
                    
                    //Next operator is a leaf: There can be only one network output, so exit the entire operation
                    if(operators_[next_operator_indices[op]] -> successors_.size() == 0) {
                        successors_remaining = false;
                        break;
                    }

                    //Update next to check (Non-branches have exactly one successor, stored in index 0)
                    next_operator_indices[op] = operators_[next_operator_indices[op]] -> successors_[0];
                }
            }
        }
        
        output_ = current_outputs[0][0];
        return current_outputs[0][0];
    }



    /**
     * Computes the backward pass, initially using `predicted` and `expected`.
     *
     * Stores updated gradients inside each operator, for use by the network's optimizer.
     *
     * The network must be enabled to use this method.
     * @param predicted network's prediction for a given input
     * @param expected what the network should have predicted for the input
     */
    void backward(xt::xarray<double> predicted, xt::xarray<double> expected) {

        //Compute initial loss
        xt::xarray<double> output_loss = loss_calc_->compute_gradient(predicted, expected);

        std::vector<std::vector<xt::xarray<double>>> current_branch_gradients = {{output_loss}}; //Each index is the gradient(s) of a branch. Layers can have many outputs
        std::vector<int32_t> prev_operator_indices = {output_index_};

        //PRECONDITION: There can be one, and EXACTLY ONE, operator with no predecessors
        bool predecessors_remaining = true;
        while(predecessors_remaining && !prev_operator_indices.empty()) {


            //Handle branches
            if(false) {
            }
            //Handle single layers
            else {
                //Move to the previous operator, moving gradients backward
                for(int32_t i = 0; i < (int32_t)prev_operator_indices.size(); i++) {

                    //Propagate gradients, one operator down in the branch
                    current_branch_gradients[i] = operators_[prev_operator_indices[i]] -> compute_backwards_pass(current_branch_gradients[i]);

                    //The ONE operator with no predecessors has just been backpropagated: Exit
                    if(operators_[prev_operator_indices[i]]->predecessors_.size() == 0) {
                        predecessors_remaining = false;
                        break;
                    }

                    //Single layers have exactly one predecessor. Move there.
                    prev_operator_indices[i] = operators_[prev_operator_indices[i]] -> predecessors_[0];
                }
            }
        }
    }



    /**
     * Runs an optimization pass on the network's layers, using the optimizer and gradients computed from the `backward` method.
     *
     * To use this method, the network must be enabled.
     *
     * WARNING: Calling `optimize` multiple times, without computing a `backward` operation prior,
     * wil cause the network to use its stored gradients multiple times.
     * @param zero_grad whether to set all operator's gradients to 0 after computing the optimization pass
     */
    void optimize(bool zero_grad = true) {
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to optimizing"); 
        }

        optimizer_->step(zero_grad);
        output_ = xt::zeros_like(output_);
    }

};




}
#endif