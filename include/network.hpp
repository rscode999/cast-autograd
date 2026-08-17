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
#include <queue>
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
    std::vector<std::shared_ptr<NetworkComponent>> operators_;

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
    * Number of branches created during the forward pass
    */
    int32_t max_branch_index_;

 
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
    void add_operator(std::shared_ptr<NetworkComponent> op, int32_t branch_index = 0) {
        str_assert(op != nullptr, "New operator cannot be nullptr");

        operators_.push_back(op);
        op->branch_id_ = branch_index;

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
            //Load the recently added operator's predecessor branch and index
            op->predecessors_.clear();
            op->predecessors_[operators_[leaf_node_indices_[branch_index]]->branch_id_] = leaf_node_indices_[branch_index];

            //Handle combiners (separate function only)
            if(std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(op)) {
                throw not_implemented("Use add_combiner instead");
            }
            //Handle branches
            else if(std::shared_ptr<Splitter> branch = std::dynamic_pointer_cast<Splitter>(op)) {
                //Add new possible branches, marking the branch's index as successors
                int32_t branch_add_index = (int32_t)operators_.size() - 1;
                for(int32_t i = 0; i < branch->branch_count() - 1; i++) {
                    leaf_node_indices_.push_back(branch_add_index);
                }
            }

            //Register recently added node as the current branch leaf node's successor
            operators_[leaf_node_indices_[branch_index]]->successors_[branch_index] = (int32_t)operators_.size() - 1;
            leaf_node_indices_[branch_index] = (int32_t)operators_.size() - 1;
        }
    }



    void add_combiner(std::shared_ptr<Combiner> combiner, int32_t branch_index = 0) {
        operators_.push_back(combiner);
        combiner->branch_id_ = branch_index;

        //Verify branch index
        if(branch_index < 0 || branch_index >= (int32_t)leaf_node_indices_.size()) {
            throw std::logic_error("Branch index out of range");
        }
        // Check if indices used are valid
        if(combiner->branch_indices().size() > (int32_t)leaf_node_indices_.size()) {
            throw std::logic_error("Number of branches in combiner must be less than the number of branches in the network"); //TODO: Replace with Combiner-specific exception
        }
        for (int i = 0; i < (int32_t)combiner->branch_indices().size(); i++)  {
            if(combiner->branch_indices()[i] < 0 || combiner->branch_indices()[i] >= (int32_t)leaf_node_indices_.size()) {
                throw std::logic_error("Branch index out of range"); //TODO: Replace with Combiner-specific exception
            }
            if(leaf_node_indices_[combiner->branch_indices()[i]] == NETWORK_BRANCH_COMBINED) {
                throw std::logic_error("Branch " + std::to_string(combiner->branch_indices()[i]) + " has been merged, so it no longer exists");
            }
            if(combiner->branch_indices()[i] == branch_index) {
                throw std::logic_error("Cannot merge the branch that the combiner is added to"); //TODO: Replace with Combiner-specific exception
            }
        }

        int32_t combiner_node_index = (int32_t)operators_.size() - 1;
        combiner->predecessors_.clear();

        // 1. Add the branch that the combiner is added to into predecessors
        int32_t target_leaf_idx = leaf_node_indices_[branch_index];
        int32_t target_branch_id = operators_[target_leaf_idx]->branch_id_;
        combiner->predecessors_[target_branch_id] = target_leaf_idx;
        operators_[target_leaf_idx]->successors_[target_branch_id] = combiner_node_index;

        // 2. Add all other combiner branch indices into predecessors and mark them as combined
        for (int32_t combiner_branch_idx : combiner->branch_indices()) {
            int32_t leaf_idx = leaf_node_indices_[combiner_branch_idx];
            int32_t branch_id = operators_[leaf_idx]->branch_id_;

            combiner->predecessors_[branch_id] = leaf_idx;
            operators_[leaf_idx]->successors_[branch_index] = combiner_node_index;

            // Mark branch as combined
            leaf_node_indices_[combiner_branch_idx] = NETWORK_BRANCH_COMBINED;
        }

        // 3. Update the leaf node index for the target branch to point to the combiner
        leaf_node_indices_[branch_index] = combiner_node_index;
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

        //Check that the network's first element is not a splitter
        if(std::dynamic_pointer_cast<Splitter>(operators_[0]) != nullptr) {
            throw invalid_config("First operator in the network cannot be a splitter");
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

        // for(std::shared_ptr<NetworkComponent> op : operators_) {
        //     std::cout << op->name() << " ";
        //     std::cout << "predecessors: ";
        //     for(std::pair<int32_t, int32_t> p : op->predecessors_) {
        //         std::cout << p.first << ", " << p.second << ";   ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "\n";
        // for(std::shared_ptr<NetworkComponent> op : operators_) {
        //     std::cout << op->name() << " ";
        //     std::cout << "successors: ";
        //       for(std::pair<int32_t, int32_t> p : op->successors_) {
        //         std::cout << p.first << ", " << p.second << ";   ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "\n";
        // std::cout << "LEAF NODE INDICES" << std::endl;
        // for(int32_t l : leaf_node_indices_) {
        //     std::cout << l << ", ";
        // }
        // std::cout << std::endl;

        optimizer_->initialize(operators_);

        max_branch_index_ = 0;
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

        struct Task {
            int32_t branch_id;
            int32_t op_idx;
            std::vector<xt::xarray<double>> data;
        };

        std::queue<Task> execution_queue;
        execution_queue.push({0, 0, {input}});

        while(!execution_queue.empty()) {
            Task current = execution_queue.front();
            execution_queue.pop();

            int32_t branch_id = current.branch_id;
            int32_t op_idx = current.op_idx;

            if (op_idx == NETWORK_BRANCH_COMBINED) {
                continue;
            }

            std::shared_ptr<NetworkComponent> current_op = operators_.at(op_idx);
            // std::cout << "Executing " << current_op->name() << std::endl;

            // Handle splitters
            if (std::shared_ptr<Splitter> splitter = std::dynamic_pointer_cast<Splitter>(current_op)) {
                std::vector<std::vector<xt::xarray<double>>> branch_output = splitter->compute(current.data, true);

                auto successors = splitter->successors();
                str_assert(!successors.empty(), "Branch must have at least one successor");

                auto succ_it = successors.begin();

                //Push all successors to the queue
                for(size_t out_idx = 0; succ_it != successors.end(); ++succ_it, ++out_idx) {
                    int32_t succ_branch_id = succ_it->first;
                    int32_t succ_op_idx = succ_it->second;
                    std::vector<xt::xarray<double>> out_data = branch_output[out_idx < branch_output.size() ? out_idx : 0];
                    execution_queue.push({succ_branch_id, succ_op_idx, out_data});
                }
            }
            // Handle combiners
            else if(std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(current_op)) {
                std::vector<xt::xarray<double>> combiner_output = combiner->compute(current.data);
                
                // Combiner output is non-empty only when all required inputs have arrived
                if(!combiner_output.empty()) {
                    // std::cout << "Combiner is ready" << std::endl;
                    
                    //No successors: Return
                    if(combiner->successors().empty()) {
                        // std::cout << "COMBINER HAS NO SUCCESSORS" << std::endl;
                        output_ = combiner_output[0];
                        return combiner_output[0];
                    }

                    auto const& succs = combiner->successors();
                    str_assert(succs.size() == 1, "Combiner must have one successor");
                    
                    int32_t target_branch_id = succs.begin()->first;
                    int32_t target_op_idx = succs.begin()->second;

                    execution_queue.push({target_branch_id, target_op_idx, combiner_output});
                }
            }
            // Handle single operator
            else {
                std::vector<xt::xarray<double>> op_output = current_op->compute(current.data);

                //No successors: Return (this is the single operator with no successors)
                if(current_op->successors().empty()) {
                    output_ = op_output[0];
                    return op_output[0];
                }

                auto const& succs = current_op->successors();
                str_assert(succs.size() == 1, "Operator must have exactly one successor");
                
                int32_t target_branch_id = succs.begin()->first;
                int32_t target_op_idx = succs.begin()->second;

                execution_queue.push({target_branch_id, target_op_idx, op_output});
            }
        }

        throw std::runtime_error("Forward pass finished without reaching an output node.");
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
        if(!enabled_) {
            throw invalid_config("Must enable the network prior to training");
        }

        xt::xarray<double> output_loss = loss_calc_->compute_gradient(predicted, expected);

        struct Task {
            int32_t branch_id;
            int32_t op_idx;
            std::vector<xt::xarray<double>> data;
        };

        std::queue<Task> execution_queue;
        int32_t output_op_idx = (int32_t)operators_.size() - 1;
        int32_t output_branch_id = operators_[output_op_idx]->branch_id_;
        
        execution_queue.push({output_branch_id, output_op_idx, {output_loss}});

        while(!execution_queue.empty()) {
            Task current = execution_queue.front();
            execution_queue.pop();

            int32_t branch_id = current.branch_id;
            int32_t op_idx = current.op_idx;

            if (op_idx == NETWORK_BRANCH_COMBINED) {
                continue;
            }

            std::shared_ptr<NetworkComponent> current_op = operators_.at(op_idx);

            // Handle splitters (act like combiners in the backwards pass, collecting inputs)
            if (std::shared_ptr<Splitter> splitter = std::dynamic_pointer_cast<Splitter>(current_op)) {
                std::vector<xt::xarray<double>> branch_grads = splitter->compute_backwards_pass(current.data);

                // Branch output is non-empty only when all required inputs have arrived
                if (!branch_grads.empty()) {
                    auto const& preds = splitter->predecessors();
                    if (preds.empty()) {
                        return;
                    }

                    // std::cout << "Output: " << branch_grads[0] << std::endl;
                    
                    str_assert(preds.size() == 1, "Splitter must have exactly one predecessor");
                    auto pred_it = preds.begin();
                    execution_queue.push({pred_it->first, pred_it->second, branch_grads});
                }
            }
            // Handle combiners (act like splitters in the backwards pass, distributing gradients)
            else if (std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(current_op)) {
                std::vector<std::vector<xt::xarray<double>>> combiner_outputs = combiner->compute_backwards_pass(current.data, true);

                auto preds = combiner->predecessors();
                str_assert(!preds.empty(), "Combiner must have at least one predecessor");

                auto pred_it = preds.begin();
                for (size_t out_idx = 0; pred_it != preds.end(); ++pred_it, ++out_idx) {
                    int32_t pred_branch_id = pred_it->first;
                    int32_t pred_op_idx = pred_it->second;
                    std::vector<xt::xarray<double>> out_data = combiner_outputs[out_idx < combiner_outputs.size() ? out_idx : 0];
                    execution_queue.push({pred_branch_id, pred_op_idx, out_data});
                }
            }
            // Handle single operators
            else {
                std::vector<xt::xarray<double>> op_output = current_op->compute_backwards_pass(current.data);

                auto const& preds = current_op->predecessors();
                if (preds.empty()) {
                    return;
                }

                str_assert(preds.size() == 1, "Operator must have exactly one predecessor");
                auto pred_it = preds.begin();
                int32_t pred_branch_id = pred_it->first;
                int32_t pred_op_idx = pred_it->second;

                execution_queue.push({pred_branch_id, pred_op_idx, op_output});
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