#ifndef CAST_NETWORK_
#define CAST_NETWORK_


#include "activation_function.hpp"
#include "cast_exceptions.hpp"
#include "control_flow.hpp"
#include "loss_calculator.hpp"
#include "network_component.hpp"
#include "optimizer.hpp"

#include <xtensor/containers/xarray.hpp>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <queue>
#include <string>



namespace cast {




/**
* Signals that a branch index has been merged with another branch
*/
const int32_t NETWORK_BRANCH_COMBINED = -256;



/**
 * Neural network with trainable weights.
 * Components (i.e. layers, activation functions) are added to the network individually.
 */
class Network {
private:

   /**
     * True if the network is ready for training and evaluation
     */
    bool enabled_;
    
    /**
     * Loss metric used by this network
     */
    std::shared_ptr<LossCalculator> loss_calc_;

    /**
     * Optimizer used by this network
     */
    std::shared_ptr<Optimizer> optimizer_;

    /**
    * Indices in `components_` that are leaf nodes, i.e. have no successors. Element `i` is the leaf node index for branch `i`.
    *
    * Leaf nodes are the only nodes that can be added to.
    * The length of this vector is the total number of branches used in this network, whether active or combined with another branch.
    *
    * Element `i` equaling `NETWORK_BRANCH_COMBINED` indicates that branch `i` has been combined, and no longer exists.
    * (This vector is never removed from.)
    */
    std::vector<int32_t> leaf_node_indices_;

    /**
     * Network components that the network uses, in the order that they were added
     */
    std::vector<std::shared_ptr<NetworkComponent>> components_;

 
    /**
    * Throws `bad_component_addition` if `combine_branch_ids` and `branch_id` are incompatible with each other or the current network configuration.
    * Called when a network component is added.
    *
    * The network must not be enabled: otherwise, throws `cast::invalid_config`.
    *
    * `branch_id` must be positive and less than the number of total branches used.
    *
    * A Combiner must have each element of `combine_branch_ids` non-negative and less than the number of total branches used.
    * `branch_id` must not equal any element from `combine_branch_ids`.
    *
    * `leaf_node_indices[branch_id or element from combine_branch_ids]` must not equal `NETWORK_BRANCH_COMBINED`. If so, a component will be added to a branch that no longer exists.
    *
    * @param combine_branch_ids branch IDs to be combined (in the case of a Combiner); if not a Combiner, this parameter is empty
    * @param branch_id branch ID that the component wil be added to
    * @param check_location location where this check is carried out; for debugging purposes
    */
    void check_component_indices_(std::initializer_list<int32_t> combine_branch_ids, int32_t branch_id, std::source_location check_location = std::source_location::current()) {
        //Enable check
        if(enabled_) {
            throw invalid_config("Network must not be enabled to add components", check_location);
        }
        
        //Stop check early if there are no leaf nodes (that is, the component is the first one added)
        if (leaf_node_indices_.empty()) {
            return;
        }

        int32_t leaf_node_indices_size = static_cast<int32_t>(leaf_node_indices_.size());

        // Check branch_id in range
        if(branch_id < 0 || branch_id >= leaf_node_indices_size) {
            throw bad_component_addition(
                "Branch ID " + std::to_string(branch_id) + " must be on the interval [0, " + std::to_string(leaf_node_indices_size - 1) + "]",
                check_location
            );
        }
        // Check that the branch ID given has not been combined
        if (leaf_node_indices_[branch_id] == NETWORK_BRANCH_COMBINED) {
            throw bad_component_addition(
                "Branch " + std::to_string(branch_id) + " has already been combined and does not exist",
                check_location
            );
        }

        //Return early if there are no combine indices
        if(combine_branch_ids.size() == 0) {
            return;
        }

        // Check combine indices
        size_t pos = 0;
        for (int32_t combine_index : combine_branch_ids) {
            //Size bounds
            if (combine_index < 0 || combine_index >= leaf_node_indices_size) {
                throw bad_component_addition(
                    "Combine indices index " + std::to_string(pos) + " (" + std::to_string(combine_index) +
                    ") must be non-negative and less than the total number of branches created (" + std::to_string(leaf_node_indices_size) + ")",
                    check_location
                );
            }
            //Cannot add to a branch that has been combined
            if (leaf_node_indices_[combine_index] == NETWORK_BRANCH_COMBINED) {
                throw bad_component_addition(
                    "Combine indices index" + std::to_string(pos) + " (" + std::to_string(combine_index) +
                    ") points to a branch that has already been combined",
                    check_location
                );
            }
            //Cannot assign to its own branch
            if (combine_index == branch_id) {
                throw bad_component_addition(
                    "Combine index " + std::to_string(pos) + " cannot equal the branch that the combiner is added to (" + std::to_string(branch_id) + ")",
                    check_location
                );
            }
            ++pos;
        }
    }



public:
    /**
     * Creates a new network
     */
    Network() : enabled_(false) {      
    };




    /**
    * Returns the 0-based indices of the ends of each branch in the internal component storage.
    *
    * The output's length is equal to the total number of branches used in the network so far (but the branches may not necessarily still exist).
    * 
    * Index `i` equals the constant `NETWORK_BRANCH_COMBINED` if branch `i` has been combined with another branch, and thus no longer exists.
    * @return indices of leaf nodes
    */
    std::vector<int32_t> active_branch_indices() const {
        return leaf_node_indices_;
    }



    /**
    * @return whether the network is ready for training and optimization
    */
    bool enabled() const {
        return enabled_;
    }
    

    
    /**
    * Adds a combiner, merging the branch IDs given in `branch_indices_to_combine`, to branch `branch_id`.
    *
    * If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, 
    * this method throws `cast::bad_component_addition`.
    *
    * `cast::bad_component_addition` is also thrown if any of the branch IDs in `branch_ids_to_combine` is out of the range [0, <number of branches used in the network - 1>],
    * has already been merged, or equals `branch_id` (combiners cannot merge their own branch).
    * A Combiner cannot be the first component added to a network.
    *
    * @param branch_indices_to_combine list of branch IDs to merge
    * @param branch_id branch to add the new splitter to
    * @param loc location where this method is called (for debugging purposes)
    */
    void add_combiner(std::initializer_list<int32_t> branch_ids_to_combine, int32_t branch_id = 0, std::source_location loc = std::source_location::current()) {
        check_component_indices_(branch_ids_to_combine, branch_id, loc);

        //First operator loaded: Not allowed
        if(leaf_node_indices_.size() == 0) {
            throw bad_component_addition("Cannot add a combiner as the first component of a network", loc);
        }

        std::shared_ptr<Combiner> combiner = std::make_shared<Combiner>(branch_ids_to_combine);
        components_.push_back(combiner);
        combiner->branch_id_ = branch_id;

        
        int32_t combiner_node_index = (int32_t)components_.size() - 1;
        combiner->predecessors_.clear();

        // 1. Add the branch that the combiner is added to into predecessors
        int32_t target_leaf_idx = leaf_node_indices_[branch_id];
        int32_t target_branch_id = components_[target_leaf_idx]->branch_id_;
        combiner->predecessors_[target_branch_id] = target_leaf_idx;
        components_[target_leaf_idx]->successors_[target_branch_id] = combiner_node_index;

        // 2. Add all other combiner branch indices into predecessors and mark them as combined
        for (int32_t combiner_branch_idx : combiner->branch_indices()) {
            int32_t leaf_idx = leaf_node_indices_[combiner_branch_idx];
            int32_t branch_id = components_[leaf_idx]->branch_id_;

            combiner->predecessors_[branch_id] = leaf_idx;
            components_[leaf_idx]->successors_[branch_id] = combiner_node_index;

            // Mark branch as combined
            leaf_node_indices_[combiner_branch_idx] = NETWORK_BRANCH_COMBINED;
        }

        // 3. Update the leaf node index for the target branch to point to the combiner
        leaf_node_indices_[branch_id] = combiner_node_index;
    }


    
    /**
    * Adds `op` to the end of branch `branch_id`.
    *
    * If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, 
    * this method throws `cast::bad_component_addition`.
    * @param new_operator operator to add to a branch
    * @param branch_id branch to add the new splitter to
    * @param loc location where this method is called (for debugging purposes)
    */
    void add_operator(std::shared_ptr<Operator> new_operator, int32_t branch_id = 0, std::source_location loc = std::source_location::current()) {
        check_component_indices_({}, branch_id, loc);

        //Make a deep copy of the operator
        std::shared_ptr<NetworkComponent> op = new_operator->shared_ptr_deep_copy();

        //Register the operator
        components_.push_back(op);
        op->branch_id_ = branch_id;

        //First operator loaded: Add the current node as an output
        if(leaf_node_indices_.size() == 0) {
            leaf_node_indices_.push_back(0);
            return;
        }

        // Set predecessor
        op->predecessors_.clear();
        op->predecessors_[components_[leaf_node_indices_[branch_id]]->branch_id_] = leaf_node_indices_[branch_id];

        // Register recently added node as the current branch leaf node's successor
        components_[leaf_node_indices_[branch_id]]->successors_[branch_id] = (int32_t)components_.size() - 1;
        leaf_node_indices_[branch_id] = (int32_t)components_.size() - 1;
    }



    /**
    * Adds a splitter that distributes execution across `branch_count` new branches, to branch `branch_id`.
    *
    * If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, 
    * this method throws `cast::bad_component_addition`.
    * @param branch_count number of branches to split execution into. At least 2.
    * @param branch_id branch to add the new splitter to
    * @param loc location where this method is called (for debugging purposes)
    */
    void add_splitter(int32_t branch_count, int32_t branch_id = 0, std::source_location loc = std::source_location::current()) {
        str_assert(branch_count >= 2, "Branch count must be at least 2; received " + std::to_string(branch_count), loc);
        check_component_indices_({}, branch_id, loc);

        std::shared_ptr<Splitter> splitter = std::make_shared<Splitter>(branch_count);
        //Register the component
        components_.push_back(splitter);
        splitter->branch_id_ = branch_id;

        //First operator loaded: Add the current node as an output
        if(leaf_node_indices_.size() == 0) {
            leaf_node_indices_.push_back(0);
            return;
        }

        // Set predecessor
        splitter->predecessors_.clear();
        splitter->predecessors_[components_[leaf_node_indices_[branch_id]]->branch_id_] = leaf_node_indices_[branch_id];

        // Add new possible branches, marking the branch's index as successors
        int32_t branch_add_index = (int32_t)components_.size() - 1;
        for(int32_t i = 0; i < splitter->branch_count() - 1; i++) {
            leaf_node_indices_.push_back(branch_add_index);
        }

        // Register recently added node as the current branch leaf node's successor
        components_[leaf_node_indices_[branch_id]]->successors_[branch_id] = (int32_t)components_.size() - 1;
        leaf_node_indices_[branch_id] = (int32_t)components_.size() - 1;
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

        //Create deep pointer of the new calculator
        loss_calc_ = calc->shared_ptr_deep_copy();
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

        //Create deep pointer of the new optimizer
        optimizer_ = optim->shared_ptr_deep_copy();
    }



    /**
     * Checks if the network has the necessary components to run. 
     * If not, throws `invalid_config`. If so, allows training and optimization.
     *
     * Conditions to run:
     * The network must have a loss calculator, optimizer, and at least one component.
     * The network must have exactly one output.
     */
    void enable() {
        if(!loss_calc_) {
            throw invalid_config("Network needs a defined loss calculator");
        }
        if(!optimizer_) {
            throw invalid_config("Network needs a defined optimizer");
        }

        //Check that the network has operators
        if((int32_t)leaf_node_indices_.size() == 0) {
            throw invalid_config("Network must have at least one operator");
        }
        if(components_.size() == 0) {
            throw invalid_config("Network must have at least one operator");
        }

        //Check that the network's first element is not a splitter
        if(std::dynamic_pointer_cast<Splitter>(components_[0]) != nullptr) {
            throw invalid_config("First operator in the network cannot be a splitter");
        }

        //Check that the network's first component is the input (i.e. has no predecessors)
        if(components_[0]->predecessors_.size() > 0) {
            throw invalid_config("First operator in the network must be the input");
        }

        //Check for single output
        int32_t output_count = 0;
        for(int32_t i = 0; i < (int32_t)leaf_node_indices_.size(); i++) {
            if(leaf_node_indices_[i] != NETWORK_BRANCH_COMBINED) {
                output_count++;
            }
        }
        if(output_count != 1) {
            throw invalid_config("Network must have exactly one output");
        }

         //All components have branch IDs assigned to them (idiot check)
        for (int32_t i = 0; i < (int32_t)components_.size(); i++) {
            try {
                components_[i]->assert_branch_id_assigned();
            }
            catch(assertion_error& e) {
                throw invalid_config(std::string(e.what()) + " (component " + std::to_string(i) + ")");
            }
        }

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

        optimizer_->initialize(components_);
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
            int32_t components_index;
            std::vector<xt::xarray<double>> data;
        };

        std::queue<Task> execution_queue;
        execution_queue.push({0, 0, {input}});

        while(!execution_queue.empty()) {
            Task current = execution_queue.front();
            execution_queue.pop();

            int32_t branch_id = current.branch_id;
            int32_t components_idx = current.components_index;

            if (components_idx == NETWORK_BRANCH_COMBINED) {
                continue;
            }

            std::shared_ptr<NetworkComponent> current_op = components_.at(components_idx);
            // std::cout << "Executing " << current_op->name() << std::endl;

            // Handle splitters: Push all of its successors, including itself, into the execution queue
            if (std::shared_ptr<Splitter> splitter = std::dynamic_pointer_cast<Splitter>(current_op)) {
                std::vector<std::vector<xt::xarray<double>>> branch_output = splitter->compute(current.data, true);

                std::unordered_map<int32_t, int32_t> successors = splitter->successors();
                str_assert(!successors.empty(), "Branch must have at least one successor");

                std::unordered_map<int32_t, int32_t>::iterator succ_it = successors.begin();

                //Push all successors to the queue
                for(size_t out_idx = 0; succ_it != successors.end(); ++succ_it, ++out_idx) {
                    int32_t succ_branch_id = succ_it->first;
                    int32_t succ_component_idx = succ_it->second;
                    std::vector<xt::xarray<double>> out_data = branch_output[out_idx < branch_output.size() ? out_idx : 0];
                    execution_queue.push({succ_branch_id, succ_component_idx, out_data});
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
                        return combiner_output[0];
                    }

                    auto const& succs = combiner->successors();
                    str_assert(succs.size() == 1, "Combiner must have one successor");
                    
                    //Push the combiner's single successor to the execution queue
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
                    return op_output[0];
                }

                auto const& succs = current_op->successors();
                str_assert(succs.size() == 1, "Operator must have exactly one successor");

                //Push the operator's single successor to the execution queue
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
            int32_t components_index;
            std::vector<xt::xarray<double>> data;
        };

        std::queue<Task> execution_queue;
        int32_t output_components_idx = (int32_t)components_.size() - 1;
        int32_t output_branch_id = components_[output_components_idx]->branch_id_;
        
        execution_queue.push({output_branch_id, output_components_idx, {output_loss}});

        while(!execution_queue.empty()) {
            Task current = execution_queue.front();
            execution_queue.pop();

            int32_t branch_id = current.branch_id;
            int32_t op_idx = current.components_index;

            if (op_idx == NETWORK_BRANCH_COMBINED) {
                continue;
            }

            std::shared_ptr<NetworkComponent> current_op = components_.at(op_idx);

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
                    //Push all predecessors to the queue
                    auto pred_it = preds.begin();
                    execution_queue.push({pred_it->first, pred_it->second, branch_grads});
                }
            }
            // Handle combiners (act like splitters in the backwards pass, distributing gradients)
            else if (std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(current_op)) {
                std::vector<std::vector<xt::xarray<double>>> combiner_outputs = combiner->compute_backwards_pass(current.data, true);

                std::unordered_map<int32_t, int32_t> preds = combiner->predecessors();
                str_assert(!preds.empty(), "Combiner must have at least one predecessor");

                //Push all predecessors to the execution queue
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

                const std::unordered_map<int32_t, int32_t>& preds = current_op->predecessors();
                if (preds.empty()) {
                    return;
                }

                str_assert(preds.size() == 1, "Operator must have exactly one predecessor");

                //Push its sole predecessor to the execution queue
                std::unordered_map<int32_t, int32_t>::const_iterator pred_it = preds.begin();
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
    }

};




}
#endif