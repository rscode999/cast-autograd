#ifndef CAST_NETWORK_
#define CAST_NETWORK_

#include "activation_function.hpp"
#include "cast_exceptions.hpp"
#include "control_flow.hpp"
#include "layer.hpp"
#include "loss_calculator.hpp"
#include "optimizer.hpp"

#include <unordered_set>
#include <xtensor/containers/xarray.hpp>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <queue>
#include <ranges>
#include <stack>
#include <string>



namespace cast {




/**
 * Neural network with trainable weights.
 * Components (i.e. layers, activation functions) are added to the network individually.
 *
 * Networks have a maximum of 2 billion components. Attempting to add more causes a `std::out_of_range` exception.
 */
class Network {
private:

    /**
     * Network components that the network uses, in the order that they were added
     */
    std::vector<std::shared_ptr<NetworkComponent>> components_;

    /**
     * True if the network is ready for training and evaluation
     */
    bool enabled_;

    /**
    * After the first component is added, this field is 1 greater than the largest branch ID 
    * (even among branches that no longer exist) assigned to any component.
    *
    * Example: If the network created 2 branches, this field equals 3.
    */
    int32_t next_branch_id_;

    /**
    * Indices in `components_` that are leaf nodes, i.e. have no successors. Element `i` is the leaf node index for branch `i`.
    *
    * Leaf nodes are the only nodes that can be added to.
    * The length of this vector is the total number of branches used in this network, whether active or combined with another branch.
    *
    * Element `i` equaling `NETWORK_BRANCH_COMBINED` indicates that branch `i` has been combined, and no longer exists.
    * (This vector is never removed from.)
    */
    std::unordered_map<int32_t, int32_t> leaf_node_indices_;
    
    /**
     * Loss metric used by this network
     */
    std::shared_ptr<LossCalculator> loss_calc_;

    /**
     * Optimizer used by this network
     */
    std::shared_ptr<Optimizer> optimizer_;


    /**
    * Stores all data required for a network component to execute.
    * Items of this type are created and placed in a queue during the forward or backward pass.
    *
    * Contains: component's branch ID, index in the `components_` list, and any inputs it has.
    */
    struct ComponentExecutionData {
        /**
        * Branch ID of the component
        */
        int32_t branch_id;

        /**
        * 0-based index of the component in the `components_` list
        */
        int32_t component_index;

        /**
        * Input tensor(s) that the component receives when its turn to execute arrives.
        * In the backwards pass, this field holds the component's upstream gradients.
        */
        std::vector<xt::xarray<double>> component_input;
    };


 
    /**
    * Throws `bad_component_addition` if `combine_branch_ids` and `branch_id` are incompatible with each other or the current network configuration.
    * Called when a network component is added.
    *
    * The network must not be enabled: otherwise, throws `cast::bad_network_config`.
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
            throw bad_network_config("Network must not be enabled to add components", check_location);
        }
        
        //Stop check early if there are no leaf nodes (that is, the component is the first one added)
        if (leaf_node_indices_.empty()) {
            return;
        }

        //Overflow check
        if((int32_t)components_.size() < 0 || (int32_t)components_.size() >= 2000000000) { //2 billion
            throw std::out_of_range("Cannot add more than 2 billion operators to the network");
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
        if (!leaf_node_indices_.contains(branch_id)) {
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
                    "Branch IDs to combine index " + std::to_string(pos) + " (" + std::to_string(combine_index) +
                    ") must be non-negative and less than the total number of branches created (" + std::to_string(leaf_node_indices_size) + ")",
                    check_location
                );
            }
            //Cannot add to a branch that has been combined
            if (!leaf_node_indices_.contains(combine_index)) {
                throw bad_component_addition(
                    "Branch IDs to combine index " + std::to_string(pos) + " (" + std::to_string(combine_index) +
                    ") was already combined",
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
     * Creates an empty network. The new network is disabled.
     */
    Network() : enabled_(false), next_branch_id_(0) {      
    };


    /**
    * Copies `other_network` into a new network.
    *
    * All copied data is a deep copy, so modifying `other_network` does not affect the newly created network.
    * @param other_network network to copy
    */
    Network(const Network& other_network) {
        enabled_ = other_network.enabled_;
        next_branch_id_ = other_network.next_branch_id_;
        leaf_node_indices_ = other_network.leaf_node_indices_;
        loss_calc_ = (other_network.loss_calc_) ? other_network.loss_calc_->shared_ptr_deep_copy() : nullptr;
        optimizer_ = (other_network.optimizer_) ? other_network.optimizer_->shared_ptr_deep_copy() : nullptr;

        components_ = {};
        for(std::shared_ptr<NetworkComponent> component : other_network.components_) {
            str_assert(component != nullptr, "INTERNAL ERROR- Each network component cannot be nullptr");
            components_.push_back(component->shared_ptr_deep_copy());
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //GETTERS

    /**
    * Returns the set of valid branch IDs.
    * 
    * Branches that have been merged are not included.
    * @return IDs of valid branches
    */
    std::unordered_set<int32_t> active_branch_ids() const {
        auto keysView = std::views::keys(leaf_node_indices_);
        return std::unordered_set<int32_t>(keysView.begin(), keysView.end());
    }



    /**
    * Returns a mapping of remaining branch IDs to the component IDs of their heads.
    *
    * A component's ID is the 0-based order in which the component was added to the network.
    * 
    * Branches that have been merged are not included.
    * @return mapping of: branch ID -> final index of the branch
    */
    std::unordered_map<int32_t, int32_t> active_branch_id_heads() const {
        return leaf_node_indices_;
    }



    /**
    * Returns a pointer to the component with ID `component_id`.
    * 
    * A component's ID is the 0-based order in which the component was added to the network.
    * ID 0 is the first component added, 1 is the second component added, and so on.
    *
    * The returned pointer cannot be used to modify the network's component.
    * @param component_id component number to access. At least 0, and less than the number of components added so far.
    * @return `i`-th component in the network
    */
    std::shared_ptr<NetworkComponent> component_at(int32_t component_id) const {
        str_assert(component_id >= 0 && component_id < (int32_t)components_.size(), "ID must be at least 0 and less than the number of components added");
        return components_[component_id]->shared_ptr_deep_copy();
    }   



    /**
    * @return whether the network is ready for training and optimization
    */
    bool is_enabled() const {
        return enabled_;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //SETTERS

    
    /**
    * Adds a combiner, merging the branch IDs given in `branch_ids_to_combine`, to branch `branch_id`.
    *
    * If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, 
    * this method throws `cast::bad_component_addition`.
    *
    * `cast::bad_component_addition` is also thrown if any of the branch IDs in `branch_ids_to_combine` is out of the range [0, <number of branches used in the network - 1>],
    * has already been merged, or equals `branch_id` (combiners cannot merge their own branch).
    * A Combiner cannot be the first component added to a network.
    *
    * To use this method, the network cannot be enabled.
    * @param branch_ids_to_combine list of branch IDs to merge. Non-empty
    * @param branch_id branch to add the new combiner to
    * @param loc location where this method is called (for debugging purposes)
    */
    void add_combiner(std::initializer_list<int32_t> branch_ids_to_combine, int32_t branch_id = 0, std::source_location loc = std::source_location::current()) {
        str_assert(branch_ids_to_combine.size() > 0, "Branch IDs to combine must be non-empty");
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

        // Add the branch that the combiner is added to into predecessors
        int32_t target_leaf_idx = leaf_node_indices_[branch_id];
        int32_t target_branch_id = components_[target_leaf_idx]->branch_id_;
        combiner->predecessors_[target_branch_id] = target_leaf_idx;
        components_[target_leaf_idx]->successors_[target_branch_id] = combiner_node_index;

        // for(auto p : leaf_node_indices_) {
        //     std::cout << p.first << "," << p.second << std::endl;
        // }

        // Add all other combiner branch indices into predecessors and mark them as combined
        for (int32_t combiner_branch_idx : combiner->branch_indices()) {
            int32_t leaf_idx = leaf_node_indices_[combiner_branch_idx];
            int32_t leaf_branch_id = components_[leaf_idx]->branch_id_;

            combiner->predecessors_[leaf_branch_id] = leaf_idx;
            components_[leaf_idx]->successors_[branch_id] = combiner_node_index;

            // Mark branch as combined
            bool erase_successful = leaf_node_indices_.erase(combiner_branch_idx);
            str_assert(erase_successful, "INTERNAL ERROR- Attempted to merge a branch that doesn't exist");
        }

        // Update the leaf node index for the target branch to point to the combiner
        leaf_node_indices_[branch_id] = combiner_node_index;
    }


    
    /**
    * Adds `op` to the end of branch `branch_id`.
    *
    * An operator is a layer or an activation function.
    *
    * If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, 
    * this method throws `cast::bad_component_addition`.
    *
    * To use this method, the network cannot be enabled.
    * @param op operator to add to a branch. Not equal to `nullptr`
    * @param branch_id branch to add the new operator to
    * @param loc location where this method is called (for debugging purposes)
    */
    void add_operator(std::shared_ptr<Operator> op, int32_t branch_id = 0, std::source_location loc = std::source_location::current()) {
        check_component_indices_({}, branch_id, loc);
        str_assert(op != nullptr, "New operator to add cannot be nullptr", loc);

        //Make a deep copy of the operator
        std::shared_ptr<NetworkComponent> new_operator = op->shared_ptr_deep_copy();

        //Register the operator
        components_.push_back(new_operator);
        new_operator->branch_id_ = branch_id;

        //First operator loaded: Add the current node as an output
        if(leaf_node_indices_.size() == 0) {
            str_assert(next_branch_id_ == 0, "Next branch ID must be 0 if the first operator is loaded");
            leaf_node_indices_[next_branch_id_] = 0;
            next_branch_id_ = 1; //After branch 0 is created, the next possible branch ID is 1.
            return;
        }

        // Set predecessor
        new_operator->predecessors_.clear();
        new_operator->predecessors_[components_[leaf_node_indices_[branch_id]]->branch_id_] = leaf_node_indices_[branch_id];

        // Register recently added node as the current branch leaf node's successor
        components_[leaf_node_indices_[branch_id]]->successors_[branch_id] = (int32_t)components_.size() - 1;
        leaf_node_indices_[branch_id] = (int32_t)components_.size() - 1;
    }



    /**
    * Adds a splitter that distributes execution across `branch_count` new branches, to branch `branch_id`.
    *
    * If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, 
    * this method throws `cast::bad_component_addition`.
    *
    * To use this method, the network cannot be enabled.
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
            str_assert(next_branch_id_ == 0, "Next branch ID must be 0 if the first splitter is loaded");
            leaf_node_indices_[next_branch_id_] = 0;
            next_branch_id_ = 1; //After branch 0 is created, the next possible branch ID is 1.
            return;
        }

        // Set predecessor
        splitter->predecessors_.clear();
        splitter->predecessors_[components_[leaf_node_indices_[branch_id]]->branch_id_] = leaf_node_indices_[branch_id];

        // Add new possible branches, marking the branch's index as successors
        int32_t branch_add_index = (int32_t)components_.size() - 1;
        for(int32_t i = 0; i < splitter->branch_count() - 1; i++) {
            leaf_node_indices_[next_branch_id_] = branch_add_index;
            next_branch_id_++;
        }

        // Register recently added node as the current branch leaf node's successor
        components_[leaf_node_indices_[branch_id]]->successors_[branch_id] = (int32_t)components_.size() - 1;
        leaf_node_indices_[branch_id] = (int32_t)components_.size() - 1;
    }



    /**
    * Sets the component with ID `component_id` to `component`.
    * 
    * A component's ID is the 0-based order in which the component was added to the network.
    * ID 0 is the first component added, 1 is the second component added, and so on.
    *
    * The pointer to `component` cannot be used to modify the network's added component.
    * @param component_id component number to set. At least 0, and less than the number of components added so far.
    * @param component component to set
    */
    void set_component_at(int32_t component_id, std::shared_ptr<NetworkComponent> component) {
        str_assert(0 <= component_id && component_id < (int32_t)components_.size(), "Component ID must be at least 0 and at most " + std::to_string(components_.size()) + ": got " + std::to_string(component_id));
        components_[component_id].reset();
        components_[component_id] = component->shared_ptr_deep_copy();
    }


    /**
     * Sets this network's loss calculator to `calc`.
     *
     * To use this method, the network cannot be enabled.
     * @param calc new loss calculator to use. Non-null
     */
    void set_loss_calculator(std::shared_ptr<LossCalculator> calc) {
        str_assert(calc != nullptr, "New loss calculator must be non-null");
        //Enable check
        if(enabled_) {
            throw bad_network_config("Network cannot be enabled to set the loss calculator");
        }

        //Reset the loss calculator if it exists
        if(loss_calc_) {
            loss_calc_.reset();
        }
        loss_calc_ = calc;
    }



    /**
     * Sets this network's optimizer to `optim`.
     *
     * The pointer to the optimizer can be manipulated from outside the network.
     *
     * To use this method, the network cannot be enabled.
     * @param optim new optimizer to use. Non-null
     */
    void set_optimizer(std::shared_ptr<Optimizer> optim) {
        str_assert(optim != nullptr, "New optimizer must be non-null");
        //Enable check
        if(enabled_) {
            throw bad_network_config("Network cannot be enabled to set the optimizer");
        }

        //Reset optimizer if it exists
        if(optimizer_) {
            optimizer_.reset();
        }
        optimizer_ = optim;
    }


    /**
    * Sets the network's optimizer hyperparameters to `new_hyperparams`.
    *
    * The preconditions on `new_hyperparams` depend on the optimizer used.
    */
    void set_optimizer_hyperparams(std::initializer_list<double> new_hyperparams) {
        if(!optimizer_) {
            throw bad_network_config("The network has no optimizer");
        }
        optimizer_->set_hyperparameters(new_hyperparams);
    }



    /**
    * Disables the network. Prevents training and optimization, but allows more components to be added.
    */
    void disable() {
        enabled_ = false;
    }


    /**
     * Checks if the network has the necessary components to run. 
     * If not, throws `cast::enable_failed_error`. If so, allows training and optimization.
     *
     * If successful, this method initializes the stored optimizer.
     *
     * Conditions to run:
     * The network must have a loss calculator, optimizer, and at least one component.
     * The network must have exactly one output.
     */
    void enable() {
        if(!loss_calc_) {
            throw enable_failed_error("Network needs a defined loss calculator");
        }
        if(!optimizer_) {
            throw enable_failed_error("Network needs a defined optimizer");
        }

        //Check that the network has operators
        if((int32_t)leaf_node_indices_.size() == 0) {
            throw enable_failed_error("Network must have at least one operator");
        }
        if(components_.size() == 0) {
            throw enable_failed_error("Network must have at least one operator");
        }

        //Check that the network's first element is not a splitter
        if(std::dynamic_pointer_cast<Splitter>(components_[0]) != nullptr) {
            throw enable_failed_error("First operator in the network cannot be a splitter");
        }

        //Check that the network's first component is the input (i.e. has no predecessors)
        if(components_[0]->predecessors_.size() > 0) {
            throw enable_failed_error("First operator in the network must be the input");
        }

        //Check for single output
        //If there is not exactly 1 unmerged branch, convert all unmerged branch IDs to a string and error
        if(leaf_node_indices_.size() != 1) {
            std::string unmerged_branches_str = "";
            for(std::pair<int32_t, int32_t> branch_end : leaf_node_indices_) {
                unmerged_branches_str += std::to_string(branch_end.first) + ", ";
            }
            throw enable_failed_error("Network must have exactly one output. Remaining branches: " + unmerged_branches_str);
        }

        //All control paths are assigned to a branch
        for(int32_t i = 0; i < (int32_t)components_.size(); i++) {
            try {
                (void)components_[i]->branch_id();
            }
            catch(unassigned_branch_error& e) {
                throw enable_failed_error("Invalid branch ID at component index " + std::to_string(i) + ": " + e.what());
            }
        }

        //IMPORTANT: Initialization logic must come AFTER the checks. Otherwise, this method could be in a try/catch and allow training, even though the check failed.
        optimizer_->initialize(components_);
        enabled_ = true;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //METHODS


    /**
     * Returns the result of the network's forward pass on `input`.
     *
     * Throws `cast::shape_error` if layer dimensions are incompatible.
     *
     * To use this method, the network must be enabled. 
     * @param input tensor to compute forward pass on
     * @return result of forward pass
     */
    xt::xarray<double> forward(xt::xarray<double> input) {
        //Thought this might help with thread safety
        if(!enabled_) {
            throw bad_network_config("Must enable the network prior to training");
        }

        std::queue<ComponentExecutionData> execution_queue;
        execution_queue.push({0, 0, {input}});

        while(!execution_queue.empty()) {
            //Thought this might help with thread safety
            if(!enabled_) {
                throw bad_network_config("Must enable the network prior to training");
            }

            ComponentExecutionData current = execution_queue.front();
            execution_queue.pop();

            int32_t branch_id = current.branch_id;
            int32_t component_idx = current.component_index;
            
            //Get the component from the network's storage list
            std::shared_ptr<NetworkComponent> current_component = components_.at(component_idx);
            // std::cout << "Executing " << current_op->name() << std::endl;

            // Handle splitters: Push all of its successors, including itself, into the execution queue
            if (std::shared_ptr<Splitter> splitter = std::dynamic_pointer_cast<Splitter>(current_component)) {
                std::vector<std::vector<xt::xarray<double>>> branch_output = splitter->compute(current.component_input, true);

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
            else if(std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(current_component)) {
                std::vector<xt::xarray<double>> combiner_output = combiner->forward(current.component_input);
                
                // Combiner output is non-empty only when all required inputs have arrived (i.e. its output is non-empty)
                if(!combiner_output.empty()) {
                    // std::cout << "Combiner is ready" << std::endl;
                    
                    //The Combiner has no successors: Return
                    if(combiner->successors().empty()) {
                        // std::cout << "COMBINER HAS NO SUCCESSORS" << std::endl;
                        return combiner_output[0];
                    }

                    const std::unordered_map<int32_t, int32_t> succs = combiner->successors();
                    str_assert(succs.size() == 1, "Combiner must have one successor");
                    
                    //Push the combiner's single successor to the execution queue
                    int32_t target_branch_id = succs.begin()->first;
                    int32_t target_op_idx = succs.begin()->second;
                    execution_queue.push({target_branch_id, target_op_idx, combiner_output});
                }
            }
            // Handle single operator
            else {
                std::vector<xt::xarray<double>> op_output;

                //Compute the output. If incompatible shapes, re-throw the shape error
                try {
                    op_output = current_component->forward(current.component_input);
                }
                catch(shape_error& e) {
                    throw shape_error("Input to " + current_component->to_string() + " (branch " + std::to_string(current_component->branch_id()) + "): " + e.what());
                }

                //No successors: Return (this is the single operator with no successors)
                if(current_component->successors().empty()) {
                    return op_output[0];
                }

                const std::unordered_map<int32_t, int32_t> succs = current_component->successors();
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
     * Computes the backward pass, beginning with loss between `predicted` and `expected`.
     *
     * Stores updated gradients inside the network layers, for use by the network's optimizer.
     *
     * The network must be enabled to use this method.
     * @param predicted network's prediction for a given input
     * @param expected what the network should have predicted for the input
     */
    void backward(xt::xarray<double> predicted, xt::xarray<double> expected) {
        if(!enabled_) {
            throw bad_network_config("Must enable the network prior to computing backwards pass");
        }
        if(!loss_calc_) {
            throw bad_network_config("INTERNAL ERROR- No loss calculator defined");
        }

        xt::xarray<double> output_loss = loss_calc_->compute_gradient(predicted, expected);

        std::queue<ComponentExecutionData> execution_queue;
        int32_t output_components_idx = (int32_t)components_.size() - 1;
        int32_t output_branch_id = components_[output_components_idx]->branch_id_;
        
        execution_queue.push({output_branch_id, output_components_idx, {output_loss}});

        while(!execution_queue.empty()) {
            //Thought this might help with thread safety
            if(!enabled_) {
                throw bad_network_config("Must enable the network prior to computing backwards pass");
            }

            ComponentExecutionData current = execution_queue.front();
            execution_queue.pop();

            int32_t branch_id = current.branch_id;
            int32_t op_idx = current.component_index;

            std::shared_ptr<NetworkComponent> current_component = components_.at(op_idx);

            // Handle splitters (act like combiners in the backwards pass, collecting inputs)
            if (std::shared_ptr<Splitter> splitter = std::dynamic_pointer_cast<Splitter>(current_component)) {
                std::vector<xt::xarray<double>> branch_grads = splitter->backward(current.component_input);

                // Branch output is non-empty only when all required inputs have arrived
                if (!branch_grads.empty()) {
                    const std::unordered_map<int32_t, int32_t> preds = splitter->predecessors();
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
            else if (std::shared_ptr<Combiner> combiner = std::dynamic_pointer_cast<Combiner>(current_component)) {
                std::vector<std::vector<xt::xarray<double>>> combiner_outputs = combiner->compute_backwards_pass(current.component_input, true);

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
                std::vector<xt::xarray<double>> op_output = current_component->backward(current.component_input);

                const std::unordered_map<int32_t, int32_t>& preds = current_component->predecessors();
                //Stop early if there are no predecessors (it's the first component in the network)
                if (preds.empty()) {
                    return;
                }
                //otherwise, the operator can have at most one component
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
     * Runs an optimization pass on the network's layers.
     *
     * Uses the network's stored optimizer and the gradients computed from the `backward` method.
     *
     * To use this method, the network must be enabled.
     *
     * WARNING: Calling `optimize` multiple times, without computing a `backward` operation prior,
     * wil cause the network to use its stored gradients multiple times.
     * @param zero_grad whether to set all operator's gradients to 0 after computing the optimization pass
     */
    void optimize(bool zero_grad = true) {
        if(!enabled_) {
            throw bad_network_config("Must enable the network prior to computing optimization pass"); 
        }
        if(!optimizer_) {
            throw bad_network_config("INTERNAL ERROR- No optimizer defined");
        }

        optimizer_->step(zero_grad);
    }



    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //OPERATOR OVERLOADS

    /**
    * Deep-copies the data from `other_network` into this network.
    * @param other_network network to copy
    * @return deep copy of `other_network`
    */
    Network& operator=(const Network& other_network) {
        //self assignment check
        if(this == &other_network) {
            return *this;
        }

        enabled_ = other_network.enabled_;
        next_branch_id_ = other_network.next_branch_id_;
        leaf_node_indices_ = other_network.leaf_node_indices_;
        loss_calc_ = (other_network.loss_calc_) ? other_network.loss_calc_->shared_ptr_deep_copy() : nullptr;
        optimizer_ = (other_network.optimizer_) ? other_network.optimizer_->shared_ptr_deep_copy() : nullptr;

        components_ = {};
        for(std::shared_ptr<NetworkComponent> component : other_network.components_) {
            str_assert(component != nullptr, "INTERNAL ERROR- Each network component cannot be nullptr");
            components_.push_back(component->shared_ptr_deep_copy());
        }

        return *this;
    }


    /**
    * Exports `network` to the output stream `output_stream`, returning `output_stream` with `network`'s information inside.
    * @param output_stream stream to put the network into
    * @param network Network object to export
    * @return `output_stream` with `network` inserted
    */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Network& network);


    /**
    * Properly destroys a Network
    */
    ~Network() = default;
};


template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Network& network) {
    output_stream << "Network, " << (network.enabled_ ? "enabled" : "disabled") << "\n";

    //Export the loss calculator if it exists
    output_stream << "Loss calculator: ";
    if(network.loss_calc_) {
        output_stream << *network.loss_calc_;
    }
    else {
        output_stream << "(none)";
    }

    //Export the optimizer if it exists
    output_stream << "\nOptimizer: ";
    if(network.optimizer_) {
        output_stream << *network.optimizer_;
    }
    else {
        output_stream << "(none)";
    }

    //Export all the layers
    if(network.components_.size() > 0) {
        output_stream << "\n";
    }
    for (int32_t i = 0; i < (int32_t)network.components_.size(); i++) {
        str_assert(network.components_[i] != nullptr, "Network component " + std::to_string(i) + " is nullptr");
        output_stream << "Component " << i << ": " << *network.components_[i] << "\n";
    }
    return output_stream;
}




}
#endif