#ifndef CAST_CONTROL_FLOW_
#define CAST_CONTROL_FLOW_

#include "cast_exceptions.hpp"
#include "tensor_operator.hpp"

#include <initializer_list>

namespace cast {




/**
* Breaks a network into one or more separate paths of execution
*
* One predecessor, many successors
*/
class Splitter : public NetworkComponent {
protected:
    /**
    * Number of separate execution paths taken by this branch
    */
    int32_t branch_count_;

    /**
    * Outputs from each branch that is merged by this Splitter during the backwards pass.
    *
    * Starts the backwards pass EMPTY.
    */
    std::vector<std::vector<xt::xarray<double>>> successor_outputs_;

public:

    /**
    * Creates a new branch that splits execution into `branch_count` paths.
    * @param branch_count number of paths to split into. At least 2.
    */
    Splitter(int32_t branch_count) : branch_count_(branch_count) {
        str_assert(branch_count > 1, "Number of branches must be at least 2; instead got " + std::to_string(branch_count));
        // predecessors_.resize(1);
        // successors_.resize(branch_count);
    }


    virtual std::string name() const override {
        return "branch";
    }


    int32_t branch_count() const {
        return branch_count_;
    }


    /**
    * USE THIS METHOD!
    */
    virtual std::vector<std::vector<xt::xarray<double>>> compute(std::vector<xt::xarray<double>> inputs, bool tag) {
        str_assert(inputs.size() == 1, "The branch must receive exactly one input; instead got " + std::to_string(inputs.size()));

        std::vector<std::vector<xt::xarray<double>>> out;

        //Clone the single input into the output
        for(int32_t i = 0; i < branch_count_; i++) {
            out.push_back(inputs);
        }

        return out;
    }



    virtual std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> successor_gradient) override {
        // Perform shape and size assertions if this is not the first input
        if (!successor_outputs_.empty()) {
            const auto& first_input = successor_outputs_[0];
            
            // Check that the new input has the same number of tensors as the first input
            str_assert(successor_gradient.size() == first_input.size(), 
                       "Splitter output length (" + std::to_string(successor_gradient.size()) + 
                       ") does not match the first input length (" + std::to_string(first_input.size()) + ")");

            // Check that each tensor's shape matches the corresponding tensor in the first input
            for (size_t i = 0; i < successor_gradient.size(); i++) {
                str_assert(successor_gradient[i].shape() == first_input[i].shape(), 
                           "Shape mismatch at index " + std::to_string(i) + " between current branch output and the first branch output");

                //Check for nonzero size
                str_assert(successor_gradient[i].size() > 0, "Splitter output " + std::to_string(i) + " has no elements");
            }
        }


        // Store the current incoming gradient vector
        successor_outputs_.push_back(successor_gradient);

        // Check if we have received gradients from all successor branches
        if ((int32_t)successor_outputs_.size() == branch_count_) {

            size_t num_tensors = (int32_t)successor_outputs_[0].size();
            std::vector<xt::xarray<double>> accumulated_gradients;
            accumulated_gradients.reserve(num_tensors);

            // Initialize accumulated gradients with zeros based on the shapes of the first branch
            for (size_t t = 0; t < num_tensors; ++t) {
                accumulated_gradients.push_back(xt::zeros<double>(successor_outputs_[0][t].shape()));
            }

            // Sum up the gradients from all successor branches
            for (const auto& branch : successor_outputs_) {
                for (size_t t = 0; t < num_tensors; ++t) {
                    accumulated_gradients[t] += branch[t];
                }
            }

            // Clear successor_outputs_ for future passes
            successor_outputs_.clear();

            return accumulated_gradients;
        }

        // Return an empty vector for any inputs prior to the branch_count_-th input
        return {};
    }


    /**
    * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    */
    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> unused) override {
        throw not_implemented("Does not exist");
    }

};





/**
* Collapses control flow from one or more other branches into the branch that this object is added to.
*
* Multiple predecessors, one successor
*/
class Combiner : public NetworkComponent {
protected:
    /**
    * 0-based indices in the network's operator list (excluding the branch that the Combiner is added to) 
    * that will be merged.
    */
    std::vector<int32_t> branch_indices_;

    /**
    * Outputs from each branch that is merged by this Combiner. Index `i` corresponds to the output from branch `branch_indices_[i]`.
    *
    * Starts EMPTY.
    */
    std::vector<std::vector<xt::xarray<double>>> combined_predecessor_outputs_;

public:

    /**
    * Creates a new combiner that pools execution from the branches given at `branch_indices`.
    * @param branch_indices 0-based branch indices to combine. Has at least 1 element.
    */
    Combiner(std::initializer_list<int32_t> branch_indices) : branch_indices_(branch_indices)  {
        str_assert(branch_indices.size() > 0, "Number of branch indices given must be at least 2");

        combined_predecessor_outputs_.reserve(branch_indices.size());
    }


    std::vector<int32_t> branch_indices() const {
        return branch_indices_;
    }


    std::string name() const override {
        return "combiner";
    }


    /**
    * Use this method!
    * @param predecessor_outputs list of layer outputs. Has length >= 1, and each element has the same size and matching corresponding shapes
    * @return sum of all outputs, or an empty vector if not all branches are combined
    */
    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> predecessor_outputs) override {
        str_assert(predecessor_outputs.size() > 0, "Combiner requires at least 1 input");

        //Add the most recent input
        combined_predecessor_outputs_.push_back(predecessor_outputs);

        //Return the sum if all outputs have been combined
        if(combined_predecessor_outputs_.size() == branch_indices_.size() + 1) {
            std::vector<xt::xarray<double>> sum;
        
            //First input
            for(int32_t o = 0; o < (int32_t)combined_predecessor_outputs_[0].size(); o++) {
                sum.push_back(combined_predecessor_outputs_[0][o]);
            }
            //All subsequent inputs
            for(int32_t c = 1; c < (int32_t)combined_predecessor_outputs_.size(); c++) {
                
                for(int32_t o = 0; o < (int32_t)combined_predecessor_outputs_[c].size(); o++) {
                    sum[o] += combined_predecessor_outputs_[c][o];
                }
                
            }

            //Clear the outputs
            combined_predecessor_outputs_.clear();
            return sum;
        }

        //Not all outputs combined: Return the empty vector
        return {};
    }


    /**
    * Use this method instead!
    */
    std::vector<std::vector<xt::xarray<double>>> compute_backwards_pass(std::vector<xt::xarray<double>> prev_gradient, bool tag) {
        str_assert(prev_gradient.size() > 0, "Combiner backwards pass requires at least 1 element in the input gradient");

        // Determine how many predecessors this layer combines based on branch_indices_
        int32_t num_predecessors = (int32_t)branch_indices_.size() + 1;

        std::vector<std::vector<xt::xarray<double>>> backprop_outputs;
        backprop_outputs.reserve(num_predecessors);

        // Clone the incoming gradient branch_outputs for each predecessor branch
        for(int32_t i = 0; i < num_predecessors; i++) {
            backprop_outputs.push_back(prev_gradient);
        }

        return backprop_outputs;
    }


    /**
    * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    */
    std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> unused) override {
        throw not_implemented("Does not exist");
    }
};



}
#endif