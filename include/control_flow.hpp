#ifndef CAST_CONTROL_FLOW_
#define CAST_CONTROL_FLOW_

#include "cast_exceptions.hpp"
#include "network_component.hpp"

#include <initializer_list>
#include <string>

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
    std::vector<xt::xarray<double>> successor_outputs_;

public:

    /**
    * Creates a new branch that splits execution into `branch_count` paths.
    * @param branch_count number of paths to split into. At least 2.
    */
    Splitter(int32_t branch_count) : branch_count_(branch_count) {
        str_assert(branch_count > 1, "Number of branches must be at least 2; instead got " + std::to_string(branch_count));
    }


    /**
    * @return deep pointer copy of this Splitter object
    */
    std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const override {
        return std::make_shared<Splitter>(*this);
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * @return number of branches after the Splitter's operation, including the Splitter's own branch. Always at least 2.
    */
    int32_t branch_count() const {
        return branch_count_;
    }
    

    /**
    * @return the string "splitter ({branch count of this splitter object})"
    */
    virtual std::string to_string() const override {
        return "splitter {->" + std::to_string(branch_count_) + "}";
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * Returns `input` copied `branch_count()` times.
    * @param input vector(s) to copy across multiple outputs. Non-empty.
    * @param tag unused; required to distinguish this method from the overridden method that returns `std::vector<xt::xarray<double>>`
    * @return vector of length `branch_count()`, where each index contains a copy of `input`
    */
    virtual std::vector<xt::xarray<double>> forward(const xt::xarray<double>& input, bool tag) {
        str_assert(input.size() >= 1, "The input must be non-empty");

        std::vector<xt::xarray<double>> out;

        //Clone the single input into the output
        for(int32_t i = 0; i < branch_count_; i++) {
            out.push_back(input);
        }

        return out;
    }


    /**
    * Returns the element-wise sum of all ements in `successor_gradients`.
    * @param successor_gradients single successor gradient. Size and shape of all its elements match those of the first given input
    * @return result of the splitter's backpropagation operation computed on all received inputs
    */
    xt::xarray<double> backward(const std::vector<xt::xarray<double>>& successor_gradients) {
        str_assert(
            !successor_gradients.empty(),
            "Splitter requires at least one output gradient"
        );

        xt::xarray<double> result =
            xt::zeros<double>(
                successor_gradients[0].shape()
            );

        for (const auto& gradient : successor_gradients) {
            result += gradient;
        }

        return result;
    }


    /**
    * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    */
    xt::xarray<double> forward(const xt::xarray<double>& unused) override {
        throw not_implemented("Splitter forward method using only a xarray<double> does not exist");
    }

    /**
    * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    */
    xt::xarray<double> backward(const xt::xarray<double>& unused) override {
        throw not_implemented("Splitter backward method using only a xarray<double> does not exist");
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
    * Uses `str_assert` to check that none of the combiner's branch indices equal the branch that the combiner is in, and that a branch index is assigned
    *
    * Does nothing if NDEBUG is defined.
    * @param loc location of where this method is called
    */
    void assert_no_self_assign_(std::source_location loc = std::source_location::current()) {
        #ifndef NDEBUG

        int32_t current_branch_index = 0;
        for(int32_t branch_index : branch_indices_) {
            str_assert(branch_index != branch_id_, "Cannot assign the combiner to merge branches in its own branch, " + std::to_string(branch_id_), loc);
            ++current_branch_index;
        }

        #endif
    }

public:

    /**
    * Creates a new combiner that pools execution from the branches given at `branch_indices`.
    * @param branch_indices 0-based branch indices to combine. Has at least 1 element.
    */
    Combiner(std::initializer_list<int32_t> branch_indices) : branch_indices_(branch_indices)  {
        str_assert(branch_indices.size() > 0, "Number of branch indices given must be at least 2");
    }


    /**
    * @return deep pointer copy of this Combiner object
    */
    std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const override {
        return std::make_shared<Combiner>(*this);
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * @return list of branch IDs that this combiner merges. Does not include the combiner's own branch ID.
    */
    std::vector<int32_t> branch_indices() const {
        return branch_indices_;
    }


    /**
    * @return the string "combiner ({branch indices combined} -> {branch of combiner})"
    */
    std::string to_string() const override {
        std::string header = "combiner {";
        std::string combined_branches = "";
        //Guarantee of at least one branch to combine
        for (int32_t i = 0; i < (int32_t)branch_indices_.size() - 1; i++) {
            combined_branches += std::to_string(branch_indices_[i]) + ", ";
        }
        combined_branches += std::to_string(branch_indices_[branch_indices_.size() - 1]) + "}"; 

        std::string footer = " -> " + std::to_string(branch_id_);

        return header + combined_branches + footer;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////


    /**
    * Returns the element-wise sum of all inputs given.
    * @param predecessor_outputs layer outputs from all predecessors
    * @return sum of all inputs
    */
    xt::xarray<double> forward(const std::vector<xt::xarray<double>>& predecessor_outputs) {
        str_assert(
            predecessor_outputs.size() == branch_indices_.size() + 1,
            "Incorrect number of combiner inputs"
        );
        str_assert(
            !predecessor_outputs.empty() && predecessor_outputs[0].size() > 0,
            "Combiner requires non-empty inputs"
        );

        assert_no_self_assign_();

        xt::xarray<double> sum = xt::zeros<double>(predecessor_outputs[0].shape());

        for (const auto& input : predecessor_outputs) {
            str_assert(
                input.shape() == predecessor_outputs[0].shape(),
                "Combiner input shapes must match"
            );

            sum += input;
        }

        return sum;
    }

    /**
    * Returns `prev_gradient` copied `branch_indices().size()` times.
    * @param prev_gradient tensor to copy across multiple outputs. Non-empty.
    * @param tag unused; required to distinguish this method from the overridden method that returns `xt::xarray<double>`
    * @return vector of length `branch_indices().size()`, where each index contains a copy of `prev_gradient`
    */
    virtual std::vector<xt::xarray<double>> backward(const xt::xarray<double>& prev_gradient, bool tag) {
        str_assert(prev_gradient.size() > 0, "Combiner backwards pass requires at least 1 element in the input gradient");
        assert_no_self_assign_();

        // Determine how many predecessors this layer combines based on branch_indices_
        int32_t num_predecessors = (int32_t)branch_indices_.size() + 1;

        std::vector<xt::xarray<double>> backprop_outputs;
        backprop_outputs.reserve(num_predecessors);

        // Clone the incoming gradient branch_outputs for each predecessor branch
        for(int32_t i = 0; i < num_predecessors; i++) {
            backprop_outputs.push_back(prev_gradient);
        }

        return backprop_outputs;
    }


    /**
    * DO NOT USE! Throws `cast::not_implemented`.
    */
    xt::xarray<double> forward(const xt::xarray<double>& unused) override {
        throw not_implemented("Do not use the single-input forward method on a Combiner");
    }

    /**
    * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    */
    xt::xarray<double> backward(const xt::xarray<double>& unused) override {
        throw not_implemented("Combiner backward method using only a xarray<double> does not exist");
    }

};



}
#endif