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
class Branch : public TensorOperator {
protected:
    /**
    * Number of separate execution paths taken by this branch
    */
    int32_t branch_count_;

public:

    /**
    * Creates a new branch that splits execution into `branch_count` paths.
    * @param branch_count number of paths to split into. At least 2.
    */
    Branch(int32_t branch_count) : branch_count_(branch_count) {
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
    virtual std::vector<std::vector<xt::xarray<double>>> compute(std::vector<std::vector<xt::xarray<double>>> inputs, bool tag) {
        str_assert(inputs.size() == 1, "The branch must receive exactly one input; instead got " + std::to_string(inputs.size()));

        std::vector<std::vector<xt::xarray<double>>> out;

        //Clone the single input into the output
        for(int32_t i = 0; i < branch_count_; i++) {
            out.push_back(inputs[0]);
        }

        return out;
    }


    virtual std::vector<std::vector<xt::xarray<double>>> compute_backwards_pass(std::vector<std::vector<xt::xarray<double>>> branch_outputs) {
        throw not_implemented("TO DO");
    }


    /**
    * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    */
    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> unused) override {
        throw not_implemented("Does not exist");
    }

    /**
    * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    */
    std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> unused) override {
        throw not_implemented("Does not exist");
    }
};





/**
* Collapses control flow from one or more other branches into the branch that this object is added to.
*
* Multiple predecessors, one successor
*/
class Combiner : public TensorOperator {
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



    // /**
    // * DO NOT USE! Throws `cast::not_implemented`. The method exists solely to implement a virtual method.
    // */
    // std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> unused) override {
    //     throw not_implemented("Does not exist");
    // }


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
            return sum;
        }

        //Not all outputs combined: Return the empty vector
        return {};
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