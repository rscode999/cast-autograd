#ifndef CAST_CONTROL_FLOW_
#define CAST_CONTROL_FLOW_

#include "cast_exceptions.hpp"
#include "tensor_operator.hpp"

#include <initializer_list>

namespace cast {




/**
* Breaks a network into one or more separate paths of execution
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
    virtual std::vector<std::vector<xt::xarray<double>>> compute(std::vector<std::vector<xt::xarray<double>>> inputs) {
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
*/
class Combiner : public TensorOperator {
private:
    std::vector<int32_t> branch_indices_;

public:

    /**
    * Creates a new combiner that pools execution from the branches given at `branch_indices`.
    * @param branch_indices 0-based branch indices to combine. Has at least 1 element.
    */
    Combiner(std::initializer_list<int32_t> branch_indices) : branch_indices_(branch_indices)  {
        str_assert(branch_indices.size() > 0, "Number of branch indices given must be at least 1");
    }


    std::vector<int32_t> branch_indices() const {
        return branch_indices_;
    }


    std::string name() const override {
        return "combiner";
    }



    /**
    * USE THIS METHOD!
    * @param inputs list of layer outputs. Has length >= 1, and each element has the same size and matching corresponding shapes
    */
    virtual std::vector<std::vector<xt::xarray<double>>> compute(std::vector<std::vector<xt::xarray<double>>> predecessors_outputs) {
        str_assert(predecessors_outputs.size() > 0, "Combiner requires at least 1 input");

        //Initialize sums
        std::vector<xt::xarray<double>> sums;
        for(xt::xarray<double> first_pred_output : predecessors_outputs[0]) {
            sums.push_back(xt::zeros_like(first_pred_output));
        }

        //Calculate the sums
        for(std::vector<xt::xarray<double>> predecessor_output : predecessors_outputs) { switch to normal for loop
            for(int32_t i = 0; i < (int32_t)predecessor_output.size(); i++) {
                str_assert(predecessor_output[i].shape() == sums[i].shape(), "element " + std::to_string(i) + " must have the same shapes as element 0");
                sums[i] += predecessor_output[i];
            }
        }

        return {sums};
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



}
#endif