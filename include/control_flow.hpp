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
private:
    /**
    * Number of separate execution paths taken by this branch
    */
    int32_t branch_count_;

public:
    friend class Network;

    /**
    * Creates a new branch that splits execution into `branch_count` paths.
    * @param branch_count number of paths to split into. At least 2.
    */
    Branch(int32_t branch_count) : branch_count_(branch_count) {
        str_assert(branch_count > 1, "Number of branches must be at least 2; instead got " + std::to_string(branch_count));
        // predecessors_.resize(1);
        // successors_.resize(branch_count);
    }


    std::string name() const override {
        return "branch";
    }


    int32_t branch_count() const {
        return branch_count_;
    }


    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>>) override {
        throw not_implemented();
    }


    std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>>) override {
        throw not_implemented();
    }
};



/**
* Collapses control flow from one or more other branches into the branch that this object is added to.
*/
class Combiner : public TensorOperator {
private:
    std::vector<int32_t> branch_indices_;

public:
    friend class Network;

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


    std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>>) override {
        throw not_implemented();
    }


    std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>>) override {
        throw not_implemented();
    }
};



}
#endif