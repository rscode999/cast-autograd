#ifndef CAST_TENSOR_OPERATOR_
#define CAST_TENSOR_OPERATOR_

#include <xtensor/containers/xarray.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


namespace cast {



const int32_t ARBITARY_INPUT_COUNT = -99;
const int32_t ARBITARY_OUTPUT_COUNT = -100;


/**
 * Computes an operation on one or more tensors.
 * 
 * Tracks which operators come before and after this operator, by 0-based numerical index.
 */
class NetworkComponent : public std::enable_shared_from_this<NetworkComponent> {
protected:
    /**
     * Indices to all tensors that are computed, just prior to this operator receiving them.
     * Mapping: branch number -> index of predecessor
     */
    std::unordered_map<int32_t, int32_t> predecessors_;
    
    /**
     * Indices to all tensors that are computed, just after this operator calculated them.
     * Mapping: branch number -> index of successor
     */
    std::unordered_map<int32_t, int32_t> successors_;

    /**
    * EXPERIMENTAL
    */
    int32_t branch_id_ = -69;

    /**
    * Number of inputs allowed by this operation
    */
    int32_t n_inputs_ = ARBITARY_INPUT_COUNT;

    /**
    * Number of outputs given by this operation
    */
    int32_t n_outputs_ = ARBITARY_OUTPUT_COUNT;

public:
    friend class Network;
    

    /**
    * @return number of input tensors used by this operator. Equals `ARBITARY_INPUT_COUNT` if unlimited tensors are accepted.
    */
    int32_t n_inputs() const {
        return n_inputs_;
    }

    /**
    * @return number of output tensors given by this operator. Equals `ARBITARY_OUTPUT_COUNT` if unlimited tensors can be given.
    */
    int32_t n_outputs() const {
        return n_outputs_;
    }


    /**
    * @return indices to this operator's inputs
    */
    std::unordered_map<int32_t, int32_t> predecessors() const {
        return predecessors_;
    }


    /**
    * @return indices to this operator's outputs
    */
    std::unordered_map<int32_t, int32_t> successors() const {
        return successors_;
    }


    /**
     * Returns the results of this operation on `inputs`.
     *
     * The operator can have one or more inputs, and one or more outputs
     * @param inputs tensors to compute this operation on
     * @return results of this operator on `inputs`
     */
    virtual std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> inputs) = 0;

    /**
     * Returns the backwards pass of this operation on `upstream_gradients`.
     *
     * The operator can have one or more inputs, and one or more outputs
     * @param upstream_gradients gradients from the previous operator
     * @return results of the operator's backwards pass on `upstream_gradients`
     */
    virtual std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> upstream_gradients) = 0;

    /**
     * @return identifying string of this operator
     */
    virtual std::string name() const {
        return "tensor_operator";
    }

    /**
     * Properly destroys a tensor operator
     */
    virtual ~NetworkComponent() = default;
};




}
#endif