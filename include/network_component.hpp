#ifndef CAST_NETWORK_COMPONENT_
#define CAST_NETWORK_COMPONENT_

#include "cast_exceptions.hpp"

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
* Initial value for a branch ID.
*
* This value is negative.
*/
const int32_t UNASSIGNED_BRANCH_ID = -69;

/**
 * Computes an operation on one or more tensors.
 * 
 * Tracks which operators come before and after this operator, by 0-based numerical index.
 */
class NetworkComponent : public std::enable_shared_from_this<NetworkComponent> {
protected:
    /**
     * Indices to all components before this one.
     * Mapping: branch number -> index of predecessor
     */
    std::unordered_map<int32_t, int32_t> predecessors_;
    
    /**
     * Indices to all components after this one.
     * Mapping: branch number -> index of successor
     */
    std::unordered_map<int32_t, int32_t> successors_;

    /**
    * Branch index that this component is added to.
    *
    * This field cannot be changed by outside users. A Network changes this value with friend access.
    */
    int32_t branch_id_ = UNASSIGNED_BRANCH_ID;

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
    * @return deep pointer copy of this network component. The deep copy cannot be used to modify the original.
    */
    virtual std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const = 0;


    /**
    * Checks if this component has a non-negative branch ID (that is, it was assigned). If not, throws `cast::assertion_error`.
    */
    void assert_branch_id_assigned() {
        str_assert(branch_id_ >= 0, name() + " has no assigned branch ID; got " + std::to_string(branch_id_));
    }


    /**
    * @return the branch ID that this component is assigned to. If unassigned, returns `UNASSIGNED_BRANCH_ID`.
    */
    int32_t branch_id() const {
        return branch_id_;
    }


    /**
     * @return identifying string of this component. Defaults to "network_component" if not overridden
     */
    virtual std::string name() const {
        return "network_component";
    }


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
    * @return indices to this operator's inputs. Maps: branch ID -> index of predecessor
    */
    std::unordered_map<int32_t, int32_t> predecessors() const {
        return predecessors_;
    }


    /**
    * @return indices to this operator's outputs. Maps: branch ID -> index of successor
    */
    std::unordered_map<int32_t, int32_t> successors() const {
        return successors_;
    }



    /**
     * Returns the results of this operation on `inputs`.
     *
     * The component can have one or more inputs, and one or more outputs
     * @param inputs tensors to compute this operation on
     * @return results of this operator on `inputs`
     */
    virtual std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> inputs) = 0;

    /**
     * Returns the backwards pass of this component on `upstream_gradients`.
     *
     * The component can have one or more inputs, and one or more outputs
     * @param upstream_gradients gradients from the previous operator
     * @return results of the operator's backwards pass on `upstream_gradients`
     */
    virtual std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> upstream_gradients) = 0;



    /**
     * Properly destroys a network component
     */
    virtual ~NetworkComponent() = default;
};




}
#endif