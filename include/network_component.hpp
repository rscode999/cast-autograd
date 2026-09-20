#ifndef CAST_NETWORK_COMPONENT_
#define CAST_NETWORK_COMPONENT_

#include "cast_exceptions.hpp"
#include "ostream_manip.hpp"

#include <source_location>
#include <xtensor/containers/xarray.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


namespace cast {


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
    * If `condition` is false, throws a `cast::shape_error` with message `error_message`.
    *
    * Will always do its check, regardless of if `NDEBUG` is defined.
    * @param condition condition to check
    * @param failure_message error message to display if `condition` is false
    * @param assert_location location where the assertion was enforced
    */
    void assert_tensor_shape_(bool condition, std::string failure_message = "", std::source_location assert_location = std::source_location::current()) const {
        if(!condition) {
            throw shape_error(failure_message, assert_location);
        }
    }

    /**
    * Checks if this component has a non-negative branch ID (that is, it was assigned). If not, throws `cast::assertion_error`.
    *
    * Does nothing if `NDEBUG` is defined.
    */
    void assert_branch_id_assigned_() const {
        #ifndef NDEBUG
        str_assert(branch_id_ >= 0, to_string() + " has no assigned branch ID; got " + std::to_string(branch_id_));
        #endif
    }


public:
    friend class Network;

    /**
    * @return deep pointer copy of this network component. The deep copy cannot be used to modify the original.
    */
    virtual std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const = 0;


    /**
    * @return the branch ID number that this component is assigned to.
    * If unassigned, throws the `cast::unassigned_branch_error` exception.
    */
    int32_t branch_id() const {
        if(branch_id_ < 0) {
            throw unassigned_branch_error("Branch ID is " + (branch_id_==UNASSIGNED_BRANCH_ID ? "UNASSIGNED" : std::to_string((int32_t)branch_id_)));
        }
        return branch_id_;
    }

    

    /**
    * Returns all predecessor and successor branch IDs, in a string.
    *
    * The string is in the format "predecessors: {index}, branch {branch ID}..., successors: {index}, branch {branch ID}".
    *
    * Largely for debugging.
    * @return predecessor and successor branches
    */
    std::string connections_to_string() const {
        std::string out = "predecessors: {";
        for(std::pair<int32_t, int32_t> pred : predecessors_) {
            out += std::to_string(pred.second) + ", branch " + std::to_string(pred.first) + "; ";
        }
        out += "}, successors: {";
        for(std::pair<int32_t, int32_t> succ : successors_) {
            out += std::to_string(succ.second) + ", branch " + std::to_string(succ.first) + "; ";
        }
        out += "}";
        
        return out;
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
     * @return information about the component, including type and parameters
     */
    virtual std::string to_string() const {
        return "network_component";
    }


    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////


    /**
     * Returns the results of this operation on `inputs`.
     * @param input tensors to compute this operation on
     * @return result of this operator on `inputs`
     */
    virtual xt::xarray<double> forward(xt::xarray<double> input) = 0;

    /**
     * Returns the backwards pass of this component on `upstream_gradients`.
     * @param upstream_gradients gradients from the previous operator
     * @return results of the operator's backwards pass on `upstream_gradients`
     */
    virtual xt::xarray<double> backward(xt::xarray<double> upstream_gradients) = 0;


    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////


    /**
     * Properly destroys a network component
     */
    virtual ~NetworkComponent() = default;


    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    

    /**
    * Exports `component` to the output stream `output_stream`, returning `output_stream` with `component`'s information inside.
    *
    * Works on any output stream, including `std::wcout`, the wide-character output.
    * @param output_stream stream to put the component into
    * @param component NetworkComponent object to export
    * @return `output_stream` with `component` inserted
    */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const NetworkComponent& component);
};


template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const NetworkComponent& component) {
    std::string component_str = component.to_string();

    std::string component_branch_str;
    try {
        component_branch_str = std::to_string(component.branch_id());
    }
    catch(unassigned_branch_error& e) {
        component_branch_str = "UNASSIGNED";
    }

    //export component as string (converted to character type of output stream)
    output_stream << std::basic_string<CharT>(component_str.begin(), component_str.end());

    //display branch
    output_stream << ", branch " <<  std::basic_string<CharT>(component_branch_str.begin(), component_branch_str.end());

    //export more if the output stream is in verbose mode
    if(output_stream.iword(get_display_idx()) == 1) {
        std::string component_connections_str = component.connections_to_string();
        output_stream << "\n" << "    " << std::basic_string<CharT>(component_connections_str.begin(), component_connections_str.end());
    }
    return output_stream;
}




}
#endif