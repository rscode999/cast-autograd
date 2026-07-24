#ifndef CAST_TENSOR_
#define CAST_TENSOR_

#include <ostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>

namespace cast {




class TensorOperator;

/**
 * Wrapper around a `xt::array` that tracks previous tensors
 */
class TensorNode {
private:

    /**
     * Pointer to the operator that created this node
     */
    std::shared_ptr<TensorOperator> prev_operator_;

    /**
     * Tensor data stored inside this node
     */
    xt::xarray<double> data_;

    /**
     * Gradients of the data stored in the node
     */
    xt::xarray<double> gradients_;

    /**
    * Creates a new Tensor. All fields are uninitialized.
    * 
    * Private constructor, for use in a `TensorOperator` base class.
    */
    TensorNode() = default;


public:
    friend class TensorOperator;


    /**
     * Creates a new tensor containing `input`.
     * 
     * The tensor has no registered previous operators.
     * Gradients are all 1's.
     */
    TensorNode(xt::xarray<double> input) : data_(input) {
        gradients_ = xt::ones_like(input);
    }

    /**
     * @return tensor data inside this object
     */
     xt::xarray<double> data() const noexcept;

    /**
     * @return gradient of the data inside this node
     */
     xt::xarray<double> gradients() const noexcept;


    /**
     * Exports `node` to `output_stream`, returning a reference to `output_stream` with `node` inside.
     * 
     * Gives the tensor's data and gradients.
     * 
     * @param output_stream stream to send the list into
     * @param node tensor node to export
     * @return reference to `output_stream` with `node` inside
     */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const TensorNode& node);
};


template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const TensorNode& node) {
    output_stream << "TensorNode" << "\n";
    output_stream << "data" << "\n";
    output_stream << node.data_ << "\n";
    output_stream << "gradients" << "\n";
    output_stream << node.gradients_;
    
    return output_stream;
}



}
#endif