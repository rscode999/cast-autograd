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
class Tensor {
private:

    /**
     * Pointer to the operator that created this node
     */
    std::shared_ptr<TensorOperator> prev_operator_;

    /**
     * Tensor data stored inside this node
     */
    xt::xarray<double> data_;


public:
    friend class TensorOperator;
    friend class CustomNetwork;
    friend class Network;


    /**
     * Creates a new tensor containing `initial_data`.
     * 
     * The tensor has no registered previous operators.
     * Gradients are all 1's.
     *
     * @param initial_data tensor data to store
     */
    Tensor(xt::xarray<double> initial_data) : data_(initial_data) {
        prev_operator_ = nullptr;
    }

    /**
     * @return tensor data inside this object
     */
     xt::xarray<double> data() const;

     /**
     * Sets the data contents of this tensor to `new_data`.
     * @param new_data data to set
     */
    void set_data(xt::xarray<double> new_data);

    /**
     * Exports `t` to `output_stream`, returning a reference to `output_stream` with `t` inside.
     * 
     * Gives the tensor's data only.
     * 
     * @param output_stream stream to send the list into
     * @param t tensor to export
     * @return reference to `output_stream` with `t` inside
     */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Tensor& t);
};


template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Tensor& t) {
    output_stream << t.data_;
    return output_stream;
}



}
#endif