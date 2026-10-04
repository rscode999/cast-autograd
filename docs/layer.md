# Layer

[Back to central documentation](Home.md)

Contains trainable weights. Only Layers are affected by optimization passes.

Subclass of `Operator`.

---
---
---

## Shared Virtual Methods

### Getters

#### gradients

*Signature:* `std::vector<xt::xarray<double>>& gradients()`

Returns the gradients of the weights, biases, etc. of this layer.

The parameter that each gradient corresponds to depends on the specific layer subclass.

**Returns**

* `std::vector<xt::xarray<double>>&`: Gradients of the layer's parameters.

---

#### parameters

*Signature:* `std::vector<xt::xarray<double>>& parameters()`

Returns parameters (weights, biases, ...) of this layer.

The role of each parameter index depends on the specific layer subclass.

**Returns**

* `std::vector<xt::xarray<double>>&`: Parameters of the layer.

---
---
---

## Linear1d

### Constructor

*Signature:* `Linear1d(int32_t input_dimension, int32_t output_dimension)`

Creates a 1d linear layer with `input_dimension` inputs and `output_dimension` outputs.

Weights and biases are randomly initialized using a normal distribution with mean 0 and standard deviation 1.  
Gradients are initialized to zeros.

A 2d weight matrix is stored in `parameters() [0]`. A 1d bias vector is in `parameters() [1]`.
Gradients of the weights and biases are stored in `gradients() [0]` and `gradients() [1]`, respectively.

**Parameters**

* `input_dimension` (`int32_t`): Required size of input vectors. Positive.
* `output_dimension` (`int32_t`): Size of output vectors. Positive.

**Exceptions**

* `std::runtime_error` (via `str_assert`): Thrown if `input_dimension` or `output_dimension` is not positive.

---

### Getters

#### to_string

*Signature:* `std::string to_string() const override`

Returns the string "linear1d ({input dimension}, {output dimension})".

Example: "linear1d (2, 4)", for a Linear1d layer with 2 inputs and 4 outputs

**Returns**

* `std::string`: The string representation of the layer.

---

### Methods

#### forward

*Signature:* `xt::xarray<double> forward(const xt::xarray<double>& input) override`

Returns the result of the linear forward pass on `input`.

**Parameters**

* `input` (`const xt::xarray<double>&`): List containing the layer input. Has exactly 1 element.

**Returns**

* `xt::xarray<double>`: Forward pass result.

**Exceptions**

* `cast::shape_error`: If input rank is not 1, or if input vector length does not match the layer's input dimension.

---

#### backward

*Signature:* `xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients) override`

Returns the gradients with respect to this layer and `upstream_gradients`, updating this layer's gradients.

**Parameters**

* `upstream_gradients` (`const xt::xarray<double>&`): Gradients from this layer's successor. Contains a single 1d vector.

**Returns**

* `xt::xarray<double>`: dY/dL, where Y is the overall derivative and L is this layer's data, contained in index 0 of the output.
