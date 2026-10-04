# ActivationFunction

[Back to central documentation](Home.md)

Network component that computes an element-wise function for all of its inputs.

Abstract class. Subclass of `Operator`.

---
---
---

## Shared Virtual Methods

### Methods

#### forward

*Signature:* `virtual xt::xarray<double> forward(const xt::xarray<double>& input) = 0`

Computes the element-wise activation function applied to each element in `input`.

**Parameters**

* `input` (`const xt::xarray<double>&`): List of values to compute. Non-empty.

**Returns**

* `<xt::xarray<double>`: Computed activation values for each element of `input`.

---

#### backward

*Signature:* `virtual xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients) = 0`

Computes the derivative of the activation function applied to each parameter of `upstream_gradients`.

**Parameters**

* `upstream_gradients` (`const xt::xarray<double>&`): List of values to compute. Non-empty.

**Returns**

* `xt::xarray<double>`: Derivative values for each element of `upstream_gradients`.

---
---
---

## ReLU

Applies the Rectified Linear Unit (ReLU) function to each element of its input.

For a number x, the ReLU function is given by
$ReLU(x) = max(0, x)$.

---

### Constructor

*Signature:* `ReLU()`

Creates a new ReLU object.

---

### Getters

#### to_string

*Signature:* `std::string to_string() const`

Returns the string "relu".

**Returns**
* `std::string`: String representation of the object.

---

### Methods

#### forward

*Signature:* `xt::xarray<double> forward(const xt::xarray<double>& input)`

Computes the element-wise activation function applied to each parameter in `input`.

**Parameters**

* `input` (`const xt::xarray<double>&`): List of values to compute. Non-empty.

**Returns**

* `xt::xarray<double>`: Computed activation values for each element of `input`.

---

#### backward

*Signature:* `xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients)`

Applies the derivative of ReLU to each parameter of `upstream_gradients`.

The derivative of ReLU(x) is 1 if x is positive, 0 if x is negative.  
At x=0, the ReLU derivative is defined to be 1.


**Parameters**

* `upstream_gradients` (`const xt::xarray<double>&`): List of values to compute. Non-empty.

**Returns**

* `xt::xarray<double>`: Derivative values for each element of `upstream_gradients`.

---
---
---

## Sigmoid

Applies the sigmoid function to each element of its inputs.

For a number x, the Sigmoid function is given by
$sigmoid(x) = \frac{1}{1 + e^-x}$.

The result is always between 0 and 1, exclusive on both ends.

---

### Constructor

*Signature:* `Sigmoid()`

Creates a new Sigmoid object.

---

### Getters

#### to_string

*Signature:* `std::string to_string() const`

Returns the string "sigmoid".

**Returns**
* `std::string`: String representation of the object.

---

### Methods

#### forward

*Signature:* `xt::xarray<double> forward(const xt::xarray<double>& input)`

Computes the element-wise activation function applied to each element in `input`.

Calling this method is required to compute the Sigmoid's `backward` method.

**Parameters**

* `input` (`const xt::xarray<double>&`): List of values to compute. Non-empty.

**Returns**

* `xt::xarray<double>`: Computed activation values for each element of `input`.

---

#### backward

*Signature:* `xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients)`

Computes the derivative of the activation function applied to each element of `upstream_gradients`.

NOTE: The `forward` method must have been called beforehand. To compute properly, the input to this method must be the output of the `forward` method call.

**Parameters**

* `upstream_gradients` (`const xt::xarray<double>&`): List of values to compute. Non-empty.

**Returns**

* `xt::xarray<double>`: Derivative values for each element of `upstream_gradients`.

---
---
---

## Softmax

Applies the Softmax vector-valued function to each input.

For each input vector, Softmax generates a probability distribution over all inputs, scaled by the magnitude of each input's elements. The sum of each Softmax output's elements is 1.

Prior to calculation, each input is divided by the Softmax object's temperature coefficient.   
The temperature coefficient increases or decreases the differences between output values. A higher temperature coefficient decreases the difference in input magnitudes, making each output element more uniform.

Softmax is for 1D vectors only.

---

### Constructor

*Signature:* `Softmax(double temperature_coefficient = 1)`

Creates a new Softmax object.

The temperature of the object is set to `temperature_coefficient`.  

**Parameters**
* `temperature_coefficient` (`double`): Amplification or attenuation factor of differences between outputs. Positive.

---

### Getters

#### temperature_coefficient

*Signature:* `double temperature_coefficient() const`

Returns the object's temperature coefficient.

**Returns**
* `double`: Object's amplification or attenuation factor of differences between outputs.


---

#### to_string

*Signature:* `std::string to_string() const`

Returns the string "softmax".

**Returns**
* `std::string`: String representation of the object.

---

### Setters

#### set_temperature_coefficient

*Signature:* `void set_temperature_coefficient(double new_temp_coeff)`

Sets the object's temperature coefficient to `new_temp_coeff`.

**Parameters**
* `new_temp_coeff` (`double`): Temperature coefficient to set. Positive.

---

### Methods

#### forward

*Signature:* `xt::xarray<double> forward(const xt::xarray<double>& input)`

Computes the activation function applied to each element of `input`.

**Parameters**

* `input` (`const xt::xarray<double>&`): List of values to compute. Non-empty, and has exactly 2 axes.

**Returns**

* `xt::xarray<double>`: Computed activation values for each element of `input`.

---

#### backward

*Signature:* `xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients)`

Applies the derivative of Sigmoid to each element of `upstream_gradients`.


**Parameters**

* `upstream_gradients` (`const xt::xarray<double>&`): List of values to compute. Non-empty, and has exactly 2 axes.

**Returns**

* `xt::xarray<double>`: Derivative values for each element of `upstream_gradients`.
