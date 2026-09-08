# LossCalculator

[Back to central documentation](Home.md)

Computes error between predicted and expected values. Used as the first step in the backwards pass.

---
---
---

## Shared Virtual Methods

### Constructor

*Signature:* `<LossCalculator concrete subclass>()`

Creates a new loss calculator.

---

### Methods

#### compute

*Signature:* `virtual double compute(xt::xarray predicted, xt::xarray expected) const = 0`

Returns the loss between `predicted` and `expected`, as computed by this calculator.

**Parameters**

* `predicted` (`xt::xarray<double>`): Network's predictions for a given input.
* `expected` (`xt::xarray<double>`): What the network should have predicted for the input.

**Returns**

* `double`: Loss between `predicted` and `expected`.

---

#### compute_gradient

*Signature:* `virtual xt::xarray compute_gradient(xt::xarray predicted, xt::xarray expected) const = 0`

Returns the tensor-valued gradient of the loss, between `predicted` and `expected`, as computed by this calculator.

**Parameters**

* `predicted` (`xt::xarray<double>`): Network's predictions for a given input.
* `expected` (`xt::xarray<double>`): What the network should have predicted for the input.

**Returns**

* `xt::xarray<double>`: Gradient of the loss between `predicted` and `expected`.

---

#### shared_ptr_deep_copy

*Signature:* `std::shared_ptr<LossCalculator> shared_ptr_deep_copy() const = 0`

Returns a deep pointer copy of this loss calculator. The deep copy cannot be used to modify the original.

**Returns**
* `std::shared_ptr<LossCalculator>`: Deep copy of the calculator.

---

#### to_string

*Signature:* `std::string to_string() const`

Returns the calculator's identifying string.

Defaults to "loss_calculator" if not overridden by an implementing class.

**Returns**
* `std::string`: Identifier of the loss calculator.

---

### Operator Overloads

#### output stream insertion (<<)

*Signature:* 
```
template<typename CharT, typename Traits>
friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const LossCalculator& calc)
```

Exports `calc` to the output stream `output_stream`, returning `output_stream` with `calc`'s information inside.

**Parameters**

* `output_stream` (`std::basic_ostream<CharT, Traits>&`): Stream to put the loss calculator into.
* `calc` (`const LossCalculator&`): LossCalculator object to export.

**Returns**

* `std::basic_ostream<CharT, Traits>&`: `output_stream` with string representation of `calc` inserted.


---
---
---


## CrossEntropy

Computes cross-entropy loss.

Cross-entropy is equal to $\sum_{y} y_{predicted} * ln(y_{expected})$, where `y` is each element of the tensor-valued predicted and expected values. The minimum value of the expected output is `CrossEntropy::epsilon`, a value which prevents taking ln(0).

Good for classification problems. Imposes more loss for being farther from the expected output.

---

### Methods

#### compute

*Signature:* `double compute(xt::xarray predicted, xt::xarray expected) const override`

Returns the computed cross-entropy loss between `predicted` and `expected`.

**Parameters**

* `predicted` (`xt::xarray<double>`): Model's predictions for a given input. Non-empty.
* `expected` (`xt::xarray<double>`): What the model should have predicted for a given input. Has the same shape as `predicted`.

**Returns**

* `double`: Cross entropy loss between `predicted` and `expected`.

---

#### compute_gradient

*Signature:* `xt::xarray compute_gradient(xt::xarray predicted, xt::xarray expected) const override`

Returns the gradient of cross-entropy loss between `predicted` and `expected`.

**Parameters**

* `predicted` (`xt::xarray<double>`): Model's predictions for a given input. Non-empty.
* `expected` (`xt::xarray<double>`): What the model should have predicted for a given input. Has the same number of elements as `predicted`.

**Returns**

* `xt::xarray<double>`: Gradient of cross-entropy loss between `predicted` and `expected`.

---

#### to_string

*Signature:* `std::string to_string() const override`

Returns the string "cross_entropy".

**Returns**

* `std::string`: String representation of the loss calculator.


---
---
---

## MeanSquaredError

Computes Mean Squared Error (MSE) loss.

Equals the sum of squared differences between corresponding elements in the predicted and expected values, divided by 2x the number of elements in the predicted value.

Assuming there are N elements in the predicted value tensor, `p` is the predicted tensor, and `e` is the expected tensor, MSE loss is given by:  
$\frac{\sum_{i=1}^{N} (p_i - e_i)^2}{2N}$

---

### Methods

#### compute

*Signature:* `double compute(xt::xarray predicted, xt::xarray expected) const override`

Returns the computed Mean Squared Error loss between `predicted` and `expected`.

For each element in the output, MSE subtracts corresponding elements of the predicted and expected values, then squares the difference. The loss is the sum of the squared differences, divided by the number of elements in the predicted value, divided by 2.

**Parameters**

* `predicted` (`xt::xarray<double>`): Model's predictions for a given input. Non-empty.
* `expected` (`xt::xarray<double>`): What the model should have predicted for a given input. Has the same shape as `predicted`.

**Returns**

* `double`: MSE loss between `predicted` and `expected`.

---

#### compute_gradient

*Signature:* `xt::xarray compute_gradient(xt::xarray predicted, xt::xarray expected) const override`

Returns the gradient of MSE loss between `predicted` and `expected`.

Computes the gradient data by taking the difference between predicted and expected values divided by the size of the predictions.

**Parameters**

* `predicted` (`xt::xarray<double>`): Model's predictions for a given input. Non-empty.
* `expected` (`xt::xarray<double>`): What the model should have predicted for a given input. Has the same number of elements as `predicted`.

**Returns**

* `xt::xarray<double>`: Gradient of MSE loss between `predicted` and `expected`.

---

#### to_string

*Signature:* `std::string to_string() const override`

Returns the string "mean_squared_error".

**Returns**

* `std::string`: String representation of the MSE calculator.
