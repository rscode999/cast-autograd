# Optimizer

[Back to central documentation](Home.md)

Improves the weights of a network's layers.

---
---
---

## Shared Pointer Cloning 

#### shared_ptr_deep_copy

*Signature:* `virtual std::shared_ptr shared_ptr_deep_copy() const = 0`

Returns a deep copy of the optimizer's pointer.

The new pointer cannot be used to modify the original.

**Returns**

* `std::shared_ptr<Optimizer>`: Deep pointer copy of the optimizer.

---

## Shared Virtual Methods

### Getters

#### hyperparameters

*Signature:* `virtual std::vector hyperparameters() const`

Returns the optimizer's hyperparameters.

The meaning of each index in the return value depends on the specific Optimizer subclass.

**Returns**

* `std::vector<double>`: Optimizer's hyperparameters.

---

#### to_string

*Signature:* `virtual std::string to_string() const`

Returns a string representation of the optimizer object and its hyperparameters.

**Returns**

* `std::string`: String representation of the optimizer.

---

### Setters

#### clear_training_state

*Signature:* `void clear_training_state() = 0`

Removes the network components under optimization, as well as any data used for training.

To restore the network components and data, use the `initialize` method.

---

#### set_hyperparameters

*Signature:* `virtual void set_hyperparameters(std::initializer_list new_hyperparams) = 0`

Sets the hyperparameters to `new_hyperparams`.

Length of `new_hyperparams` and preconditions for each hyperparameter depend on the optimizer subclass.

**Parameters**

* `new_hyperparams` (`std::initializer_list<double>`): Hyperparameters to set.

---

### Methods

#### initialize

*Signature:* `virtual void initialize(std::vector<std::shared_ptr>& operators) = 0`

Loads the optimizer with all information needed for training.

**Parameters**

* `operators` (`std::vector<std::shared_ptr<NetworkComponent>>&`): Network components to optimize.

---

#### step

*Signature:* `virtual void step(bool zero_grad) = 0`

Updates the parameters of each Layer object in the optimizer's components, which are given in the `initialize` method.

This method uses each layer's stored gradients.

Any component that is not a subclass of `Layer` is unchanged.

**Parameters**

* `zero_grad` (`bool`): Whether to set each operator's gradients to 0 after computing the optimization pass

---

### Operator Overloads

#### output stream insertion (<<)

*Signature:* 
```
template<typename CharT, typename Traits>
friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Optimizer& optimizer)
```

Exports `optimizer` to the output stream `output_stream`, returning `output_stream` with `optimizer`'s information inside.

Works for output streams of any type, including `std::wcout`, the wide-character output stream.

**Parameters**

* `output_stream` (`std::basic_ostream<CharT, Traits>&`): Stream to put the optimizer into.
* `optimizer` (`const Optimizer&`): Optimizer object to export.

**Returns**

* `std::basic_ostream<CharT, Traits>&`: `output_stream` with `optimizer` inserted.

---
---
---


## SGD

Stochastic Gradient Descent optimizer with momentum.

---

### Constructor

*Signature:* `SGD(double initial_lr, double initial_momentum_coeff)`

Creates a new SGD optimizer with initial learning rate `initial_lr`, and initial momentum coefficient `initial_momentum_coeff`.

**Parameters**

* `initial_lr` (`double`): Initial learning rate to use. Positive.
* `initial_momentum_coeff` (`double`): Initial momentum coefficient to use. Non-negative.

---

### Getters

#### learning_rate

*Signature:* `double learning_rate() const`

Returns the learning rate used by this optimizer.

**Returns**

* `double`: Current learning rate.

---

#### momentum_coefficient

*Signature:* `double momentum_coefficient() const`

Returns the momentum coefficient used by this optimizer.

**Returns**

* `double`: Current momentum coefficient.

---

#### to_string

*Signature:* `std::string to_string() const override`

Returns the string "sgd (learning rate {learning rate}, momentum coefficient {momentum coefficient})".

Example: "sgd (learning rate 0.05, momentum coefficient 0.9)"

**Returns**
* `std::string`: String representation of the SGD optimizer.

---

### Setters

#### clear_training_state

*Signature:* `void clear_training_state()`

Removes all network components under optimization, as well as resetting layer velocities.

---

#### set_hyperparameters

*Signature:* `void set_hyperparameters(std::initializer_list<double> new_hyperparams) override`

Sets the optimizer's learning rate to `new_hyperparams[0]`, and the momentum coefficient to `new_hyperparams[1]`.

**Parameters**

* `new_hyperparams` (`std::initializer_list<double>`): New hyperparameters. Of length 2. Index 0 is positive, index 1 is non-negative.

---

#### set_learning_rate

*Signature:* `void set_learning_rate(double new_learning_rate)`

Sets this SGD optimizer's learning rate to `new_learning_rate`.

**Parameters**

* `new_learning_rate` (`double`): Learning rate to set. Positive.

---

#### set_momentum_coefficient

*Signature:* `void set_momentum_coefficient(double new_momentum_coeff)`

Sets this SGD optimizer's momentum coefficient to `new_momentum_coeff`.

**Parameters**

* `new_momentum_coeff` (`double`): Momentum coefficient to set. Non-negative.

---

### Methods

#### initialize

*Signature:* `void initialize(std::vector<std::shared_ptr<NetworkComponent>>& components) override`

Loads the SGD optimizer with layer velocities taken from `components`.

**Parameters**

* `components` (`std::vector<std::shared_ptr<NetworkComponent>>&`): Network components to optimize. Non-empty, and no element can be `nullptr`.

---

#### step

*Signature:* `void step(bool zero_grad = true) override`

Updates the optimizer's components using SGD.

Any non-layer (i.e. components that are not subclasses of `Layer`) are ignored.

This method must be called after using `initialize`. The components cannot have been modified since calling `initialize`.

**Parameters**

* `zero_grad` (`bool`): Whether to set each operator's gradients to 0, after computing the optimization pass.

---