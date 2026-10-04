# Network

[Back to central documentation](Home.md)

Trainable predictor with user-defined structure.

Components (activation functions, layers, branching structure...) are added one at a time. A network can have at most 2 billion components.

The network's loss calculator and optimizer are also added individually.
The network makes deep copies of the loss calculator and optimizer, so they cannot be modified from outside the network.
Use the `loss_calculator` and `optimizer` methods to get access to the pointers used by the network.

After building the desired architecture, the network must be enabled through the `enable` method to train it.
To be enabled, a network must have a loss calculator, optimizer, and at least one layer. The network must have exactly one unterminated branch ([more information about branches](Home.md#network-branch-management)).  
The `disable` method allows the network to be modified again.

When assigned to another network object, a network deep-copies its layers, so modifying the new network doesn't affect the original.  
A newly assigned network is disabled, regardless of whether the old network is enabled. Stored gradients from the old network, but not the old network's optimizer data, are copied.

---
---
---

### Constructor

*Signature:* `Network()`

Creates a new, empty network.

The new network is not enabled, with no components, loss calculator, or optimizer.

---

### Getters

#### active_branch_ids

*Signature:* `std::unordered_set<int32_t> active_branch_ids() const`

Returns the set of valid branch IDs in the network.

A valid branch ID is one that can be added to.
Branches that have been merged (and thus permanently terminated) are not included in the output.

**Returns**
* `std::unordered_set<int32_t>`: All valid branch IDs.

---

#### active_branch_id_heads

*Signature:* `std::unordered_map<int32_t, int32_t> active_branch_id_heads() const`

Returns a mapping of all valid branch IDs to the component ID of the branch's final component.

A valid branch ID is one that can be added to.
Branches that have been merged (and thus permanently terminated) are not included in the output.

A component's ID is the 0-based order in which the component was added to the network.
The first component added has an ID of 0. The second has an ID of 1, and so on.

**Returns**
* `std::unordered_map<int32_t, int32_t>`: Mapping of branch ID -> component ID of the branch's head.

---

#### batch_size

*Signature:* `int32_t batch_size() const`

Returns the batch size currently used by this network. Returns 0 if the network does not train in batches.

**Returns**
* `int32_t`: Network's batch size.

**Exceptions**
* `cast::bad_network_config`: If the network's loss calculator is not set.

---

#### component_at

*Signature:* `std::shared_ptr<NetworkComponent> component_at(int32_t component_id) const`

Returns a pointer to the component with ID `component_id`.

A component's ID is the 0-based order in which the component was added to the network.
ID 0 is the first component added, 1 is the second component added, and so on.

The returned pointer cannot be used to modify the network's component.

**Parameters**
* `component_id` (`int32_t`): ID to access. At least 0, and less than the number of components added so far.

**Returns**
* `std::shared_ptr<NetworkComponent>`: Component with the given ID.

---

#### is_enabled

*Signature:* `bool is_enabled() const`

Returns whether the network is ready for training and optimization.

**Returns**

* `bool`: Whether the network is enabled.

---

#### loss_calculator

*Signature:* `std::shared_ptr<LossCalculator> loss_calculator() const`

Returns the network's loss calculator, or `nullptr` if no loss calculator is assigned.

The value returned is a shallow copy.

**Returns**
* `std::shared_ptr<LossCalculator>`: Network's loss calculator, as a shallow copy.

---

#### optimizer

*Signature:* `std::shared_ptr<Optimizer> optimizer() const`

Returns the network's optimizer, or `nullptr` if no optimizer is assigned.

The value returned is a shallow copy. The returned pointer can be used to modify the network's optimizer.

**Returns**
* `std::shared_ptr<Optimizer>`: Network's optimizer, as a shallow copy.

---

### Setters

#### add_combiner

*Signature:* `void add_combiner(std::initializer_list<int32_t> branch_ids_to_combine, int32_t branch_id = 0, std::source_location loc = std::source_location::current())`

Adds a combiner, merging the branch IDs given in `branch_ids_to_combine`, to branch `branch_id`.

If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, this method throws `cast::bad_component_addition`.

`cast::bad_component_addition` is also thrown if any of the branch IDs in `branch_ids_to_combine` is out of the range [0, <number of branches used in the network - 1>], has already been merged, or equals `branch_id` (combiners cannot merge their own branch).

A Combiner cannot be the first component added to a network.

To use this method, the network cannot be enabled. 

Example usage
```
//`net` has active branch IDs 0, 1, and 4

//Adds a combiner to branch 4, merging branches 0 and 1 into branch 4
net.add_combiner({0,1}, 4); 
```


**Parameters**

* `branch_ids_to_combine` (`std::initializer_list<int32_t>`): List of branch IDs to merge. Non-empty.
* `branch_id` (`int32_t`): Branch to add the new combiner to.
* `loc` (`std::source_location`): For debugging only- callers should not modify this parameter.

**Exceptions**

* `cast::bad_network_config`: Thrown if the network is enabled.
* `cast::bad_component_addition`: Thrown if `branch_id` is negative, out of range, or already combined; if any branch ID to combine is out of range, already combined, or equals `branch_id`; or if attempting to add a combiner as the first component.
* `std::out_of_range`: If more than 2 billion components (operators, splitters, or combiners) have been added to the network.

---

#### add_operator

*Signature:* `void add_operator(std::shared_ptr<Operator> op, int32_t branch_id = 0, std::source_location loc = std::source_location::current())`

Adds the Operator `op` to the end of branch `branch_id`.

An Operator is a subclass of `Layer` or `ActivationFunction`.

The operator pointer is deep-copied, so the operator pointer cannot be used to modify the network's new operator.

To use this method, the network cannot be enabled.

**Parameters**

* `op` (`std::shared_ptr<Operator>`): Operator to add to a branch. Not equal to `nullptr`.
* `branch_id` (`int32_t`): Branch to add the new operator to.
* `loc` (`std::source_location`): For debugging only- callers should not modify this parameter.

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.
* `cast::bad_component_addition`: If `branch_id` is negative, out of range, or corresponds to a merged branch.
* `std::out_of_range`: If more than 2 billion components (operators, splitters, or combiners) have been added to the network.

---

#### add_splitter

*Signature:* `void add_splitter(int32_t branch_count, int32_t branch_id = 0, std::source_location loc = std::source_location::current())`

Adds a splitter that distributes execution across `branch_count` new branches, to branch `branch_id`.

To use this method, the network cannot be enabled.

Example usage
```
//`net` has active branches with ID 0, 1, 4. The latest branch created has ID 4.

//Creates 3 branches (with IDs 5, 6, 7), splitting from branch 1
net.add_splitter(3, 1);
```

**Parameters**

* `branch_count` (`int32_t`): Number of branches to split execution into. Must be at least 2.
* `branch_id` (`int32_t`): Branch to add the new splitter to.
* `loc` (`std::source_location`): For debugging only- callers should not modify this parameter.

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.
* `cast::bad_component_addition`: If `branch_id` is negative, out of range, or corresponds to a merged branch.
* `std::out_of_range`: If more than 2 billion components (operators, splitters, or combiners) have been added to the network, or the network has created more than 2 billion branches, including the ones that will be created.

---

#### clear_training_state

*Signature:* `void clear_training_state()`

Resets the network optimizer's training data.

Equivalent to removing the network's optimizer, then assigning an idential optimizer with the same hyperparameters.

**Exceptions**
* `cast::bad_network_config`: If the network has no defined optimizer.

---

#### disable

*Signature:* `void disable()`

Disables the network.

Prevents training and optimization, but allows the network to be modified.

---

#### enable

*Signature:* `void enable()`

Enables the network, allowing training and preventing modification, if the enable check passes.

Enable check: The network must have a loss calculator, optimizer, at least one component, and exactly one unterminated branch.

If the enable check fails, this method throws `enable_failed_error` (a subclass of `cast::bad_network_config`, which is itself a subclass of `std::exception`).


**Exceptions**

* `cast::enable_failed_error`: If the enable check fails.

---

#### set_batch_size

*Signature:* `void set_batch_size(int32_t new_batch_size)`

Sets the network's batch size to `new_batch_size`. If `new_batch_size` is 0, the network does not train in batches.

To use this method, the network must be disabled.

**Parameters**
* `new_batch_size` (`int32_t`): Batch size. Non-negative.

**Exceptions**
* `cast::bad_network_config`: If the network's loss calculator is not set, or the network is enabled.

---

#### set_loss_calculator

*Signature:* `void set_loss_calculator(std::shared_ptr<LossCalculator> calc)`

Sets this network's loss calculator to `calc`.

Setting `calc` to `nullptr` removes the network's loss calculator.

To use this method, the network cannot be enabled.

**Parameters**

* `calc` (`std::shared_ptr<LossCalculator>`): New loss calculator to use.

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.

---

#### set_operator_at

*Signature:* `void set_operator_at(int32_t component_id, std::shared_ptr<Operator> op)`

Sets the operator with ID `component_id` to `op`. An Operator is a subclass of ActivationFunction or Layer.
 
A component's ID is the 0-based order in which the component was added to the network.
ID 0 is the first component added, 1 is the second component added, and so on.

The pointer to `op` cannot be used to modify the network's newly altered component.

Note: If the datatypes of `op` and the component with ID `component_id` are not the same,
this method throws a `cast::bad_component_addition` exception.

To use this method, the network must be disabled.

**Parameters**
* `component_id` (`int32_t`): Component number to set. At least 0, and less than the number of components added so far.
* `op` (`std::shared_ptr<Operator>`): Operator to set at the given number.

**Exceptions**
* `cast::bad_component_addition`: If the datatypes of `op` and the component with ID `component_id` differ.
* `cast::bad_network_config`: If the network is enabled.
---

#### set_optimizer

*Signature:* `void set_optimizer(std::shared_ptr<Optimizer> optim)`

Sets this network's optimizer to `optim`.

Setting `optim` to `nullptr` removes the network's optimizer.

To use this method, the network cannot be enabled.

**Parameters**

* `optim` (`std::shared_ptr<Optimizer>`): New optimizer to use.

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.

---

#### set_optimizer_hyperparams

*Signature:* `void set_optimizer_hyperparams(std::initializer_list<double> new_hyperparams)`

Sets the hyperparameters of the network's optimizer.

The length and preconditions of each element in `new_hyperparams` must match those
in the optimizer's `set_hyperparameters` method.

**Parameters**

* `new_hyperparams` (`std::initializer_list<double>`): New hyperparameters to set.

**Exceptions**

* `cast::bad_network_config`: If the network has no optimizer.

---

### Methods

#### forward

*Signature:* `xt::xarray<double> forward(const xt::xarray<double>& input)`

Returns the result of the network's forward pass on `input`.

If the network trains in batches, i.e. `batch_size()` is positive, axis 0 of `input` is treated as the batch index.
Example: If `batch_size()` is 2, axis 0 must have 2 elements, each of which are valid inputs to the network.

To use this method, the network must be enabled.

**Parameters**

* `input` (`const xt::xarray<double>&`): Tensor to compute forward pass on.

**Returns**

* `xt::xarray<double>`: Result of the forward pass.

**Exceptions**

* `cast::bad_network_config`: If the network is not enabled.
* `cast::shape_error`: If input shapes are incompatible between successive network components.

---

#### backward

*Signature:* `void backward(const xt::xarray<double>& predicted, const xt::xarray<double>& expected)`

Computes the backward pass, beginning with loss between `predicted` and `expected`.

If `batch_size()` is nonzero, axis 0 of each input must contain `batch_size()` elements, where each element is a valid input to the network.

Stores updated gradients inside the network layers, for use by the network's optimizer.

The network must be enabled to use this method.

**Parameters**

* `predicted` (`const xt::xarray<double>&`): Network's prediction for a given input.
* `expected` (`const xt::xarray<double>&`): What the network should have predicted instead of `predicted`. Has the same shape as `predicted`.

**Exceptions**

* `cast::bad_network_config`: If the network is not enabled.
* `cast::shape_error`: If input shapes are incompatible between successive network components.
* `std::runtime_error`: If given batched training data when `batch_size()` is 0. Error message is "Dot shape mismatch".

---

#### optimize

*Signature:* `void optimize(bool zero_grad = true)`

Runs an optimization pass on the network's layers.

Uses the network's stored optimizer and the gradients computed from the `backward` method.

To use this method, the network must be enabled.

**Parameters**

* `zero_grad` (`bool`): Whether to set all operator's gradients to 0 after computing the optimization pass.

**Exceptions**

* `cast::bad_network_config`: If the network is not enabled.

---

### Operator Overloads

#### output stream insertion (<<)

*Signature:*
``` 
template<typename CharT, typename Traits>
friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Network& network)
```

Exports `network` to the output stream `output_stream`, returning `output_stream` with `network`'s information inside.

Information includes: enabled/disabled status, loss calculator, optimizer, and each layer. Each network part is on a new line.

Works for any output stream, including `std::wcout`, the wide-character output.

**Parameters**

* `output_stream` (`std::basic_ostream<CharT, Traits>&`): Stream to put the network into.
* `network` (`const Network&`): Network object to export.

**Returns**

* `std::basic_ostream<CharT, Traits>&`: `output_stream` with `network` inserted.