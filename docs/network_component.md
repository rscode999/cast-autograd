# NetworkComponent

[Back to central documentation](Home.md)

Node in a network.

When added to a Network, a NetworkComponent receives a unique numerical index. The NetworkComponent stores one or more predecessor indices, and one or more successor indices.

NetworkComponents are capable of creating deep pointer copies of themselves, so outside users cannot modify a class' private pointers.

*Subclasses:*
* [Operator](operator.md)
    * [ActivationFunction](activation_function.md)
    * [Layer](layer.md)
* [Splitter, Combiner](control_flow.md)

---
---
---

### Constructor

Each subclass of `NetworkComponent` has its own constructor.

---

### Shared Pointer Copying

#### shared_ptr_deep_copy

*Signature:* `virtual std::shared_ptr<NetworkComponent> shared_ptr_deep_copy() const = 0`

Returns a deep pointer copy of this network component. The deep copy cannot be used to modify the original.

**Returns**

* `std::shared_ptr<NetworkComponent>`: Shared pointer to a deep copy of the component.

---

### Getters

#### branch_id

*Signature:* `int32_t branch_id() const`

Returns the branch ID number that this component is assigned to. If unassigned, throws the `cast::unassigned_branch_error` exception.

**Returns**

* `int32_t`: The assigned branch ID.

**Exceptions**

* `cast::unassigned_branch_error`: If the component has not been assigned a branch ID.

---


#### connections_to_string

*Signature:* `std::string connections_to_string() const`

Returns all predecessor and successor branch IDs in a string.

The string is in the format "predecessors: {index}, branch {branch ID}..., successors: {index}, branch {branch ID}".  
Example: "{predecessors: 1, branch 0; 2, branch 4}, {successors: 3, branch 2}"

Largely for debugging.

**Returns**

* `std::string`: Formatted string containing predecessor and successor branch connections.

---

#### predecessors

*Signature:* `std::unordered_map<int32_t, int32_t> predecessors() const`

Returns indices to this operator's inputs, mapping branch ID to predecessor's ID.

**Returns**

* `std::unordered_map<int32_t, int32_t>`: A map of branch IDs to predecessor IDs.

---

#### successors

*Signature:* `std::unordered_map<int32_t, int32_t> successors() const`

Returns indices to this operator's outputs, mapping branch ID to ID of successor.

**Returns**

* `std::unordered_map<int32_t, int32_t>`: A map of branch IDs to successor IDs.

---

#### to_string

*Signature:* `virtual std::string to_string() const`

Returns information about the component, including component name, parameters, and configuration.

If not overridden by a subclass, returns "network_component".

**Returns**

* `std::string`: String representation of the component.

---

### Methods

#### forward

*Signature:* `xt::xarray<double> forward(const xt::xarray<double>& input) = 0`

Returns the results of this operation on `input`.

Axis 0 of `upstream_gradients` separates elements in a batch.
Even if not batch training, inputs to this method must be placed in batches.

**Parameters**

* `input` (`const xt::xarray<double>&`): Tensors to compute this operation on.

**Returns**

* `xt::xarray<double>`: Results of this operator on `input`.

---

#### backward

*Signature:* `virtual xt::xarray<double> backward(const xt::xarray<double>& upstream_gradients) = 0`

Returns the backwards pass of this component on `upstream_gradients`.

Axis 0 of `upstream_gradients` separates elements in a batch.
Even if not batch training, inputs to this method must be placed in batches.

**Parameters**

* `upstream_gradients` (`const xt::xarray<double>&`): Gradients from the previous operator. Has at least 2 dimensions.

**Returns**

* `xt::xarray<double>`: Results of the operator's backwards pass on `upstream_gradients`.

---

### Operator Overloads

#### output stream insertion (<<)

*Signature:* 
```
template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const NetworkComponent& component)
```

Exports `component` to the output stream `output_stream`, returning `output_stream` with `component`'s information inside.

This method works for `std::cout`, `std::cerr`, and `std::wcout`.

**Parameters**

* `output_stream` (`std::basic_ostream<CharT, Traits>&`): Stream to put the component into.
* `component` (`const NetworkComponent&`): NetworkComponent object to export.

**Returns**

* `std::basic_ostream<CharT, Traits>&`: The provided output stream with `component` inserted.
