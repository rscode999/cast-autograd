# Control Flow (Splitters and Combiners)

[Back to central documentation](Home.md)

Special network components for creating and destroying additional branches.

---
---
---

## Splitter

Breaks control flow into two or more separate paths of execution.

Always has one predecessor and multiple successors.

Upon receiving an input from its predecessor, the Splitter copies the input and distributes it across its multiple successors.

---

### Constructor

*Signature:* `Splitter(int32_t branch_count)`

Creates a new branch that splits execution into `branch_count` paths.

**Parameters**

* `branch_count` (`int32_t`): Number of paths to split into. At least 2.

---

### Getters

#### branch_count

*Signature:* `int32_t branch_count() const`

Returns the number of branches after the Splitter's operation, including the Splitter's own branch. Always at least 2.

**Returns**

* `int32_t`: Number of branches.

---

#### to_string

*Signature:* `virtual std::string to_string() const override`

Returns the string "splitter ({branch count of this splitter object})".

**Returns**

* `std::string`: String representation of the splitter object.

---

### Methods

#### forward (multiple outputs)

*Signature:* `virtual std::vector<std::vector<xt::xarray<double>>> forward(const xt::xarray<double>& input, bool tag)`

Returns `input` copied `branch_count()` times.

**Parameters**

* `input` (`const xt::xarray<double>&`): Vector(s) to copy across multiple outputs. Non-empty.
* `tag` (`bool`): Unused; required to distinguish this method from the overridden method that returns `xt::xarray<double>`.

**Returns**

* `std::vector<xt::xarray<double>>`: Vector of length `branch_count()`, where each index contains a copy of `input`.

---

#### forward (single output- DO NOT USE)

*Signature:* `xt::xarray<double> forward(const xt::xarray<double>& unused) override`

DO NOT USE THIS METHOD!

Throws `cast::not_implemented`. The method exists solely to implement a virtual method and should not be used.

---

#### backward

*Signature:* `xt::xarray<double> backward(const xt::xarray<double>& successor_gradient) override`

Returns the empty tensor. Upon receiving the `branch_count()`-th input, returns the result of the splitter's backpropagation operation computed on all received inputs.

**Parameters**

* `successor_gradient` (`xt::xarray<double>`): Single successor gradient. Size and shape of all its elements match those of the first given input received after this method returns a non-empty tensor.

**Returns**

* `xt::xarray<double>`: Empty tensor, or backprop gradients if all inputs are received.


---
---
---

## Combiner

Collapses one or more branches into its own branch.

Combiners have multiple predecessors and exactly one successor.

A Combiner cannot collapse its own branch into itself.

---

### Constructor

*Signature:* `Combiner(std::initializer_list<int32_t> branch_indices)`

Creates a new combiner that pools execution from the branches given at `branch_indices`.

**Parameters**

* `branch_indices` (`std::initializer_list<int32_t>`): 0-based branch indices to combine. Has at least 1 element.

---

### Getters

#### branch_indices

*Signature:* `std::vector<int32_t> branch_indices() const`

Returns a list of branch IDs that this combiner merges, not including the combiner's own branch ID.

**Returns**

* `std::vector<int32_t>`: List of combined branch indices.

---

#### to_string

*Signature:* `std::string to_string() const override`

Returns the string "combiner ({branch indices combined} -> {branch of combiner})".

**Returns**

* `std::string`: String representation of the Combiner.

---

### Methods

#### forward

*Signature:* `xt::xarray<double>forward(const xt::xarray<double>& predecessor_outputs) override`

Returns the empty vector. Upon receiving the `branch_indices().size()`-th input, returns the element-wise sum of all inputs given.

**Parameters**

* `predecessor_outputs` (`const xt::xarray<double>&`): List of layer outputs. Has the same size and matching corresponding shapes as the first input given

**Returns**

* `std::vector<xt::xarray<double>>`: Sum of all inputs, or an empty vector if not all branches are combined.

**Exceptions**

* `std::runtime_error` (via `str_assert`): Thrown if inputs are empty or if shapes do not match.

---

#### backward (multiple outputs)

*Signature:* `virtual std::vector<xt::xarray<double>> backward(const xt::xarray<double>& prev_gradient, bool tag)`

Returns `prev_gradient` copied `branch_indices().size()` times.

**Parameters**

* `prev_gradient` (`const xt::xarray<double>&`): Tensor(s) to copy across multiple outputs. Non-empty.
* `tag` (`bool`): Unused; required to distinguish this method from the overridden method that returns `xt::xarray<double>`.

**Returns**

* `std::vector<xt::xarray<double>>`: Vector of length `branch_indices().size()`, where each index contains a copy of `prev_gradient`.

---

#### backward (single output- DO NOT USE)

*Signature:* `xt::xarray<double> backward(const xt::xarray<double>& unused) override`

DO NOT USE THIS METHOD!

Throws `cast::not_implemented`. The method exists solely to implement a virtual method and should not be used.

---