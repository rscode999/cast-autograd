# Change Log

## 0.11.0
*In Progress*

Will include batch training.

## 0.10.1
*15 September 2026*

- Network assignment creates deep copies of each layer. Previously, the default copy and assignment operations created shallow copies of the layer pointers.
- Add methods to get and set individual network components
- Add cross-entropy loss and softmax activation

- Fix broken links in documentation pages

## 0.10.0
*6 September 2026*

- Decision to use sequential component addition is finalized.
    - *Rationale:* Implementing PyTorch-like method definitions requires two objects: a proxy object that the user creates, and a network-internal object that stores the weights. In the case where an operator is created multiple times, each operator instance, as well as the proxy object, must be stored within the network. Having multiple types of objects adds unnecessary complexity. The framework matching CNet's style of network creation negates the small gains in flexibility.
- Add unit testing through custom CUnit methods. Temporary measure until a more robust testing framework can be installed and used
- Reorganize documentation to make it better suited to be a github wiki
- Add Github workflow to run the unit tests

## 0.9.0
*17 August 2026*

Full implementation of branched network topologies.

- Predecessors and successors within network components now use a `std::unordered_map` instead of a vector.
    - Forward and backward pass computation use a queue-based approach. Each branch ID, index in the components vector, and the component's output is pushed to the queue after computation.
- Splitters and Combiners properly compute their forward and backward passes
- First successful convergence of a branched network
    - XOR classifier converged (17 Aug 2026)
- Renaming of classes:
    - Branch is now Splitter. A Branch is a separate path of execution in a network. 
    - TensorOperator is now NetworkComponent.
- Separate methods for adding Combiners, Splitters, and Operators (single-input, single-output components. Direct subclasses are the ActivationFunction and Layer)
- Proper implementation of the `enable` check, along with verification when each new component is added to the network
- All `std::shared_ptr`s added to the network become deep copies

## 0.7.0
*10 August 2026*

Overhaul of network creation.

- Sequential layer addition. Branches are user-added.
    - *Rationale:* PyTorch-style custom network definition didn't work. Operators go out of scope when the `forward` method is called, so the network cannot access the operators in the backward pass. Having to register operators before configuring execution order is considered too confusing for users.
    - Each layer is stored in a std::vector. Layers track their predecessor and successor index upon addition.
    - Branches are special layers with multiple predecessors or successors.
- Tensor objects (as used in v0.5) are no longer used. The user's data storage is the xt::xarray.

New architecture converged to XOR dataset.

## 0.5.0
*6 August 2026*

First functional release

- Implemented functionality from old CNet
    - Sequentially defined Network class
    - Linear 1d layer, Sigmoid activation function
    - SGD optimizer
    - Mean Squared Error loss
- Completed first successful training run
    - Loss converged on XOR dataset (6 Aug 2026)