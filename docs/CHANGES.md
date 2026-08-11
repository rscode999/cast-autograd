# Change Log

## 0.7.0
*10 August 2026*

Overhaul of network creation.
- Sequential layer addition. Branches are user-added.
    - Rationale: PyTorch-style custom network definition didn't work. Operators go out of scope when the `forward` method is called, so the network cannot access the operators in the backward pass
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