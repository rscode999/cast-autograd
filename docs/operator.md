# Operator

[Back to central documentation](Home.md)

Single-input, single-output network components. An Operator is any component that does not create or destroy network branches.

Operators have exactly one predecessor and exactly one successor.

Subclass of `NetworkComponent`.

Subclasses
* [ActivationFunction](activation_function.md)
* [Layer](layer.md)