(Forward pass completed)

NEXT MAJOR STEP: Compute a backwards pass.
- Make the operator-wise backwards pass store weights and biases, then make them accessible to the optimizer
- Have the optimizer pass throught the network

Then, expose the Tensor object as the primary data storage, as opposed to the xt::xarray.