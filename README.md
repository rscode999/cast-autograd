# CAST (Compartmentalized Autograd System with Tensors)
*CNet 2.0*  
**C++ framework for neural networks**


Design Goals
- **FIRST AND FOREMOST:** Allow users to create networks of any architecture
- Make all training and validation operations available using only XTensor's `xt::xarray`, with no need to wrap it in another class
- Decentralize and compartmentalize data as much as possible
    - Keep as much data out of the network as possible. Instead, store data in network components, such as tensors and operators
- Use architecture where the network is the central object. All operations can be done through a network.
    - Ideally, CNet's 3 main operations (forward, predict, backward/reverse) are called as network methods
    - So far, this goal seems possible only if layers are added sequentially

NOTE: If you want to use clangd, ensure VS Build Tools (with C++ optional functionality) is installed