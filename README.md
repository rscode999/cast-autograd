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

## Installation of XTensor
Installation links: [XTensor](https://github.com/xtensor-stack/xtensor); [XTensor BLAS](https://github.com/xtensor-stack/xtensor-blas); [XTL](https://github.com/xtensor-stack/xtl/tree/master). Requires C++20 or later.

1. Go to each github repo, given by the links

2. On each github repo, visit the "Tags" page. Ensure that the following required version tags exist:  
XTensor: 0.27.0  
XTensor BLAS: 0.23.0  
XTL: 0.8.2

3. Clone each repo, with the required version tags only, into the "lib" folder:
```
git clone --branch <VERSION TAG> <REPO LINK>
```

4. Check installation by compiling the main program:
```
cd ..
make
```