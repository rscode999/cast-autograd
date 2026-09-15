# CAST (Compartmentalized Autograd System with Tensors)
*CNet 2.0*  
**C++ framework for neural networks**

[Documentation](docs/Home.md)

[Change Log](CHANGES.md)

## Foreword

I have worked on the CNet project throughout the 2025-26 school year. When I started work on CNet in August 2025, I had no formal education in deep learning, so the CNet project became my de facto deep learning class. I learned about the strengths and weaknesses of different activation functions, how an optimizer works, and the intricacies of the C++ language. I take great pride in the final product, knowing that I built a solid framework and mastered the extremely fine details of deep learning.

But I needed *more power...*

The seeds of a CNet successor arrived at the end of the 2025-26 school year.  
To aid in future image-recognition projects, I wanted to add convolutional layers and the Adam optimizer.  
I soon learned that CNet was not as expansion-friendly as I thought. The forward pass logic supported only 1D linear layers without branching structures. Worse, the existing SGD optimizer code turned out to be bloated and unmaintainable. I had also learned more about how deep learning frameworks are implemented, explaining the unusual constructs of the PyTorch framework, especially why the optimization pass is not called through the network. Worse, the Eigen package that CNet uses did not support tensors of arbitrary dimension.

I realized that the best way to accomplish my goals and expand CNet is to rewrite the framework from the ground up.

In the summer of 2026, I started work on a CNet successor with these design goals:

- Allow users to create networks of any architecture
- Make all training and validation operations available using only XTensor's `xt::xarray`, with no need to wrap it in another class. Operations and underlying logic become much simpler.
- Decentralize logic and data as much as possible. Avoid a situation where a single class or method takes the majority of the logic, becoming bloated.
    - Reduce the number of fields in the network. Put data and functionality (i.e. gradients of layer weights) within the network fields.
- Use architecture where the network is the central object. All operations can be done through a network.
    - Ideally, CNet's 3 main operations (forward, predict, backward/reverse) are called as network methods

Guided by its design goals, the CAST framework took shape over the coming months.

## Setup Instructions

You will need a compiler that supports C++20 or later.

### XTensor Installation

This project requires the XTensor package and some of its supporting libraries.

Open a terminal in the "lib" folder. Clone each repo, with the required version tags only:
```
git clone --branch 0.27.0 https://github.com/xtensor-stack/xtensor
git clone --branch 0.23.0 https://github.com/xtensor-stack/xtensor-blas
git clone --branch 0.8.2 https://github.com/xtensor-stack/xtl
```

You should have 3 directories inside the "lib" folder: xtensor, xtensor-blas, and xtl. Each should contain the "include" subdirectory, which contains the library source files.

If any of the installations didn't work, go to the repo's tags. The Tags button should be near the Branches dropdown. Ensure that the version tag exists. If not, contact the CAST development team.

### Other Setup

Due to the large size of the XTensor libraries, the C/C++ VS Code extension is not recommended. A better alternative is ClangD, which is much faster and gives better warnings.  
Configuration files come with this repo.

NOTE: If you want to use ClangD, ensure [MSVC Build Tools](https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc?view=msvc-170) is installed.  
Ensure the options for "Desktop development with C/C++" and "MSVC Build Tools for x64/x86" are selected during installation.

## Quick Start Guide

All objects are under the `cast` namespace. 

Import CAST functionality using the files in the "include" directory:
```
#include "include/cast.hpp"
```
Or, point your compiler to use "include/cast.hpp" as a standard library header.

You may want to use the `cast` namespace:
```
using namespace cast;
```

<br>

To create a  network:
```
Network net;
```
The network begins empty. Components must be added to the network.

<br>

To add layers and activation functions to the network:
```
//Add a fully-connected 1d linear layer, with 2 inputs and 4 outputs
net.add_operator(std::shared_ptr<Linear1d>(2, 4));

//Add a Sigmoid activation function
net.add_operator(std::shared_ptr<Sigmoid>());
```

<br>

To add separate paths of execution in the network:
```
//Adds a splitter object, breaking execution into 3 paths
net.add_splitter(3);
```
Each branch (separate path of execution) has a unique branch ID. The ID of the default branch is 0.  
The splitter added above created two new branches, with IDs 1 and 2.

To add to new branches:
```
//Adds Linear1d layer to branch 0
net.add_operator(std::shared_ptr<Linear1d>(4, 4), 0);

//Adds Linear1d layer to branch 1
net.add_operator(std::shared_ptr<Linear1d>(4, 4), 1);

//Adds Linear1d layer to branch 2
net.add_operator(std::shared_ptr<Linear1d>(4, 4), 2);
```
A network cannot be trained unless it has exactly one unterminated branch.

To merge branches:
```
//Adds a Combiner object to branch 0, set to merge branches 1 and 2 into branch 0
net.add_combiner({1, 2}, 0);
```
Once merged, branch IDs are not reused. Even if more branches are created, branches 1 and 2 cannot be added to.

See the [network branch creation section](docs/Home.md#network-branch-management) for more details on branches.

<br>

To add a loss calculator and optimizer:
```
//Sets to use Mean Squared Error loss
net.set_loss_calculator(std::make_shared<MeanSquaredError>());

//Sets optimizer to SGD, with learning rate 0.02 and momentum coefficient 0.9
std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(0.02, 0.9);
net.set_optimizer(optimizer);
```

<br>

To check network configuration, export the network directly to an output stream:
```
std::cout << net << std::endl;
```

<br>

To enable the network, allowing for training and optimization:
```
net.enable();
```
In order to enable, the network must have:
- A loss calculator
- An optimizer
- At least one component
- Exactly one output: that is, there is exactly one unmerged branch

If any of these traits are not met, the `enable` method throws the `cast::enable_failed_error` exception.

<br>

To compute the forward pass on an input:
```
xt::xarray<double> input = {0, 1};

xt::xarray<double> forward_output = net.forward(input);
```

If layer dimensions are incompatible, the `forward` method throws the `cast::shape_error` exception.

<br>

To compute the backwards pass, computing gradients for each layer:
```
//Pass the predicted value from the forward pass, along with the expected forward-pass output
net.backward(forward_output, {1, 1});
```

<br>

To optimize the network, using the network's optimizer and the gradients stored from the `backward` call:
```
net.optimize();
```
If you have a reference to the optimizer used by the network, you may also optimize the network through the Optimizer object:
```
optimizer->step();
```


## Compilation Instructions

Compile for the **C++20 standard**, treating the "xtensor/include", "xtensor-blas/include", and "xtl/include" directories inside the lib/ folder as built-in libraries.

The Makefile has the following command for the G++ compiler:
```
g++ 'main.cpp'   -o 'main'  --std=c++20  -I 'lib/xtensor/include/' -I 'lib/xtl/include/' -I 'lib/xtensor-blas/include'
```

From the project's root directory, compilation can also be executed as:
```
make c
```
if you have GnuMake installed.

<br>

To build, you will need to install [CMake](https://cmake.org/download/) (at least version 3.16), as well as build tools for your platform. Windows users require [MSVC Build Tools](https://aka.ms/vs/stable/vs_BuildTools.exe) with the "Desktop development with C/C++" and MSVC Build Tools for x64/x86" option.

Move to the project root directory. Rename your main file to "main.cpp" (or use the provided main file), move it to the project root directory, and run the command (if you have GnuMake installed):
```
make build
```

The build files will be placed in a directory called "build", inside the project root directory.


## File Layout
- "build" (which does not appear until the `make build` command is run) contains the project build files
- "docs" contains extra information about CAST methods, long-term records, and the change log
- "include" contains all CAST source files
- "lib" should contain the 3 XTensor libraries, upon proper installation
- ".clang-format", ".clangd", and "compile-flags.txt" are configuration files for the ClangD extension, should you choose to use it