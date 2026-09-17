# Micrograd From Scratch

This is my learning project for understanding neural networks and automatic differentiation from the ground up.

I am building a tiny version of **micrograd** in Python while learning how neural networks calculate gradients and improve their parameters. The goal is not to create a production-ready deep learning library. The goal is to make each step visible and understandable.

The main work is currently in [`micrograd.ipynb`](micrograd.ipynb).

## What I am learning

This project follows the basic ideas behind training a neural network:

1. Start with numbers and mathematical operations.
2. Build a computation graph from those operations.
3. Calculate how much each value affects the final result.
4. Use those gradients to update values and reduce the error.
5. Combine the same ideas to represent a small neuron.

The important idea is that a neural network is mostly a collection of mathematical operations with parameters. Once we can calculate gradients for those operations, we can use gradient descent to learn useful parameter values.

## What is micrograd?

Micrograd is a very small automatic differentiation engine. It works with scalar values instead of large tensors, which makes it easier to see what is happening internally.

In this project, the `Value` class represents one scalar number. Each `Value` can remember:

- Its numerical value, stored in `data`.
- Its gradient, stored in `grad`.
- The values that were used to create it.
- The operation that created it.
- A small function that knows how to send gradients backward through that operation.

For example:

```python
a = Value(2.0)
b = Value(-3.0)
c = a * b
```

The value `c` remembers that it came from multiplying `a` and `b`. This memory is what allows the project to build a computation graph and later calculate gradients.

## Current features

The notebook currently demonstrates:

- Basic Python functions and plotting.
- Numerical differentiation using a small value of `h`.
- A computation graph made from scalar values.
- A custom `Value` class.
- Addition and multiplication operations.
- The hyperbolic tangent activation function, `tanh`.
- Parent tracking through `_prev`.
- Operation tracking through `_op`.
- Reverse-mode automatic differentiation.
- Topological ordering of graph nodes.
- Graph visualization with Graphviz.
- Manual parameter updates using gradients.
- A small neuron calculation using inputs, weights, a bias, and `tanh`.
- Examples that show why gradients must be accumulated when a value is used more than once.

## Project structure

```text
micrograd/
|-- micrograd.ipynb   # Experiments, explanations, code, and visualizations
`-- README.md         # Project documentation
```

## Getting started

### Requirements

You need:

- Python 3
- Jupyter Notebook or VS Code with the Jupyter extension
- NumPy
- Matplotlib
- Graphviz Python package
- Graphviz system software for rendering the graphs

### Install the Python packages

From this directory, run:

```bash
pip install numpy matplotlib graphviz jupyter
```

Depending on your operating system, you may also need to install the Graphviz application itself.

On Ubuntu or Debian:

```bash
sudo apt install graphviz
```

On macOS with Homebrew:

```bash
brew install graphviz
```

On Windows, install Graphviz from its official installer and make sure it is available on your system `PATH`.

### Open the notebook

Start Jupyter with:

```bash
jupyter notebook
```

Then open `micrograd.ipynb`.

You can also open the notebook directly in VS Code if the Python and Jupyter extensions are installed.

## Notebook walkthrough

### 1. Numerical differentiation

The notebook starts with a simple function:

```python
def f(x):
    return 3 * x**2 - 4 * x + 5
```

It plots the function and estimates a derivative by comparing two nearby points:

```python
(f(x + h) - f(x)) / h
```

This is called a finite-difference approximation. It is useful for checking whether a manually calculated gradient is correct, although it is not the method normally used to train neural networks.

### 2. A small computation graph

The notebook then creates a few values and combines them:

```python
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
```

The same idea is later represented with `Value` objects. Each result keeps track of its inputs, creating a graph such as:

```text
a ----┐
      multiply ----┐
b ----┘            add ---- d
c -----------------┘
```

This graph lets us work backward from a final output and determine how much each input contributed to it.

### 3. The `Value` class

The `Value` class is the heart of the project. It currently supports:

```python
c = a + b
c = a * b
c = a.tanh()
```

When an operation creates a new `Value`, it also stores a `_backward` function. That function contains the local derivative for the operation.

For addition:

$$
\frac{\partial (a + b)}{\partial a} = 1,
\qquad
\frac{\partial (a + b)}{\partial b} = 1
$$

For multiplication:

$$
\frac{\partial (a b)}{\partial a} = b,
\qquad
\frac{\partial (a b)}{\partial b} = a
$$

For the hyperbolic tangent:

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
$$

The backward functions use the chain rule to combine these local derivatives with the gradient arriving from the output.

### 4. Reverse-mode automatic differentiation

Calling:

```python
L.backward()
```

starts gradient calculation from the final value `L`.

The method does three important things:

1. Visits every node that leads to `L`.
2. Stores the nodes in topological order, so each operation can be processed after its output gradient is known.
3. Walks through the nodes in reverse order and calls each node's `_backward` function.

The final output starts with a gradient of `1.0` because:

$$
\frac{\partial L}{\partial L} = 1
$$

After backward propagation, every value contains its gradient in the `grad` attribute.

### 5. Visualizing the graph

The notebook uses Graphviz to draw the computation graph. Each value shows:

- Its label.
- Its data value.
- Its gradient.

Operation nodes such as `+`, `*`, and `tanh` are shown between their input and output values. This makes the forward computation and backward gradient flow easier to inspect.

The helper functions are:

```python
trace(root)
draw_dot(root)
```

For example:

```python
draw_dot(L)
```

### 6. A simple neuron

The notebook builds a small neuron from two inputs:

```python
x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735878195432)
```

The neuron calculates a weighted sum and applies `tanh`:

$$
 n = x_1w_1 + x_2w_2 + b
$$

$$
 o = \tanh(n)
$$

Here:

- `x1` and `x2` are inputs.
- `w1` and `w2` are weights.
- `b` is the bias.
- `n` is the pre-activation value.
- `o` is the neuron's output.

Calling `o.backward()` calculates how the output changes with respect to every input, weight, and the bias.

## A small gradient-descent update

Once gradients have been calculated, parameters can be changed in the direction that improves the result. The notebook demonstrates a simple update like this:

```python
a.data += 0.01 * a.grad
```

The exact sign and update rule depend on whether the value represents a quantity we want to maximize or minimize. In a typical loss-minimization training loop, parameters are usually updated in the opposite direction of the gradient:

```python
parameter.data -= learning_rate * parameter.grad
```

This notebook is currently focused on understanding the gradient mechanism. A complete training loop is a natural next step.

## Important implementation notes

This is a learning implementation, so it intentionally keeps the API small and readable.

At the moment:

- Values are scalar numbers rather than arrays or tensors.
- Addition and multiplication expect `Value` objects on both sides.
- The available activation function is `tanh`.
- There is no complete `Neuron`, `Layer`, or `MLP` class yet.
- There is no dataset loader or full training loop yet.
- Gradients are accumulated with `+=`, which is important when a node is used multiple times.
- Gradients should be reset before a new backward pass when repeatedly training the same parameters.

These limitations are useful for learning because they keep the computation graph visible.

## Suggested next steps

A possible learning path from here is:

1. Add support for operations with Python numbers, such as `Value(2.0) + 3`.
2. Add subtraction, negation, division, and power operations.
3. Add gradient-reset functionality, such as `zero_grad()`.
4. Move the `Value` class into a Python module such as `micrograd/engine.py`.
5. Create `Neuron`, `Layer`, and `MLP` classes.
6. Build a small dataset for binary classification.
7. Add a loss function.
8. Write a full training loop using gradient descent.
9. Compare the results with a library such as PyTorch.
10. Add tests that compare analytical gradients with finite-difference estimates.

## Learning resources

This project is inspired by Andrej Karpathy's educational walkthrough, **The spelled-out intro to neural networks and backpropagation: building micrograd**. It is a great companion for understanding how the ideas in this notebook fit together.

Useful topics to study alongside this project:

- Derivatives and the chain rule.
- Computation graphs.
- Reverse-mode automatic differentiation.
- Backpropagation.
- Activation functions.
- Loss functions.
- Gradient descent.
- Neural network layers and parameters.

## Why I am building this

Deep learning libraries make it easy to train models, but they can hide the mechanics behind many abstractions. Building a tiny engine from scratch makes those mechanics easier to see:

- A forward pass creates values.
- The computation graph records how those values were created.
- A backward pass applies the chain rule.
- Gradients tell us how to change parameters.
- Repeating this process allows a model to learn.

This repository is a record of that learning process, so unfinished experiments and manual calculations are part of the project.

## Status

This project is actively being built while I learn. The current notebook contains the foundations of scalar automatic differentiation and a first small neuron example.
