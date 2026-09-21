# Micrograd

This is my practice implementation while learning the foundations of neural networks from Andrej Karpathy's **Neural Networks: Zero to Hero** material.

The goal is to understand automatic differentiation and backpropagation by building the pieces myself in a notebook. This is an educational exercise, not the official micrograd project.

For simple, detailed explanations of the notebook, see the [micrograd study notes](NOTES.md).

## What is included?

The notebook currently covers:

- Numerical differentiation and computation graphs.
- A scalar `Value` class with `data` and `grad`.
- Addition, multiplication, and `tanh` operations.
- Reverse-mode automatic differentiation with `backward()`.
- Computation graph visualization with Graphviz.
- `Neuron`, `Layer`, and `MLP` classes.
- A small dataset, loss function, and gradient-descent training loop.
- A comparison with PyTorch tensors and gradients.

## Quick start

From the repository root, activate the virtual environment and install the dependencies:

```bash
source .venv/bin/activate
pip install numpy matplotlib graphviz jupyter torch
```

Graphviz also needs to be installed on the system. On Ubuntu or Debian:

```bash
sudo apt install graphviz
```

Open [`micrograd.ipynb`](micrograd.ipynb) in VS Code, or run:

```bash
jupyter notebook
```

## Main idea

The project follows this training cycle:

1. Build values through mathematical operations.
2. Record those operations in a computation graph.
3. Propagate gradients backward using the chain rule.
4. Update the parameters to reduce the loss.

The notebook builds a multilayer perceptron like this:

```python
n = MLP(3, [4, 4, 1])
```

## Current status

The scalar autodiff engine and a small trainable MLP are implemented. The next steps are to move the implementation into Python modules, add more operations, and write tests that compare analytical gradients with numerical gradients.

## Attribution

This project is based on what I am learning from Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course and [micrograd](https://github.com/karpathy/micrograd).

The notebook contains my own practice code and notes. It is not an official implementation or a replacement for the original project.
