# Micrograd From Scratch

I am learning neural networks by building a tiny automatic differentiation engine from scratch in Python.

The main project is [`micrograd.ipynb`](micrograd.ipynb). It is an educational notebook, not a production deep learning library. The goal is to understand what happens during a forward pass, backpropagation, and gradient descent.

## What is included?

The notebook currently covers:

- Numerical differentiation and computation graphs.
- A scalar `Value` class with `data` and `grad`.
- Addition, multiplication, and `tanh` operations.
- Reverse-mode automatic differentiation with `backward()`.
- Computation graph visualization with Graphviz.
- `Neuron`, `Layer`, and `MLP` classes.
- A small dataset, mean-squared-error-style loss, and gradient-descent training.
- A comparison with PyTorch tensors and gradients.

The core idea is simple:

1. Build values through mathematical operations.
2. Store the computation graph.
3. Propagate gradients backward using the chain rule.
4. Update parameters to reduce the loss.

## Quick start

Install the Python dependencies:

```bash
pip install numpy matplotlib graphviz jupyter torch
```

Graphviz also needs to be installed on your system. On Ubuntu or Debian:

```bash
sudo apt install graphviz
```

Open the notebook with Jupyter:

```bash
jupyter notebook
```

Then open `micrograd.ipynb`. It also works in VS Code with the Python and Jupyter extensions.

## Example

The notebook builds a multilayer perceptron like this:

```python
n = MLP(3, [4, 4, 1])
```

Then it trains the model by repeatedly:

```python
ypred = [n(x) for x in xs]
loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

for p in n.parameters():
    p.grad = 0.0

loss.backward()

for p in n.parameters():
    p.data += -0.01 * p.grad
```

This makes the relationship between neural networks, gradients, and parameter updates easier to see.

## Project structure

```text
Neural Network Zero to Hero/
|-- README.md
`-- micrograd/
    `-- micrograd.ipynb
```

## Learning goals

I am using this project to understand:

- Derivatives and the chain rule.
- Backpropagation and automatic differentiation.
- Activation functions and loss functions.
- Neurons, layers, and multilayer perceptrons.
- Gradient descent and model training.

## Acknowledgement

This is an independent learning exercise based on Andrej Karpathy's educational walkthrough, [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0).

The ideas and learning material come from Karpathy's work. This notebook is my own practice implementation and notes while following along. It is not the original micrograd project, an official implementation, or an attempt to claim ownership of Karpathy's work. Please see the original tutorial and [micrograd repository](https://github.com/karpathy/micrograd) for the source material.

## Next steps

- Move the implementation from the notebook into Python modules.
- Add more operations and activation functions.
- Add tests for the gradients.
- Improve the training examples and experiment with different network sizes.

This project is still in progress as I learn.
