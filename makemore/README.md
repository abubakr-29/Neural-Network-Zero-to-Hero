# Makemore

This is my practice implementation while learning character-level language
models from Andrej Karpathy's **Neural Networks: Zero to Hero** material.

The goal is to understand how a model can learn patterns from names and
generate new names one character at a time. This is an educational exercise,
not the official makemore project.

For simple, detailed explanations of the notebook, see the
[makemore study notes](NOTES.md).

## What is included?

The notebook currently covers:

- Loading and inspecting a dataset of 32,033 names.
- Adding a special start/end token to each name.
- Counting character pairs, also called bigrams.
- Building a `27 x 27` character transition table.
- Converting counts into probabilities.
- Sampling new names from the probability table.
- Measuring the model with negative log-likelihood.
- Rebuilding the bigram model as a small neural network.
- Training the model with backpropagation and gradient descent.
- Sampling new names from the trained neural network.

The main files are:

- [`makemore.ipynb`](makemore.ipynb): the experiments and code.
- [`names.txt`](names.txt): one training name per line.
- [`NOTES.md`](NOTES.md): detailed explanations and learning notes.

## Quick start

From the repository root, activate the virtual environment and install the
dependencies:

```bash
source .venv/bin/activate
pip install jupyter torch matplotlib
```

Open [`makemore.ipynb`](makemore.ipynb) in VS Code, or run the following from
this folder:

```bash
cd makemore
jupyter notebook makemore.ipynb
```

Run the notebook cells from top to bottom. The cells depend on variables
created earlier, so **Restart Kernel and Run All** is the easiest way to start
fresh.

## Main idea

The project follows this learning cycle:

1. Add a start/end marker to each name.
2. Count which characters commonly follow other characters.
3. Turn the counts into probabilities.
4. Generate a name by sampling one character at a time.
5. Measure the model with negative log-likelihood.
6. Learn the same probabilities using a neural network and gradient descent.

The first model is a bigram model, so it only looks at the current character
when predicting the next one. For example, after seeing `e`, it learns how
likely the next character is to be `m`, `l`, `r`, or another character.

## Current status

The count-based bigram model and its neural-network version are implemented.
The next steps are to add train/validation/test splits, build a multilayer
perceptron language model, and explore batching and improved optimization.

## Attribution

This project is based on what I am learning from Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course
and [makemore](https://github.com/karpathy/makemore).

The notebook contains my own practice code and notes. It is not an official
implementation or a replacement for the original project.
