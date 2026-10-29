# Makemore

This is my practice implementation while learning character-level language
models from Andrej Karpathy's **Neural Networks: Zero to Hero** material.

The goal is to understand how a model learns patterns from names and generates
new names one character at a time. This is an educational exercise, not the
official makemore project.

For detailed explanations of the ideas and code, see the
[makemore study notes](NOTES.md).

## What is included?

The project is organized into five main notebooks. Each part builds on the
previous one:

1. [`makemore_1.ipynb`](makemore_1.ipynb) — character bigrams and a simple
   neural-network version.
2. [`makemore_2.ipynb`](makemore_2.ipynb) — character embeddings and an MLP
   that uses a three-character context.
3. [`makemore_3.ipynb`](makemore_3.ipynb) — manually implemented layers,
   gradient checking, and batch normalization.
4. [`makemore_4.ipynb`](makemore_4.ipynb) — improved batch normalization,
   initialization, and evaluation.
5. [`makemore_5.ipynb`](makemore_5.ipynb) — reusable layers and a deeper
   hierarchical model with a longer context.

The notebooks cover:

- Loading and inspecting a dataset of 32,033 names.
- Adding a special start/end token to each name.
- Counting character pairs and turning counts into probabilities.
- Measuring models with negative log-likelihood and cross-entropy.
- Preparing training, development, and test splits.
- Learning character embeddings.
- Building an MLP language model from basic tensor operations.
- Implementing reusable `Linear`, `Embedding`, `Tanh`, and
  `BatchNorm1d` layers.
- Checking gradients against numerical approximations.
- Training with minibatches and learning-rate decay.
- Building a deeper hierarchical model that processes context in groups.
- Sampling new names from the trained models.

The other important files are:

- [`names.txt`](names.txt): one training name per line.
- [`NOTES.md`](NOTES.md): detailed explanations for all five parts.

## Quick start

From the repository root, activate the virtual environment and install the
dependencies:

```bash
source .venv/bin/activate
pip install jupyter torch matplotlib
```

Open a notebook in VS Code, or run the following from this folder:

```bash
cd makemore
jupyter notebook
```

Run each notebook from top to bottom. The cells depend on variables created
earlier, so **Restart Kernel and Run All** is the easiest way to start fresh.

## Recommended order

Start with `makemore_1.ipynb` and continue through `makemore_5.ipynb` in
order. The progression is:

```text
counts and bigrams
        ↓
embeddings and a basic MLP
        ↓
manual layers and gradient checking
        ↓
better batch normalization and initialization
        ↓
reusable layers and a deeper hierarchical network
```

Each notebook is a learning milestone. It is useful to run the earlier
notebooks first, even when the later notebook contains a more complete model.

## Main idea

At every stage, the model learns to predict the next character:

1. Add a start/end marker to each name.
2. Turn text into integer character indexes.
3. Build context/target examples.
4. Produce logits for the next character.
5. Calculate a loss.
6. Use backpropagation to calculate gradients.
7. Update the parameters with gradient descent.
8. Sample characters repeatedly to generate a name.

The first part only uses the current character. Later parts use longer
contexts, learned embeddings, hidden layers, normalization, and a deeper
architecture to make better predictions.

## Current status

All five learning parts are present:

- Part 1 implements the count-based bigram model.
- Part 2 implements an embedding-based MLP.
- Part 3 introduces reusable ideas such as manual layers, gradient checking,
  and batch normalization.
- Part 4 improves the training setup and batch-normalization handling.
- Part 5 builds a deeper hierarchical model with reusable layer classes.

The notebooks are educational experiments rather than production code. The
next improvements can include more careful hyperparameter experiments,
consistent test-set reporting, and further convolution-style language-model
work.

## Attribution

This project is based on what I am learning from Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course
and [makemore](https://github.com/karpathy/makemore).

The notebooks contain my own practice code and notes. They are not an official
implementation or a replacement for the original project.
