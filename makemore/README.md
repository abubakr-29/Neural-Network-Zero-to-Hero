# Makemore

This is my practice implementation while learning character-level language
models from Andrej Karpathy's **Neural Networks: Zero to Hero** material.

The goal is to understand how a model can learn patterns from names and
generate new names one character at a time. This is an educational exercise,
not the official makemore project.

For simple, detailed explanations of both notebooks, see the
[makemore study notes](NOTES.md).

## What is included?

The project currently has two notebooks that show the model growing in stages:

- [`makemore_1.ipynb`](makemore_1.ipynb): a character bigram model.
- [`makemore_2.ipynb`](makemore_2.ipynb): a small multilayer perceptron
  language model with embeddings and a 3-character context.

Together, the notebooks cover:

- Loading and inspecting a dataset of 32,033 names.
- Adding a special start/end token to each name.
- Counting character pairs, also called bigrams.
- Building a `27 x 27` character transition table.
- Converting counts into probabilities.
- Sampling new names from a probability table.
- Measuring a model with negative log-likelihood.
- Splitting data into training, development, and test sets.
- Representing characters with learned embeddings.
- Predicting the next character from a context of three characters.
- Training an MLP with minibatches, backpropagation, and gradient descent.
- Evaluating training and development losses.

The other important files are:

- [`names.txt`](names.txt): one training name per line.
- [`NOTES.md`](NOTES.md): detailed explanations and learning notes.

## Quick start

From the repository root, activate the virtual environment and install the
dependencies:

```bash
source .venv/bin/activate
pip install jupyter torch matplotlib
```

Open either notebook in VS Code, or run the following from this folder:

```bash
cd makemore
jupyter notebook
```

Run each notebook from top to bottom. The cells depend on variables created
earlier, so **Restart Kernel and Run All** is the easiest way to start fresh.

## Recommended order

Start with [`makemore_1.ipynb`](makemore_1.ipynb). It introduces the basic
ideas using character counts and a simple bigram neural network.

Then continue with [`makemore_2.ipynb`](makemore_2.ipynb). It rewrites the
dataset preparation more clearly and builds a neural network that uses the
previous three characters to predict the next one.

## Main idea

The project follows this learning cycle:

1. Add a start/end marker to each name.
2. Prepare examples of context characters and next-character targets.
3. Convert characters into integer indexes and learned embeddings.
4. Use a neural network to produce logits for the next character.
5. Calculate cross-entropy loss.
6. Update the parameters with backpropagation and gradient descent.
7. Generate names by repeatedly sampling the next character.

The first notebook looks only at the current character. The second notebook
uses a context of three characters, which gives the model more information
when making a prediction.

## Current status

The bigram model and the first MLP language model are implemented. The second
notebook now includes an 80% training split, a 10% development split, and a
10% test split, along with learned character embeddings and minibatch
training.

The next steps are to improve the model and continue through the later
makemore lessons, including better initialization, normalization, deeper
networks, and more systematic evaluation.

## Attribution

This project is based on what I am learning from Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course
and [makemore](https://github.com/karpathy/makemore).

The notebooks contain my own practice code and notes. They are not an official
implementation or a replacement for the original project.
