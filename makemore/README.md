# Makemore

This folder contains my notes and experiments while learning how to build a
small character-level language model. The project follows the makemore lessons
from Andrej Karpathy's **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**
playlist.

The basic idea is simple: give the model a list of names, let it learn which
characters commonly follow other characters, and then ask it to create new
names one character at a time. The generated names are not copied from the
dataset; they are new samples from the patterns the model learned.

## Current status

The first notebook is in progress and currently includes a complete **bigram
model**:

- It loads and inspects the name dataset.
- It counts how often one character follows another.
- It samples names directly from those counts.
- It trains the same model as a tiny neural network using gradient descent.
- It samples names from the trained neural network.

The later multilayer language-model lessons have not been added yet.

## Files in this folder

```text
makemore/
|-- README.md          # This guide
|-- NOTES.md           # Space for short written notes as the project grows
|-- names.txt          # Training data: one name per line
`-- makemore.ipynb     # Notebook containing the experiments and explanations
```

### `names.txt`

This is the dataset used by the notebook. Each line contains one name. The
notebook currently reads **32,033 names** from this file.

### `makemore.ipynb`

This is the main working file. It is intentionally written as a learning
notebook, so it contains intermediate values, visualizations, printed output,
and small experiments instead of being packaged as a reusable library.

### `NOTES.md`

This file is kept for extra explanations, questions, and reminders that do not
fit naturally beside the code in the notebook.

## What the notebook does

### 1. Load and inspect the data

The notebook reads the names with:

```python
words = open("names.txt", "r").read().splitlines()
```

It then checks a few basic facts, such as the number of names and the shortest
and longest name. These checks are useful because they make the dataset
familiar before any model is built.

### 2. Add a start/end token

The model needs to know where a name begins and ends. The notebook uses `.`
as a special token:

```text
. emma .
```

The dot is not part of a real name. It means “start” when it appears first and
“stop” when it appears last. Because the alphabet has 26 letters plus `.`, the
model has **27 possible characters**.

### 3. Count bigrams

A **bigram** is a pair of neighboring characters. For example, the name
`emma` produces these pairs:

```text
. -> e
e  -> m
m  -> m
m  -> a
a  -> .
```

The notebook counts every such pair across all names. These counts form a
`27 x 27` table called `N`. A row represents the current character, and a
column represents the next character.

### 4. Turn counts into probabilities

Counts are converted into probabilities by normalizing each row. For example,
the row for `e` tells us the probability of the next character being `a`, `m`,
`r`, and so on.

The notebook adds `1` to every count before normalizing. This is simple
smoothing: it prevents a probability from becoming exactly zero, which makes
sampling and log-loss calculations safer.

### 5. Generate names by sampling

Generation starts at the `.` token. The model:

1. Looks at the probability distribution for the current character.
2. Randomly chooses the next character.
3. Uses that character as the new current character.
4. Stops when it chooses `.` again.

This process is repeated to create several sample names. A fixed random seed
is used in the notebook where reproducible output is helpful.

### 6. Measure the model with negative log-likelihood

The notebook evaluates how much probability the model gives to the real
character transitions in the dataset. The main concepts are:

- **Likelihood:** how probable the observed data is under the model.
- **Log-likelihood:** a more convenient version of likelihood for many
  multiplied probabilities.
- **Negative log-likelihood (NLL):** the value we minimize during training.
- **Average NLL:** the average loss across all character transitions.

A better model assigns higher probability to the correct next character, which
means a lower NLL.

### 7. Rebuild the model as a neural network

The notebook then creates a weight matrix `W` with shape `27 x 27`. Each row
corresponds to a current character and each column corresponds to a possible
next character.

The same process is expressed with neural-network operations:

1. Convert each input character into a one-hot vector.
2. Multiply the vector by `W` to produce logits.
3. Convert logits into probabilities with softmax-equivalent operations.
4. Calculate the average negative log-likelihood.
5. Compute gradients with `loss.backward()`.
6. Update `W` with gradient descent.

This is a small but important step: the count-based table and the learned
neural-network weights are two ways of building essentially the same bigram
model.

## How to run it

### 1. Create the environment

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install the notebook dependencies

```bash
pip install jupyter torch matplotlib
```

If the virtual environment already exists, activate it before opening the
notebook:

```bash
source .venv/bin/activate
```

### 3. Open the notebook from this folder

Run this command from the `makemore` directory so that the relative path to
`names.txt` works:

```bash
cd makemore
jupyter notebook makemore.ipynb
```

You can also open `makemore.ipynb` directly in VS Code with the Jupyter
extension.

### 4. Run the cells

For a fresh run, use **Restart Kernel and Run All**. The cells build on values
created earlier in the notebook, so running them from top to bottom avoids
undefined-variable errors.

## Useful things to remember

- The notebook expects `names.txt` to be in the current working directory.
- The model predicts one character at a time; it does not understand the
  meaning of a name.
- Sampling is random, so generated names can change between runs.
- Setting the random seed makes the sampling output repeatable.
- `W.data` is used in the current learning experiment to update the weights
  directly. This is fine for following the lesson, but future versions may
  use a more structured optimizer.
- The notebook is educational code. It is not intended to be a production
  text-generation system.

## Learning roadmap

- [x] Load and inspect the names dataset.
- [x] Build a character bigram count table.
- [x] Generate names from normalized counts.
- [x] Calculate negative log-likelihood.
- [x] Train the bigram model with a small neural network.
- [ ] Split data into training, validation, and test sets.
- [ ] Build a multilayer perceptron language model.
- [ ] Study embeddings, batching, and better optimization.
- [ ] Explore deeper models and batch normalization.
- [ ] Continue toward the later makemore and GPT-style lessons.

## Attribution

This project is based on ideas from Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course
and [makemore](https://github.com/karpathy/makemore) project.

The code, notes, and experiments in this repository are my own learning work.
They are not an official implementation of Karpathy's projects.
