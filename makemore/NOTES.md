# Makemore Study Notes

These notes explain the five main makemore notebooks:

1. [`makemore_1.ipynb`](makemore_1.ipynb)
2. [`makemore_2.ipynb`](makemore_2.ipynb)
3. [`makemore_3.ipynb`](makemore_3.ipynb)
4. [`makemore_4.ipynb`](makemore_4.ipynb)
5. [`makemore_5.ipynb`](makemore_5.ipynb)

They are written as reminders for my future self. The important thing is not
only what the code does, but why each change makes the model more useful.

## 1. What is makemore?

Makemore is a character-level language-modeling project. We give the model a
list of names and ask it to learn the patterns in those names. After training,
it generates new names by predicting one character at a time.

The model does not understand the meaning of a name. It learns statistical
patterns such as:

- which letters often begin a name,
- which characters commonly follow one another,
- which sequences tend to appear near the end, and
- how a longer context can help predict the next character.

The generated names are samples from those learned patterns. They can look
name-like without being real names from the dataset.

## 2. Shared dataset and vocabulary

All five parts use `names.txt`, which contains one name per line. The notebook
loads it with:

```python
words = open("names.txt", "r").read().splitlines()
```

The dataset contains 32,033 names.

The model uses lowercase letters plus `.` as a special token. The dot marks
both the beginning and the end of a name:

```text
. emma .
```

For `emma`, the neighboring character pairs are:

```text
. -> e
e  -> m
m  -> m
m  -> a
a  -> .
```

There are 27 possible symbols in total: 26 letters and the special dot.
`stoi` maps a character to an integer, while `itos` maps an integer back to a
character:

```python
stoi["a"]  # character -> integer
itos[1]    # integer -> character
```

The dot receives index `0`, which makes a context filled with zeros represent
the beginning of a name.

## 3. Part 1: the bigram model

Notebook: [`makemore_1.ipynb`](makemore_1.ipynb)

### 3.1 Counting character transitions

A **bigram** is a pair of neighboring characters. The first notebook counts
how often each character is followed by every other character.

The counts are stored in a `27 x 27` tensor:

```python
N = torch.zeros((27, 27), dtype=torch.int32)
```

The rows represent the current character, and the columns represent the next
character. Therefore, `N[ix1, ix2]` tells us how many times `ix2` followed
`ix1`.

### 3.2 Converting counts into probabilities

The count table is normalized row by row:

```python
P = (N + 1).float()
P /= P.sum(1, keepdim=True)
```

Adding `1` is simple smoothing. It prevents unseen transitions from having
exactly zero probability. This is useful because zero probabilities cannot be
sampled and make `log(0)` invalid.

Each row of `P` is now a probability distribution for the next character.

### 3.3 Generating names

Generation starts with the dot token. The model repeatedly:

1. Looks up the probability row for the current character.
2. Samples the next character with `torch.multinomial`.
3. Appends it to the output.
4. Uses it as the next current character.
5. Stops when it samples the ending dot.

Sampling is random. A fixed random seed makes an experiment repeatable, but
different seeds produce different names.

### 3.4 Negative log-likelihood

The model is evaluated by checking how much probability it gives to the real
transitions in the dataset. For every transition, we take its log-probability
and add the results together.

Because training normally minimizes a value, we use negative log-likelihood:

```text
NLL = -log-likelihood
```

A lower average NLL means the model assigns higher probability to the observed
character transitions.

### 3.5 The neural-network version

Part 1 then expresses the same bigram model with a weight matrix:

```python
W = torch.randn((27, 27), requires_grad=True)
```

The input character becomes a one-hot vector. Multiplying it by `W` produces
27 logits, one for each possible next character. Exponentiating and
normalizing the logits gives probabilities.

The model is trained with:

1. a forward pass,
2. negative log-likelihood,
3. `loss.backward()`, and
4. a gradient-descent update.

This shows the connection between a count table and a neural network: both
learn a distribution over the next character, but the neural-network version
learns its values through optimization.

## 4. Part 2: embeddings and a basic MLP

Notebook: [`makemore_2.ipynb`](makemore_2.ipynb)

Part 2 moves beyond bigrams. Instead of using only the current character, the
model uses the previous **three characters** as context.

For example:

```text
context: . . e  ->  target: m
```

### 4.1 Building context/target examples

The notebook uses `block_size = 3`. For every name, it keeps a sliding window
of three characters and uses the next character as the target.

If the name is `emma`, examples begin like this:

```text
. . .  -> e
. . e  -> m
. e m  -> m
e m m  -> a
m m a  -> .
```

The dataset is split into:

- 80% training data,
- 10% development/validation data, and
- 10% test data.

The training split updates the model. The development split helps compare
settings while building the model. The test split should be used as a final
unbiased check.

### 4.2 Character embeddings

Instead of representing a character only as a one-hot vector, the model learns
a compact vector for each character:

```python
C = torch.randn((27, 10))
```

Here, every one of the 27 symbols has a 10-dimensional embedding. Characters
that behave similarly can learn similar locations in this embedding space.

The three embeddings are concatenated into 30 inputs:

```text
3 characters x 10 features = 30 inputs
```

### 4.3 MLP architecture

The basic model in part 2 contains:

```text
Embedding: 27 -> 10 features per character
Flatten:   3 x 10 -> 30 features
Hidden:    30 -> 200 neurons with tanh
Output:    200 -> 27 logits
```

The logits are compared with the target character using cross-entropy loss.
The model is trained with minibatches, backpropagation, and gradient descent.
The learning rate is reduced during training so the updates become smaller
near the end.

## 5. Part 3: manual layers and batch normalization

Notebook: [`makemore_3.ipynb`](makemore_3.ipynb)

Part 3 focuses on understanding what happens inside a neural network instead
of relying on a large high-level model abstraction.

### 5.1 Manual network components

The notebook organizes the model from basic operations such as:

- an embedding lookup,
- flattening the context,
- a linear layer,
- `tanh` activation, and
- the final output layer.

The layers are kept as simple Python objects so their parameters and forward
passes are visible.

### 5.2 Gradient checking

Backpropagation gives analytical gradients. To check that those gradients are
correct, the notebook compares them with numerical gradients.

The numerical approximation for a parameter is based on slightly changing it
in both directions:

```text
numerical gradient ≈ (loss(x + h) - loss(x - h)) / (2h)
```

The analytical and numerical values should be very close. This is a practical
way to find mistakes in a backward pass.

The notebook also checks gradients for operations such as matrix
multiplication and batch normalization.

### 5.3 Batch normalization

Batch normalization normalizes activations in a minibatch. For an activation
value `x`, it approximately performs:

```text
xhat = (x - mean) / sqrt(variance + epsilon)
output = gain * xhat + bias
```

The learnable `gain` and `bias` parameters allow the network to choose the
most useful scale and offset after normalization.

Batch normalization can make training more stable by keeping activation
values in a more controlled range.

### 5.4 Training and evaluation modes

Batch normalization behaves differently during training and evaluation:

- During training, it can use statistics from the current minibatch.
- During evaluation, it should use running estimates collected during
  training.

Part 3 introduces the need for a `training` flag and separate evaluation
logic. This distinction matters when calculating validation loss or sampling
from the final model.

## 6. Part 4: improving batch normalization

Notebook: [`makemore_4.ipynb`](makemore_4.ipynb)

Part 4 improves the batch-normalization implementation and the training
workflow.

### 6.1 Running statistics

Instead of depending on one final minibatch, batch normalization keeps running
estimates of the mean and variance. These estimates are updated during
training and used when the model is evaluated.

The notebook also calibrates the batch-normalization statistics by passing the
training data through the model after training. This gives the evaluation
path better estimates of the distribution seen by the network.

### 6.2 Better initialization and diagnostics

Part 4 continues improving parameter initialization and includes utilities for
comparing gradients and checking the scale of parameter updates.

Useful diagnostics include:

- comparing analytical and numerical gradients,
- comparing the size of updates with the size of parameters,
- tracking training and validation loss, and
- checking whether the model is overfitting.

These checks make it easier to tell whether a problem comes from the model,
the data, or the optimization settings.

### 6.3 Evaluation

The notebook uses a `split_loss` helper to evaluate the model on different
splits without tracking gradients. The model is put into evaluation mode
before calculating the final train and validation losses.

This is an important improvement over looking only at minibatch training loss:
the validation loss gives a better idea of how well the model generalizes to
names it did not use for parameter updates.

## 7. Part 5: reusable layers and a deeper model

Notebook: [`makemore_5.ipynb`](makemore_5.ipynb)

Part 5 turns the ideas from the earlier notebooks into reusable layer classes
and builds a deeper model.

### 7.1 Reusable layer classes

The notebook defines small components with a common pattern:

- `Linear`
- `BatchNorm1d`
- `Tanh`
- `Embedding`
- `FlattenConsecutive`
- `Sequential`

Each layer has a forward call and a `parameters()` method. `Sequential`
collects the parameters from all of its layers, which makes the training loop
shorter and easier to reuse.

### 7.2 Longer context

Part 5 uses `block_size = 8`, so the model looks at eight previous characters
when predicting the next one.

A longer context gives the model more information, but it also increases the
number of values that must be processed. Simply flattening all eight
embeddings into one huge layer would make the model unnecessarily large.

### 7.3 Hierarchical processing

The model uses `FlattenConsecutive` to combine characters in groups. It first
combines neighboring pairs, then combines the resulting representations again.
This creates a hierarchical structure:

```text
8 characters
    ↓
4 groups of 2
    ↓
2 groups of 2
    ↓
1 final representation
```

This is similar to the way a convolutional or hierarchical model can build
larger patterns from smaller local patterns.

The final model includes:

- 24-dimensional character embeddings,
- hidden layers with 128 neurons,
- batch normalization,
- `tanh` activations,
- several `Linear` layers, and
- a final output layer with 27 logits.

The notebook also reduces the scale of the final layer's weights so the model
does not begin with overly confident predictions.

### 7.4 Training and results

The model is trained with minibatches of 32 examples, simple SGD, and a
learning-rate step-down later in training. It evaluates train and validation
loss and samples generated names from the trained network.

The notebook records an example performance result for the larger model:

```text
about 76K parameters
training loss: about 1.769
validation loss: about 1.993
```

These numbers are tied to the notebook's exact initialization and training
run. They are useful as a rough reference, not as a guaranteed result.

## 8. The training loop

Although the architecture changes across the five parts, the core training
cycle stays the same:

1. Select a random minibatch from the training split.
2. Run the forward pass.
3. Calculate cross-entropy loss.
4. Clear old gradients.
5. Run `loss.backward()`.
6. Update every parameter with gradient descent.
7. Record loss and diagnostic values.
8. Periodically evaluate the training and validation splits.

The final sampling loop starts with a context of start tokens. It predicts one
character, shifts the context window, and repeats until the end token is
sampled.

## 9. How to run the notebooks

From the repository root:

```bash
source .venv/bin/activate
pip install jupyter torch matplotlib
cd makemore
jupyter notebook
```

Alternatively, open the notebooks in VS Code with the Jupyter extension.

For a clean run:

1. Open a notebook.
2. Select the Python environment containing PyTorch.
3. Restart the kernel.
4. Run all cells from the beginning.

Run the five parts in order when learning the material. The notebooks contain
intermediate experiments and printed outputs, so they are intended to be read
and run as learning documents rather than as production scripts.

## 10. Progress checklist

- [x] Load and inspect the names dataset.
- [x] Add start and end tokens.
- [x] Build a character bigram count table.
- [x] Convert counts into probabilities.
- [x] Generate names from the count-based model.
- [x] Train a simple bigram neural network.
- [x] Build an embedding-based MLP.
- [x] Split data into training, development, and test sets.
- [x] Implement and check neural-network layers manually.
- [x] Add batch normalization and evaluation mode.
- [x] Calibrate running batch-normalization statistics.
- [x] Build reusable layer classes.
- [x] Build a deeper hierarchical model with a longer context.
- [ ] Continue experimenting with convolution-style language models.
- [ ] Improve hyperparameter searches and final test-set reporting.

## 11. Quick reminders

- Run the notebooks from the `makemore` directory so `names.txt` is found.
- Run cells from top to bottom because later cells use earlier variables.
- Start with `makemore_1.ipynb` and move through `makemore_5.ipynb` in order.
- Generated names are random samples, not guaranteed to be real names.
- A fixed random seed makes an experiment repeatable.
- The model predicts characters, not words or meanings.
- Training loss and validation loss measure different things.
- Batch normalization needs different behavior during training and evaluation.
- This code is for learning and experimentation, not production use.
