# Makemore Study Notes

These notes explain what is happening in
[`makemore.ipynb`](makemore.ipynb). They are written as a reminder for my
future self, so the explanations focus on the ideas behind the code rather
than only describing what each line does.

## 1. What is makemore?

Makemore is a small character-level language-modeling project. We give the
model a list of names and ask it to learn the patterns in those names. After
training, it can generate new names by predicting one character at a time.

The model does not understand what a name means. It only learns which
characters tend to appear together and in which order.

For example, after seeing many names, the model may learn that:

- `q` is often followed by `u`.
- `m` can be followed by `a`, `e`, or another `m`.
- A name often begins with a few common letters.
- Some letters are more likely to appear near the end of a name.

The generated names are samples from these learned patterns. They may look
name-like, but they do not need to be real names.

## 2. Loading the dataset

The notebook reads the names with:

```python
words = open("names.txt", "r").read().splitlines()
```

`splitlines()` turns the file into a Python list where each item is one name.
The current dataset contains 32,033 names.

The notebook also checks the shortest and longest names. These quick checks
help us understand the data before building the model.

The file is loaded with a relative path, so the notebook expects the current
working directory to be the `makemore` folder. If it cannot find
`names.txt`, check the notebook's working directory first.

## 3. Start and end tokens

The model needs to know where a name starts and where it ends. The notebook
uses `.` as a special token:

```text
. emma .
```

The dot is not a letter in the dataset. It means:

- The first `.` marks the beginning of a name.
- The final `.` marks the end of a name.

For `emma`, the character transitions are:

```text
. -> e
e  -> m
m  -> m
m  -> a
a  -> .
```

There are 26 lowercase letters plus the special `.` token, giving the model
27 possible characters.

The notebook stores the mapping between characters and integer indexes in two
dictionaries:

```python
stoi  # string to integer
itos  # integer to string
```

For example, `stoi["a"]` gives the integer index for `a`, while
`itos[index]` converts that index back into a character.

## 4. What is a bigram?

A bigram is a pair of neighboring items. In this project, the items are
characters.

For a name such as `emma`, we look at every adjacent pair:

```text
.e
em
mm
ma
a.
```

The notebook first stores these pairs in a Python dictionary. The dictionary
keeps a count for each pair, so it can answer questions such as:

- How many times did `e` come before `m`?
- How often did `a` appear at the end of a name?
- What usually follows `.` at the beginning of a name?

## 5. The count table

The counts are then stored in a PyTorch tensor called `N`:

```python
N = torch.zeros((27, 27), dtype=torch.int32)
```

The meaning of the table is:

- Row: the current character.
- Column: the next character.
- Value: how many times that transition appeared.

So `N[ix1, ix2]` is the number of times character `ix2` followed character
`ix1`.

The notebook displays this table as a grid. The row and column labels make it
easier to see common transitions visually.

## 6. Turning counts into probabilities

Counts are useful, but generation needs probabilities. The notebook
normalizes each row:

```python
P = (N + 1).float()
P /= P.sum(1, keepdim=True)
```

After this operation, every row sums to 1. A row can now be treated as a
probability distribution for the next character.

### Why add 1?

The `N + 1` step is simple smoothing. Without it, a transition that never
appeared in the dataset would have probability zero. Zero probabilities are
awkward because:

- The model can never sample that transition.
- `log(0)` is negative infinity.
- A single zero probability can make a likelihood calculation unusable.

Adding one keeps every transition possible, although transitions that appeared
many times still receive much higher probabilities.

## 7. Generating names

Generation begins with the start token, whose index is `0` in the notebook.
The model then repeats this process:

1. Look up the probability row for the current character.
2. Sample one next-character index with `torch.multinomial`.
3. Convert the index back to a character with `itos`.
4. Add the character to the output.
5. Use the new character as the current character.
6. Stop when the model samples `.`.

In pseudocode:

```text
current = start token
while current is not end token:
    next = sample_from_probability_row(current)
    append next to output
    current = next
```

Sampling is random, so the output changes between runs. The notebook uses a
fixed random seed in several places when it needs reproducible results.

## 8. Likelihood and negative log-likelihood

The model can be evaluated by asking how much probability it assigns to the
transitions that actually appear in the dataset.

For every real transition:

1. Look up its probability.
2. Take the logarithm of that probability.
3. Add the log-probabilities together.

This gives the log-likelihood of the dataset under the model.

We usually minimize a loss instead of maximizing a score, so we use negative
log-likelihood:

```text
NLL = -log-likelihood
```

The average negative log-likelihood is easier to compare across datasets of
different sizes. A lower average NLL means the model gives better probability
to the observed transitions.

The logarithm is useful because a sequence probability is a product of many
small probabilities:

```text
P(sequence) = P(first transition) * P(second transition) * ...
```

Logarithms turn multiplication into addition:

```text
log(P(sequence)) = log(P1) + log(P2) + ...
```

## 9. The neural-network version

The notebook next creates a weight matrix:

```python
W = torch.randn((27, 27), requires_grad=True)
```

This matrix has the same basic shape as the count table:

- 27 input characters.
- 27 possible output characters.

Each input character is converted to a one-hot vector. A one-hot vector has a
`1` at the position for the current character and `0`s everywhere else:

```python
xenc = F.one_hot(xs, num_classes=27).float()
```

The one-hot vector is multiplied by `W`:

```python
logits = xenc @ W
```

The result is a set of logits, one for each possible next character. Larger
logits should correspond to characters the model considers more likely.

The notebook converts logits into probabilities by exponentiating and
normalizing:

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
```

This is equivalent to applying softmax. In later notebooks, the code can use
PyTorch's built-in cross-entropy functions for a more numerically stable
implementation.

## 10. Training with gradient descent

The training loop repeats these steps:

1. Run a forward pass to calculate probabilities.
2. Calculate the average negative log-likelihood.
3. Clear the old gradients.
4. Run `loss.backward()` to calculate new gradients.
5. Move the weights in the direction that reduces the loss.

The update in the notebook is written as:

```python
W.data += -50 * W.grad
```

The number `50` is the learning rate used in this experiment. It controls the
size of each update:

- Too small: training improves slowly.
- Too large: training can become unstable.
- A useful value: the loss generally decreases over time.

The notebook also adds a small regularization term based on `W**2`. This
discourages unnecessarily large weights.

The direct `W.data` update is used because it keeps the lesson close to the
mathematics. A future version can use a more structured optimizer and update
the parameters inside a safer `torch.no_grad()` block.

## 11. Why the two models are related

The first model uses a table of observed counts. The second model learns a
weight table through optimization.

They are not unrelated approaches:

- The count table is a direct estimate from the data.
- The neural network learns values that produce similar next-character
  probabilities.
- One-hot inputs make each row of `W` correspond to one input character.

This makes the bigram model a useful first neural-network example. It is small
enough to inspect while still demonstrating datasets, logits, probabilities,
losses, gradients, and parameter updates.

## 12. How to run the notebook

From the repository root:

```bash
source .venv/bin/activate
pip install jupyter torch matplotlib
cd makemore
jupyter notebook makemore.ipynb
```

Alternatively, open the notebook in VS Code with the Jupyter extension.

For a clean run:

1. Open `makemore.ipynb`.
2. Select the Python environment containing PyTorch.
3. Restart the kernel.
4. Run all cells from the beginning.

The notebook is an interactive learning document. It includes intermediate
outputs and experiments, so it is normal for it to contain more steps than a
small production script.

## 13. Current progress and next steps

- [x] Load and inspect the names dataset.
- [x] Add start and end tokens.
- [x] Build a character bigram count table.
- [x] Convert counts into probabilities.
- [x] Generate names from the count-based model.
- [x] Calculate negative log-likelihood.
- [x] Train the bigram model with a small neural network.
- [ ] Split the data into training, validation, and test sets.
- [ ] Build a multilayer perceptron language model.
- [ ] Learn embeddings and use a larger context.
- [ ] Add batching and improve the optimization process.
- [ ] Explore deeper models and batch normalization.

## 14. Quick reminders

- Run the notebook from the `makemore` directory so `names.txt` is found.
- Run cells from top to bottom because later cells use earlier variables.
- Generated names are random samples, not guaranteed to be real names.
- A fixed random seed makes an experiment repeatable.
- The model predicts characters, not words or meanings.
- This code is for learning and experimentation, not production use.