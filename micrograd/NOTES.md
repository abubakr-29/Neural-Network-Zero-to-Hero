# Micrograd Notes

These notes explain the `micrograd.ipynb` notebook in simple language. They are written for my future self, so I can come back after a long time and understand what I built and why it works.

This is my own learning implementation while following Andrej Karpathy's Neural Networks: Zero to Hero material. It is not the official `micrograd` project.

## 1. The big picture

A neural network is a group of mathematical operations with adjustable numbers called **parameters**.

The learning process is usually:

1. Give the network some input.
2. Let it calculate an output. This is the **forward pass**.
3. Compare the output with the answer and calculate a **loss**.
4. Find how each parameter affected the loss. This is the **backward pass**.
5. Change the parameters a little so the loss becomes smaller.
6. Repeat many times.

Micrograd helps us understand step 4. It records the calculations and calculates gradients for them.

The notebook uses single numbers instead of large arrays. This makes the process slower, but much easier to see.

## 2. What is a derivative?

A derivative tells us how much one number changes when another number changes.

For example:

```python
y = 2 * x
```

If `x` increases by `1`, `y` increases by `2`. Therefore:

$$
\frac{dy}{dx} = 2
$$

A derivative is also called a **slope**.

- Positive slope: increasing the input increases the output.
- Negative slope: increasing the input decreases the output.
- Zero slope: a small input change does not change the output much.

In neural networks, derivatives tell us how to change parameters to improve the prediction.

## 3. Numerical differentiation

The notebook starts with this function:

```python
def f(x):
    return 3 * x**2 - 4 * x + 5
```

Instead of calculating the derivative by hand, we can estimate it by checking two nearby points:

```python
h = 0.000001
(f(x + h) - f(x)) / h
```

The formula is:

$$
\text{slope} \approx \frac{f(x + h) - f(x)}{h}
$$

Here, `h` is a very small change. This method is called a **finite difference**.

Finite differences are useful for checking our gradients. They are not normally used for training large neural networks because they require many extra calculations.

## 4. A computation graph

Consider this expression:

```python
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
```

The calculation happens in steps:

```text
a * b = -6
-6 + c = 4
```

We can draw it as a graph:

```text
a ----┐
      * ----┐
b ----┘      + ---- d
c -----------┘
```

This is called a **computation graph** because each node stores a value and each connection represents a calculation.

To calculate gradients, we start at the final result and move backward through this graph.

## 5. The `Value` class

The `Value` class represents one number in the graph.

```python
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
```

Each property has a job:

### `data`

The actual number.

Example:

```python
x = Value(3.0)
print(x.data)  # 3.0
```

### `grad`

The gradient of the final output with respect to this value.

It starts at `0.0`. After calling `backward()`, it contains the calculated gradient.

If `L` is the final result, then `x.grad` means:

$$
\frac{\partial L}{\partial x}
$$

In plain language: how much does `L` change if `x` changes a little?

### `_prev`

The values that were used to create this value.

For example, if:

```python
c = a * b
```

then `c._prev` contains `a` and `b`.

This lets us move backward through the graph.

### `_op`

The operation that created the value, such as `+`, `*`, `tanh`, or `**2`.

This is mainly useful for understanding and drawing the graph.

### `label`

A human-readable name used in graph drawings. It does not affect the calculation.

### `_backward`

A function that knows how to send the output gradient to the input values.

Every operation creates its own `_backward` function.

## 6. Addition

The notebook implements addition like this:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad

    out._backward = _backward
    return out
```

Suppose:

```python
c = a + b
```

The local derivatives are:

$$
\frac{\partial c}{\partial a} = 1
$$

$$
\frac{\partial c}{\partial b} = 1
$$

The chain rule says:

$$
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial c}
\frac{\partial c}{\partial a}
$$

Because the local derivative is `1`, the gradient passed to both inputs is the output gradient.

The code uses `+=` because a value may be used more than once. Gradients from all paths must be added together.

## 7. Multiplication

The notebook implements multiplication like this:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

If:

```python
c = a * b
```

then:

$$
\frac{\partial c}{\partial a} = b
$$

$$
\frac{\partial c}{\partial b} = a
$$

So the gradient for `a` is the gradient arriving at `c`, multiplied by `b`. The gradient for `b` is the gradient arriving at `c`, multiplied by `a`.

## 8. Powers, negatives, and division

The notebook also supports powers:

```python
x ** power
```

For:

$$
 y = x^n
$$

we use:

$$
\frac{dy}{dx} = n x^{n-1}
$$

The implementation supports integer and floating-point powers:

```python
def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f"**{other}")

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
```

The other operators are built from the existing ones:

```python
-x       # self * -1
x - y    # x + (-y)
x / y    # x * y**-1
```

The methods with an `r` prefix handle the value on the right side:

```python
3 + x    # __radd__
3 * x    # __rmul__
3 - x    # __rsub__
```

This allows normal Python numbers and `Value` objects to work together.

## 9. The `tanh` activation function

A neuron usually calculates a weighted sum and then applies an activation function.

The notebook uses `tanh`:

$$
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

Its derivative is:

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
$$

The notebook first calculates the result as a normal Python number, then stores it in a new `Value`.

```python
def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out
```

`tanh` compresses numbers into the range from `-1` to `1`. It also gives the neuron a nonlinear behavior. Without nonlinear activation functions, several layers would still behave like one simple linear operation.

## 10. The `exp` function

The notebook also implements the exponential function:

```python
def exp(self):
    out = Value(math.exp(self.data), (self,), "exp")

    def _backward():
        self.grad += out.data * out.grad
```

For:

$$
 y = e^x
$$

we know:

$$
\frac{dy}{dx} = e^x = y
$$

The notebook uses `exp` to build `tanh` another way:

$$
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

This is a useful exercise because it shows that larger operations can be built from smaller operations already understood by the autodiff engine.

## 11. The chain rule

The chain rule is the main idea behind backpropagation.

If:

$$
 a \rightarrow b \rightarrow c
$$

then:

$$
\frac{dc}{da}
=
\frac{dc}{db}
\frac{db}{da}
$$

In plain language:

1. Find how the final value changes with the middle value.
2. Find how the middle value changes with the first value.
3. Multiply those effects together.

A computation graph may have many operations. Backpropagation applies this rule repeatedly, starting at the final output and moving toward the inputs.

## 12. Why gradients are added

Consider:

```python
a = Value(3.0)
b = a + a
```

The value `a` is used twice. Both paths affect `b`.

Since:

$$
 b = a + a = 2a
$$

we know:

$$
\frac{db}{da} = 2
$$

When `b.backward()` runs, both uses of `a` send a gradient to `a`. That is why the code uses:

```python
self.grad += ...
```

It must add the contribution from every path instead of replacing the old gradient.

This is one of the most important details in the whole implementation.

## 13. The `backward()` method

The `backward()` method calculates all gradients leading to one final value.

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

### Step 1: Build a topological order

`build_topo()` visits every value that leads to the final value.

A node is added to `topo` only after its parents have been visited. This means the list goes from inputs to output.

Example:

```text
inputs -> intermediate values -> output
```

### Step 2: Start at the output

The final value gets:

```python
self.grad = 1.0
```

This is because:

$$
\frac{\partial L}{\partial L} = 1
$$

### Step 3: Walk backward

The reversed topological order goes from output back to inputs. Each node runs its `_backward()` function and passes its gradient to its parents.

## 14. Drawing the graph

The notebook uses Graphviz to visualize the graph.

`trace(root)` collects:

- Every node connected to `root`.
- Every edge from a parent to a child.

`draw_dot(root)` creates a Graphviz diagram. Each value node displays:

- Its label.
- Its data.
- Its gradient.

Operation nodes display things like `+`, `*`, `tanh`, or `**2`.

Example:

```python
draw_dot(o)
```

The graph is helpful when learning because it shows both directions:

- Forward direction: values are calculated.
- Backward direction: gradients are sent back.

## 15. One neuron

A neuron calculates a weighted sum:

$$
 n = x_1w_1 + x_2w_2 + b
$$

Then it applies an activation function:

$$
 o = \tanh(n)
$$

Where:

- `x1`, `x2`: input values.
- `w1`, `w2`: learnable weights.
- `b`: learnable bias.
- `n`: weighted sum before activation.
- `o`: neuron output.

In the notebook:

```python
x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735878195432)

n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()
```

After `o.backward()`, every input and parameter has a gradient. For example, `w1.grad` tells us how much the output changes when `w1` changes.

## 16. Comparing with PyTorch

The notebook repeats the same neuron calculation using PyTorch:

```python
x1 = torch.Tensor([2.0]).double()
x1.requires_grad_(True)
```

`requires_grad_(True)` tells PyTorch to remember operations involving this tensor.

Then:

```python
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)
o.backward()
```

PyTorch calculates gradients automatically. The purpose of this comparison is not to replace micrograd. It is a correctness check and a way to connect the small implementation to a real deep learning framework.

The important lesson is that the core idea is the same:

- Build a graph during the forward pass.
- Store operations.
- Run backward to calculate gradients.

PyTorch does this for tensors and large models. Micrograd does it for simple scalar values.

## 17. Neurons, layers, and MLPs

### `Neuron`

A `Neuron` owns:

- One weight for each input.
- One bias.
- A `tanh` activation.

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
```

`nin` means **number of inputs**.

The `__call__` method lets us use a neuron like a function:

```python
out = neuron(x)
```

The `parameters()` method returns all learnable values:

```python
return self.w + [self.b]
```

### `Layer`

A layer is a group of neurons. Every neuron in the layer receives the same input, but each neuron has different random parameters.

```python
layer = Layer(nin=3, nout=4)
```

This means:

- Each neuron receives `3` inputs.
- The layer contains `4` neurons.
- The layer produces `4` outputs.

### `MLP`

MLP means **multilayer perceptron**.

```python
n = MLP(3, [4, 4, 1])
```

This means:

- The input has `3` numbers.
- The first layer has `4` neurons.
- The second layer has `4` neurons.
- The final layer has `1` neuron.

The list `[4, 4, 1]` describes the output size of each layer.

The MLP sends the output of one layer into the next layer:

```text
3 inputs -> 4 neurons -> 4 neurons -> 1 output
```

The `parameters()` method collects every weight and bias from every layer so the training loop can update them all.

## 18. Training the MLP

The notebook uses this small dataset:

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]
```

- `xs` contains the input examples.
- `ys` contains the desired answers.
- Each input has three numbers.
- Each target is either `1.0` or `-1.0`.

### Forward pass

```python
ypred = [n(x) for x in xs]
```

The MLP produces one prediction for every input.

### Loss

```python
loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
```

This adds the squared error for every example:

$$
\text{loss} = \sum (\text{prediction} - \text{target})^2
$$

A smaller loss means the predictions are closer to the targets.

### Reset gradients

```python
for p in n.parameters():
    p.grad = 0.0
```

Gradients accumulate with `+=`. If we do not reset them, the next training step will accidentally include gradients from previous steps.

### Backward pass

```python
loss.backward()
```

This calculates how the loss changes with respect to every weight and bias in the MLP.

### Update parameters

```python
for p in n.parameters():
    p.data += -0.01 * p.grad
```

This is gradient descent:

$$
 p \leftarrow p - \text{learning rate} \times \frac{\partial \text{loss}}{\partial p}
$$

The learning rate here is `0.01`.

If a parameter has a positive gradient, decreasing it usually decreases the loss. If it has a negative gradient, increasing it usually decreases the loss. The update rule handles both cases.

### Repeat

The notebook repeats this process for multiple iterations. Ideally:

- The loss becomes smaller.
- The predictions move closer to the target values.

One iteration is often called a **training step** or **optimization step**.

## 19. Important words

### Value

A scalar number that remembers how it was created and can store a gradient.

### Parameter

A number the model is allowed to change during training, such as a weight or bias.

### Gradient

The direction and amount of change of the output with respect to a value.

### Computation graph

A record of values and operations used to calculate an output.

### Forward pass

Calculating an output from the inputs.

### Backward pass

Calculating gradients from the output back to the inputs.

### Loss

A number that measures how wrong the predictions are.

### Activation function

A function such as `tanh` that adds nonlinear behavior to a neuron.

### Learning rate

The size of each parameter update.

### Epoch or iteration

One repeated training step in this small example. In larger projects, an epoch often means one complete pass through the dataset.

## 20. Common mistakes to remember

### Forgetting to reset gradients

Because gradients are added, always set parameter gradients back to zero before a new backward pass.

### Calling backward on the wrong value

Call `backward()` on the final loss or output. That gives the method the complete graph it needs to visit.

### Overwriting gradients instead of adding them

Use `+=` inside backward functions. A value may affect the output through multiple paths.

### Confusing data and grad

- `data` is the value used in the forward calculation.
- `grad` is the derivative calculated during the backward calculation.

Changing `grad` does not directly change `data`. The training loop uses `grad` to decide how to update `data`.

### Using a learning rate that is too large

A large learning rate can make the loss jump around or become larger. A very small learning rate can make learning extremely slow.

### Expecting a notebook variable to exist after restarting

Notebook variables live in the current kernel. After restarting the kernel, run the cells from the beginning in order.

## 21. A complete mental model

When reading the code, remember this story:

1. `Value` stores a number.
2. An operation creates a new `Value` and remembers its parents.
3. The new value also stores a local `_backward` rule.
4. A final output is produced by chaining many values together.
5. `backward()` finds every value in the graph.
6. The final output receives gradient `1.0`.
7. `_backward()` functions run from the output toward the inputs.
8. Every parameter receives a gradient.
9. The optimizer changes each parameter using its gradient.
10. Repeating the process trains the model.

That is the basic idea behind backpropagation in much larger frameworks too.

## 22. What to review later

When returning to this project, follow this order:

1. Read the `Value` constructor.
2. Review `__add__` and `__mul__`.
3. Check the derivative rules for `__pow__`, `tanh`, and `exp`.
4. Read `backward()` slowly, especially `build_topo()`.
5. Draw a small graph such as `a * b + c`.
6. Calculate one gradient by hand.
7. Compare it with the value in `.grad`.
8. Review `Neuron`, `Layer`, and `MLP`.
9. Read the training loop from top to bottom.
10. Check whether the loss decreases and predictions improve.

## 23. Next experiments

Good next steps are:

- Add a `zero_grad()` method.
- Move `Value` into `engine.py`.
- Add tests using finite differences.
- Add more activation functions.
- Add a random seed for repeatable experiments.
- Try different learning rates.
- Try different network sizes.
- Add a validation dataset.
- Compare the final predictions with PyTorch.
